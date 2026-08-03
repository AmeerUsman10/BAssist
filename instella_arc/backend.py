"""Inference backends for frozen Instella ARC evaluation.

The Transformers backend supports exact candidate log-probabilities. The
OpenAI-compatible backend exists for vLLM/SGLang deployments and supports
structured generation; it is not used for candidate-scoring claims unless the
server exposes token log-probabilities in a future extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib import request

from .catalog import CHECKPOINTS, CheckpointSpec
from .prompts import Prompt


class BackendError(RuntimeError):
    pass


class GenerationBackend(Protocol):
    def generate(self, prompt: Prompt, *, max_new_tokens: int = 256) -> str: ...


@dataclass(frozen=True)
class LoadPlan:
    checkpoint_key: str = "think"
    quantization: str = "int4"
    dtype: str = "float16"
    device_map: str | Mapping[str, Any] = "auto"
    max_memory: Mapping[int | str, str] | None = None
    offload_folder: str = "outputs/instella_offload"
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True
    revision: str | None = None
    max_context_tokens: int | None = 8192

    @property
    def checkpoint(self) -> CheckpointSpec:
        try:
            return CHECKPOINTS[self.checkpoint_key]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint_key must be one of {sorted(CHECKPOINTS)}"
            ) from exc

    def validate(self) -> None:
        if self.quantization not in {"none", "int8", "int4"}:
            raise ValueError("quantization must be none, int8, or int4")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported dtype")
        if self.max_context_tokens is not None and self.max_context_tokens < 128:
            raise ValueError("max_context_tokens must be at least 128")


@dataclass
class TransformersBackend:
    model: Any
    tokenizer: Any
    spec: CheckpointSpec
    max_context_tokens: int | None = 8192
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: LoadPlan) -> "TransformersBackend":
        plan.validate()
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise BackendError(
                "TransformersBackend requires torch, transformers, accelerate, "
                "and bitsandbytes for quantized loading"
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        quantization_config = None
        if plan.quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif plan.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        checkpoint = plan.checkpoint.repository_id
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=plan.trust_remote_code,
            revision=plan.revision,
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": plan.trust_remote_code,
            "revision": plan.revision,
            "device_map": plan.device_map,
            "low_cpu_mem_usage": plan.low_cpu_mem_usage,
            "offload_folder": plan.offload_folder,
            "torch_dtype": dtype_map[plan.dtype],
        }
        if plan.max_memory is not None:
            kwargs["max_memory"] = dict(plan.max_memory)
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        Path(plan.offload_folder).mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            spec=plan.checkpoint,
            max_context_tokens=plan.max_context_tokens,
            metadata={
                "checkpoint": checkpoint,
                "quantization": plan.quantization,
                "dtype": plan.dtype,
                "revision": plan.revision,
                "device_map": getattr(model, "hf_device_map", None),
            },
        )

    @property
    def input_device(self):
        embedding = self.model.get_input_embeddings()
        return embedding.weight.device

    def render(self, prompt: Prompt) -> str:
        if self.spec.chat_tuned and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                list(prompt.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt.plain_text()

    def _prompt_ids(self, prompt: Prompt) -> list[int]:
        ids = self.tokenizer.encode(self.render(prompt), add_special_tokens=False)
        if self.max_context_tokens is not None and len(ids) > self.max_context_tokens:
            # Preserve the system prefix and the newest evidence/query suffix.
            prefix = min(256, self.max_context_tokens // 4)
            suffix = self.max_context_tokens - prefix
            ids = ids[:prefix] + ids[-suffix:]
        return ids

    def generate(
        self,
        prompt: Prompt,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        import torch

        prompt_ids = self._prompt_ids(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.input_device)
        attention_mask = torch.ones_like(input_ids)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = self.model.generate(**generation_kwargs)
        continuation = generated[0, input_ids.shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=False)

    def score_completions(
        self,
        prompt: Prompt,
        completions: Sequence[str],
    ) -> tuple[float, ...]:
        """Return summed autoregressive log-probability for each completion."""
        import torch
        from torch.nn import functional as F

        if not completions:
            raise ValueError("at least one completion is required")
        prompt_ids = self._prompt_ids(prompt)
        scores: list[float] = []
        for completion in completions:
            target_ids = self.tokenizer.encode(completion, add_special_tokens=False)
            if not target_ids:
                raise ValueError("a candidate completion tokenized to an empty sequence")
            max_context = self.max_context_tokens
            if max_context is not None and len(prompt_ids) + len(target_ids) > max_context:
                prompt_budget = max_context - len(target_ids)
                if prompt_budget < 32:
                    raise ValueError("completion leaves insufficient prompt context")
                local_prompt = prompt_ids[-prompt_budget:]
            else:
                local_prompt = prompt_ids
            sequence = local_prompt + target_ids
            input_ids = torch.tensor(
                [sequence], dtype=torch.long, device=self.input_device
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[0]
            start = len(local_prompt) - 1
            target_logits = logits[start : start + len(target_ids)]
            target_tensor = torch.tensor(
                target_ids, dtype=torch.long, device=target_logits.device
            )
            log_probability = F.log_softmax(target_logits, dim=-1).gather(
                1, target_tensor.unsqueeze(1)
            )[:, 0].sum()
            scores.append(float(log_probability.item()))
        return tuple(scores)


@dataclass
class OpenAICompatibleBackend:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: int = 300

    def generate(self, prompt: Prompt, *, max_new_tokens: int = 256) -> str:
        endpoint = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": list(prompt.messages),
                "temperature": 0.0,
                "max_tokens": max_new_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise BackendError(f"OpenAI-compatible request failed: {exc}") from exc
        try:
            return str(result["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("server returned an unexpected response shape") from exc


@dataclass
class MockBackend:
    """Deterministic backend used only by unit tests."""

    output: str = '<FINAL>{"action":"A1","x":null,"y":null}</FINAL>'
    scores: Mapping[str, float] = field(default_factory=dict)

    def generate(self, prompt: Prompt, *, max_new_tokens: int = 256) -> str:
        del prompt, max_new_tokens
        return self.output

    def score_completions(
        self, prompt: Prompt, completions: Sequence[str]
    ) -> tuple[float, ...]:
        del prompt
        return tuple(float(self.scores.get(value, 0.0)) for value in completions)
