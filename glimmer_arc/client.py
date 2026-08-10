"""Minimal loopback-only OpenAI-compatible client with sanitized receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import canonical_sha256, extract_json_object, sha256_text


class ClientError(RuntimeError):
    """A stable provider error that contains no response body."""


@dataclass(frozen=True)
class CallReceipt:
    index: int
    purpose: str
    prompt_sha256: str
    response_sha256: str | None
    parsed_sha256: str | None
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    valid_json: bool
    error_code: str | None
    used_image: bool
    temperature: float
    max_tokens: int


class LlamaClient:
    """Synchronous client for one local llama-server instance.

    Only loopback URLs are accepted.  The class records hashes and final parsed
    objects, never hidden reasoning or unrestricted raw model output.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        *,
        request_timeout: float = 120.0,
        seed: int = 0,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not (
            normalized.startswith("http://127.0.0.1:")
            or normalized.startswith("http://localhost:")
        ):
            raise ValueError("llama client accepts loopback URLs only")
        self.base_url = normalized
        self.request_timeout = float(request_timeout)
        self.seed = int(seed)
        self.model_id: str | None = None
        self.receipts: list[CallReceipt] = []
        self.parsed_objects: list[dict[str, Any]] = []

    def _json_request(
        self,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urlopen(request, timeout=timeout or self.request_timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raise ClientError(f"http_{exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise ClientError("transport_failure") from None
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ClientError("server_json_invalid") from None
        if not isinstance(value, dict):
            raise ClientError("server_json_root_invalid")
        return value

    def health(self) -> dict[str, Any]:
        # /health is outside /v1 in llama-server.
        request = Request(self.base_url.removesuffix("/v1") + "/health", method="GET")
        try:
            with urlopen(request, timeout=min(self.request_timeout, 10.0)) as response:
                raw = response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            raise ClientError("health_unavailable") from None
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = {"raw_sha256": sha256_text(raw.decode("utf-8", errors="replace"))}
        return value if isinstance(value, dict) else {"status": str(value)}

    def discover_model(self) -> str:
        value = self._json_request("/models")
        data = value.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
            raise ClientError("models_response_invalid")
        model_id = data[0].get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ClientError("model_id_missing")
        self.model_id = model_id
        return model_id

    @staticmethod
    def data_uri(path: str | Path) -> str:
        resolved = Path(path)
        suffix = resolved.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
        if mime is None:
            raise ValueError("only PNG/JPEG images are supported")
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        purpose: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        image_path: str | Path | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self.model_id is None:
            self.discover_model()
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens outside safe bound")
        if not math.isfinite(float(temperature)) or not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature outside safe bound")

        if image_path is None:
            user_content: Any = user
        else:
            user_content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": self.data_uri(image_path)}},
            ]
        messages: Sequence[Mapping[str, Any]] = (
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        )
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": 1.0 if temperature == 0.0 else 0.95,
            "top_k": 64,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        prompt_hash = canonical_sha256({"system": system, "user": user, "image": bool(image_path)})
        started = time.perf_counter()
        response_hash: str | None = None
        parsed_hash: str | None = None
        parsed: dict[str, Any] | None = None
        error_code: str | None = None
        usage: Mapping[str, Any] = {}
        try:
            response = self._json_request(
                "/chat/completions", payload=payload, timeout=timeout or self.request_timeout
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise ClientError("choices_invalid")
            message = choices[0].get("message")
            if not isinstance(message, Mapping):
                raise ClientError("message_invalid")
            content = message.get("content")
            if isinstance(content, list):
                fragments = [
                    part.get("text")
                    for part in content
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                ]
                content = "".join(fragments)
            if not isinstance(content, str):
                raise ClientError("content_missing")
            content = content.strip()
            for suffix in ("<|eot|>", "<|end_of_text|>"):
                if content.endswith(suffix):
                    content = content[: -len(suffix)].rstrip()
            response_hash = sha256_text(content)
            parsed = extract_json_object(content)
            parsed_hash = canonical_sha256(parsed)
            raw_usage = response.get("usage")
            usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        except ClientError as exc:
            error_code = str(exc)
        except Exception as exc:
            # Protocol errors are reduced to stable class names; no model text
            # or server body is copied into evidence.
            error_code = type(exc).__name__
        latency = time.perf_counter() - started

        def integer_or_none(value: Any) -> int | None:
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

        receipt = CallReceipt(
            index=len(self.receipts) + 1,
            purpose=purpose,
            prompt_sha256=prompt_hash,
            response_sha256=response_hash,
            parsed_sha256=parsed_hash,
            latency_seconds=latency,
            prompt_tokens=integer_or_none(usage.get("prompt_tokens")),
            completion_tokens=integer_or_none(usage.get("completion_tokens")),
            total_tokens=integer_or_none(usage.get("total_tokens")),
            valid_json=parsed is not None,
            error_code=error_code,
            used_image=image_path is not None,
            temperature=float(temperature),
            max_tokens=max_tokens,
        )
        self.receipts.append(receipt)
        if parsed is None:
            raise ClientError(error_code or "completion_invalid")
        self.parsed_objects.append(parsed)
        return parsed

    def receipt_slice(self, start_index: int = 0) -> list[dict[str, Any]]:
        return [asdict(receipt) for receipt in self.receipts[start_index:]]


__all__ = ["CallReceipt", "ClientError", "LlamaClient"]
