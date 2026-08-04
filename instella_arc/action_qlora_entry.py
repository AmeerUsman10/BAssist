"""Apply the Instella MoE QLoRA compatibility bridge, then run training."""

from __future__ import annotations

from .qlora_compat import patch_peft_prepare_model_for_kbit_training


def main() -> None:
    patch_peft_prepare_model_for_kbit_training()
    from .action_qlora import main as training_main

    training_main()


if __name__ == "__main__":
    main()
