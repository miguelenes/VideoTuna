"""Save Flux LoRA checkpoints in Diffusers-compatible format."""

from __future__ import annotations

from pathlib import Path

from diffusers import FluxPipeline
from peft.utils import get_peft_model_state_dict


def save_lora_checkpoint(transformer, output_dir: str | Path, step: int) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    transformer_lora = get_peft_model_state_dict(transformer)
    FluxPipeline.save_lora_weights(
        save_directory=str(save_path),
        transformer_lora_layers=transformer_lora,
    )
    return save_path
