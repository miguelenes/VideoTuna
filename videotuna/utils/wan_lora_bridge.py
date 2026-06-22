"""Bridge Wan 2.1 native Lightning LoRA checkpoints onto Wan 2.2 Diffusers pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict


WAN_NATIVE_LORA_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
DEFAULT_LORA_RANK = 16


def is_native_wan_lora_ckpt(path: str | Path) -> bool:
    """Return True when path looks like a Wan native Lightning LoRA .ckpt."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix not in (".ckpt", ".pt", ".pth"):
        return False
    try:
        state = load_native_wan_lora_state_dict(p)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(state) and any("lora" in k.lower() for k in state)


def load_native_wan_lora_state_dict(ckpt_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load and normalize LoRA tensors from a Wan training checkpoint."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint type in {ckpt_path}")

    lora_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if "lora" not in key.lower():
            continue
        normalized = key
        for prefix in ("denoiser.", "model.", "module."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        lora_state[normalized] = tensor
    if not lora_state:
        raise ValueError(f"No LoRA keys found in {ckpt_path}")
    return lora_state


def _infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            return int(tensor.shape[0])
        if key.endswith(".lora_B.weight"):
            return int(tensor.shape[1])
    return DEFAULT_LORA_RANK


def _remap_native_to_diffusers_keys(
    native_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Best-effort remap from native Wan block names to Diffusers transformer keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in native_state.items():
        new_key = key
        if new_key.startswith("blocks."):
            new_key = "transformer_blocks." + new_key[len("blocks.") :]
        remapped[new_key] = tensor
    return remapped


def apply_native_wan_lora_to_pipeline(
    pipeline: Any,
    ckpt_path: str | Path,
    *,
    lora_scale: float = 1.0,
) -> None:
    """
    Attach Wan 2.1 native LoRA weights to a Wan 2.2 Diffusers pipeline transformer.

    Uses PEFT injection on ``pipeline.transformer``. Unmapped keys are logged;
    at least one tensor must load successfully.
    """
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    rank = _infer_lora_rank(native_state)
    remapped = _remap_native_to_diffusers_keys(native_state)

    transformer = pipeline.transformer
    if not hasattr(transformer, "peft_config") or not transformer.peft_config:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights=True,
            target_modules=WAN_NATIVE_LORA_TARGETS,
        )
        pipeline.transformer = get_peft_model(transformer, lora_config)

    peft_state = {
        k if k.startswith("base_model.model.") else f"base_model.model.{k}": v
        for k, v in remapped.items()
    }
    result = set_peft_model_state_dict(
        pipeline.transformer,
        peft_state,
        adapter_name="default",
    )
    unexpected: list[str] = []
    if isinstance(result, tuple) and len(result) > 1:
        unexpected = list(result[1])
    if unexpected:
        logger.warning(
            "Wan LoRA bridge: {} unexpected keys (first 5): {}",
            len(unexpected),
            unexpected[:5],
        )

    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(["default"], [lora_scale])
    elif hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=lora_scale)

    loaded = sum(
        1
        for name, _ in pipeline.transformer.named_parameters()
        if "lora" in name.lower()
    )
    if loaded == 0:
        raise RuntimeError(
            f"Wan LoRA bridge failed to load any parameters from {ckpt_path}. "
            "Validate on GPU with a domain smoke checkpoint."
        )
    logger.info(
        "Wan LoRA bridge: rank={} loaded_params={} scale={}",
        rank,
        loaded,
        lora_scale,
    )
