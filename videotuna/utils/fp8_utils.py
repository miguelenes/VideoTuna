"""FP8 validation helpers for Hunyuan inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


def fp8_dtype_available() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def fp8_map_path(dit_weight: str) -> str:
    return dit_weight.replace(".pt", "_map.pt")


def validate_fp8_inference(
    dit_weight: str,
    *,
    require_map: bool = True,
) -> None:
    """
    Validate runtime and checkpoint prerequisites for Hunyuan FP8 inference.

    Raises:
        RuntimeError: if PyTorch float8 or the FP8 scale map is unavailable.
    """
    if not fp8_dtype_available():
        raise RuntimeError(
            "FP8 inference requires torch.float8_e4m3fn (PyTorch 2.6+). "
            f"Current torch: {torch.__version__}"
        )

    try:
        import torchao  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "FP8 inference requires torchao (poetry dependency). "
            "Install with: poetry install"
        ) from exc

    if not require_map:
        return

    if not dit_weight:
        raise ValueError("dit_weight must be set when --enable_fp8 is used.")

    map_path = fp8_map_path(dit_weight)
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"FP8 scale map not found: {map_path}. "
            "Hunyuan FP8 weights require a companion *_map.pt file beside the DiT checkpoint."
        )

    logger.info(f"FP8 map found: {map_path}")


def precision_from_dtype_flag(dtype_flag: Optional[str], default: str = "bf16") -> str:
    """Map CLI --dtype bf16|fp16 to Hunyuan precision string."""
    if dtype_flag is None:
        return default
    mapping = {"bf16": "bf16", "fp16": "fp16"}
    if dtype_flag not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_flag!r}; expected bf16 or fp16.")
    return mapping[dtype_flag]
