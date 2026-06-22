"""FP8 validation helpers for Hunyuan inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from videotuna.utils.device_utils import detect_compute_backend, gpu_is_available


def require_nvidia_cuda() -> None:
    """Fail when the active backend is not NVIDIA CUDA."""
    backend = detect_compute_backend()
    if backend != "cuda":
        raise RuntimeError(
            f"NVIDIA CUDA is required but detected backend is {backend!r}."
        )


def _fp8_min_compute_capability() -> tuple[int, int]:
    """Ada Lovelace (sm 8.9) minimum for FP8 tensor cores in practice."""
    return (8, 9)


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
    if detect_compute_backend() == "cpu":
        raise RuntimeError(
            "FP8 inference (--enable_fp8) is not supported on CPU. "
            "Use --dtype fp32 or fp16 for CPU smoke runs."
        )

    if detect_compute_backend() == "rocm":
        raise RuntimeError(
            "FP8 inference (--enable_fp8) is not supported on AMD ROCm. "
            "Use --dtype bf16 with CPU offload instead."
        )

    require_nvidia_cuda()
    if gpu_is_available():
        major, minor = torch.cuda.get_device_capability(0)
        min_major, min_minor = _fp8_min_compute_capability()
        if (major, minor) < (min_major, min_minor):
            raise RuntimeError(
                f"FP8 inference requires NVIDIA GPU compute capability >= "
                f"{min_major}.{min_minor} (Ada/Hopper); detected {major}.{minor}."
            )

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
