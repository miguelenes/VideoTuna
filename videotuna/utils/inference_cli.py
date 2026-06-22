"""Shared CLI flags for VideoTuna inference entrypoints."""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from videotuna.utils.memory_presets import apply_memory_preset

_CPU_MODE_ENV = "VIDEOTUNA_CPU_MODE"
_ATTN_BACKEND_ENV = "VIDEOTUNA_ATTN_BACKEND"
_TORCH_COMPILE_ENV = "VIDEOTUNA_TORCH_COMPILE"


def add_standard_inference_flags(
    parser: argparse.ArgumentParser,
    *,
    include_fp8: bool = True,
    include_parallel: bool = True,
    include_compile: bool = True,
    dtype_default: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Register standardized memory/performance flags on *parser*."""
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help=(
            "CPU smoke mode: tiny resolution/steps, eager attention, device=cpu. "
            "For dev/CI only — not for production video generation."
        ),
    )
    parser.add_argument(
        "--device",
        "--gpu-id",
        dest="device",
        type=str,
        default=None,
        help=(
            "Inference device: cpu, cuda, cuda:1, or integer GPU index. "
            "Respects CUDA_VISIBLE_DEVICES remapping."
        ),
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=None,
        help="Fail before model load if selected GPU total VRAM is below this.",
    )
    parser.add_argument(
        "--memory-preset",
        choices=["low_vram", "balanced", "max_speed"],
        default=None,
        help="Named VRAM/performance preset (overrides offload flags when set).",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enable VAE tiling to reduce decode VRAM.",
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        help="Enable VAE slicing to reduce decode VRAM.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Offload model components to CPU between stages (Diffusers-style).",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Sequential CPU offload (lowest VRAM; slower than model offload).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=dtype_default,
        choices=["bf16", "fp16", "fp32"],
        help="Inference compute dtype (bf16, fp16, or fp32 for CPU smoke).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        choices=["auto"],
        help="Multi-GPU device_map for large Diffusers models (experimental).",
    )
    if include_parallel:
        parser.add_argument(
            "--ulysses_degree",
            type=int,
            default=None,
            help="Ulysses sequence-parallel degree (xfuser).",
        )
        parser.add_argument(
            "--ring_degree",
            type=int,
            default=None,
            help="Ring attention parallel degree (xfuser).",
        )
    if include_compile:
        parser.add_argument(
            "--compile",
            action="store_true",
            help="torch.compile the denoiser (sets VIDEOTUNA_TORCH_COMPILE=1).",
        )
    parser.add_argument(
        "--fuse_qkv",
        action="store_true",
        help="Fuse QKV projections on the Diffusers pipeline when supported.",
    )
    parser.add_argument(
        "--enable_attention_cache",
        action="store_true",
        help="Enable transformer attention cache when supported by the pipeline.",
    )
    if include_fp8:
        parser.add_argument(
            "--enable_fp8",
            action="store_true",
            help="Use Hunyuan pre-quantized FP8 DiT weights (requires *_map.pt).",
        )
    return parser


def apply_compile_env(compile_flag: bool) -> None:
    """Set VIDEOTUNA_TORCH_COMPILE before model load when --compile is passed."""
    if os.environ.get(_CPU_MODE_ENV) == "smoke":
        os.environ[_TORCH_COMPILE_ENV] = "0"
        return
    os.environ[_TORCH_COMPILE_ENV] = "1" if compile_flag else "0"


def apply_cpu_smoke_env(args: argparse.Namespace) -> None:
    """Set environment for CPU smoke mode from --cpu-smoke."""
    if not getattr(args, "cpu_smoke", False):
        return
    os.environ[_CPU_MODE_ENV] = "smoke"
    os.environ[_ATTN_BACKEND_ENV] = "eager"
    os.environ[_TORCH_COMPILE_ENV] = "0"


def validate_cpu_offload_flags(args: Any) -> None:
    """Reject GPU VRAM offload flags when running CPU-only inference."""
    from videotuna.utils.device_utils import detect_compute_backend, gpu_is_available, resolve_cpu_mode

    cpu_mode = resolve_cpu_mode(cli_smoke=getattr(args, "cpu_smoke", False))
    device = (getattr(args, "device", None) or "").strip().lower()
    cpu_inference = (
        cpu_mode in ("smoke", "force")
        or device == "cpu"
        or detect_compute_backend() == "cpu"
        or not gpu_is_available()
    )
    if not cpu_inference:
        return

    offload = (
        getattr(args, "enable_sequential_cpu_offload", False)
        or getattr(args, "enable_model_cpu_offload", False)
        or getattr(args, "memory_preset", None) == "low_vram"
    )
    if offload:
        raise RuntimeError(
            "CPU offload flags (--enable_model_cpu_offload, --enable_sequential_cpu_offload, "
            "--memory-preset low_vram) require a GPU accelerator to stage weights. "
            "They are not CPU-only inference modes.\n"
            "Install a GPU stack (poetry install --extras cuda) or run without offload flags."
        )


def apply_cpu_smoke_limits(
    inference_config: DictConfig,
    flow_config: Optional[DictConfig] = None,
) -> None:
    """Cap resolution, frames, and steps for CPU smoke runs."""
    caps = {
        "frames": 2,
        "height": 256,
        "width": 256,
        "num_inference_steps": 4,
        "ddim_steps": 4,
    }
    for key, cap in caps.items():
        current = getattr(inference_config, key, None)
        if current is not None and int(current) > cap:
            logger.warning("CPU smoke: capping {} from {} to {}", key, current, cap)
            inference_config[key] = cap
        elif current is None and key in ("num_inference_steps", "ddim_steps"):
            inference_config[key] = cap

    if getattr(inference_config, "device", None) is None:
        inference_config.device = "cpu"
    if getattr(inference_config, "dtype", None) is None:
        inference_config.dtype = "fp32"

    if flow_config is not None:
        params = flow_config.get("params", OmegaConf.create())
        if OmegaConf.select(params, "height") and int(params.height) > caps["height"]:
            params.height = caps["height"]
        if OmegaConf.select(params, "width") and int(params.width) > caps["width"]:
            params.width = caps["width"]


def resolve_offload_mode(args) -> str:
    """Return offload mode string from parsed args."""
    if getattr(args, "enable_sequential_cpu_offload", False):
        return "sequential"
    if getattr(args, "enable_model_cpu_offload", False):
        return "model"
    return "none"


def prepare_cli_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply memory presets and validate parallel degrees before config merge."""
    apply_cpu_smoke_env(args)
    apply_memory_preset(args)
    validate_cpu_offload_flags(args)
    ulysses = getattr(args, "ulysses_degree", None)
    ring = getattr(args, "ring_degree", None)
    if ulysses is not None or ring is not None:
        from videotuna.utils.device_utils import validate_sequence_parallel_degrees

        validate_sequence_parallel_degrees(ulysses, ring)
    return args
