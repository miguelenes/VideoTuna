"""Shared CLI flags for VideoTuna inference entrypoints."""

from __future__ import annotations

import argparse
import os
from typing import Optional


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
        choices=["bf16", "fp16"],
        help="Inference compute dtype (bf16 or fp16).",
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
    os.environ["VIDEOTUNA_TORCH_COMPILE"] = "1" if compile_flag else "0"


def resolve_offload_mode(args) -> str:
    """Return offload mode string from parsed args."""
    if getattr(args, "enable_sequential_cpu_offload", False):
        return "sequential"
    if getattr(args, "enable_model_cpu_offload", False):
        return "model"
    return "none"
