"""Named memory/performance presets for inference CLI."""

from __future__ import annotations

import argparse
from typing import Literal

MemoryPreset = Literal["low_vram", "balanced", "max_speed"]


def apply_memory_preset(args: argparse.Namespace) -> None:
    """Mutate *args* in place to apply a named memory preset."""
    preset = getattr(args, "memory_preset", None)
    if not preset:
        return

    if preset == "low_vram":
        args.enable_sequential_cpu_offload = True
        args.enable_model_cpu_offload = False
        args.enable_vae_tiling = True
        if getattr(args, "dtype", None) is None:
            args.dtype = "fp16"
    elif preset == "balanced":
        args.enable_model_cpu_offload = True
        args.enable_sequential_cpu_offload = False
        args.enable_vae_tiling = True
        if getattr(args, "dtype", None) is None:
            args.dtype = "bf16"
    elif preset == "max_speed":
        args.enable_model_cpu_offload = False
        args.enable_sequential_cpu_offload = False
        if getattr(args, "dtype", None) is None:
            args.dtype = "bf16"
    else:
        raise ValueError(
            f"Unknown memory preset {preset!r}. "
            "Expected low_vram, balanced, or max_speed."
        )
