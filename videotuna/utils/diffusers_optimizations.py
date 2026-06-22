"""Shared Diffusers pipeline memory and performance optimizations."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

from loguru import logger

from videotuna.utils.inference_cli import resolve_offload_mode
from videotuna.utils.device_utils import gpu_is_available, resolve_inference_device


def apply_diffusers_optimizations(
    pipe: Any,
    args: Any,
    *,
    model_family: Optional[str] = None,
    disable_progress_bar: bool = False,
) -> None:
    """Apply offload, VAE tiling/slicing, QKV fusion, and optional cache APIs."""
    offload = resolve_offload_mode(args)
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "to"):
        if gpu_is_available():
            pipe.to(resolve_inference_device())

    if getattr(args, "enable_vae_slicing", False) and hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
    if getattr(args, "enable_vae_tiling", False):
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    if getattr(args, "fuse_qkv", False) and hasattr(pipe, "fuse_qkv_projections"):
        pipe.fuse_qkv_projections()
        logger.info("Enabled fuse_qkv_projections on pipeline")

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=disable_progress_bar)

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and getattr(args, "enable_attention_cache", False):
        if hasattr(transformer, "enable_cache"):
            transformer.enable_cache()
            logger.info("Enabled transformer attention cache")
        else:
            logger.warning(
                "enable_attention_cache requested but transformer has no enable_cache()"
            )


def transformer_cache_context(pipe: Any):
    """Return a cache context manager when the transformer supports it."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and hasattr(transformer, "cache_context"):
        return transformer.cache_context()
    return nullcontext()
