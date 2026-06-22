"""Shared Diffusers pipeline memory and performance optimizations."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
from loguru import logger

from videotuna.utils.attention import apply_diffusers_attention_backend
from videotuna.utils.device_utils import gpu_is_available, resolve_inference_device
from videotuna.utils.inference_cli import resolve_offload_mode


def apply_diffusers_optimizations(
    pipe: Any,
    args: Any,
    *,
    model_family: Optional[str] = None,
    disable_progress_bar: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    """Apply offload, VAE tiling/slicing, QKV fusion, attention backend, and cache APIs."""
    offload = resolve_offload_mode(args)
    target_device = device or resolve_inference_device(
        getattr(args, "device", None)
    )
    device_map = getattr(args, "device_map", None)

    if device_map == "auto" and offload == "none":
        _apply_device_map(pipe, target_device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "to"):
        if gpu_is_available():
            pipe.to(target_device)

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

    apply_diffusers_attention_backend(pipe)

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


def _apply_device_map(pipe: Any, device: torch.device) -> None:
    """Spread large Diffusers models across visible GPUs (experimental)."""
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError as exc:
        raise RuntimeError(
            "device_map=auto requires accelerate. Install with: poetry install"
        ) from exc

    if not gpu_is_available() or torch.cuda.device_count() < 2:
        logger.warning(
            "device_map=auto requested but fewer than 2 GPUs visible; using single GPU"
        )
        if hasattr(pipe, "to"):
            pipe.to(device)
        return

    main_module = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if main_module is None:
        logger.warning("device_map=auto: no transformer/unet on pipeline; skipping")
        if hasattr(pipe, "to"):
            pipe.to(device)
        return

    max_memory = {str(i): "22GiB" for i in range(torch.cuda.device_count())}
    device_map = infer_auto_device_map(
        main_module,
        max_memory=max_memory,
    )
    dispatched = dispatch_model(main_module, device_map=device_map)
    if hasattr(pipe, "transformer"):
        pipe.transformer = dispatched
    elif hasattr(pipe, "unet"):
        pipe.unet = dispatched
    logger.info("Applied accelerate device_map=auto across {} GPUs", torch.cuda.device_count())


def apply_flow_memory_config(flow: Any, inference_config: Any) -> None:
    """Apply memory/offload settings after from_pretrained for all flow types."""
    flow_name = flow.__class__.__name__
    if flow_name == "DiffusersVideoFlow":
        if flow.pipeline is not None:
            device = resolve_inference_device(getattr(inference_config, "device", None))
            apply_diffusers_optimizations(
                flow.pipeline,
                inference_config,
                model_family=getattr(flow, "model_family", None),
                device=device,
            )
        return

    if flow_name == "HunyuanVideoFlow":
        pipeline = getattr(flow, "pipeline", None)
        if pipeline is not None:
            _apply_hunyuan_pipeline_offload(flow, pipeline, inference_config)
        return

    if flow_name == "WanVideoModelFlow":
        if getattr(inference_config, "enable_model_cpu_offload", False):
            flow.offload_model = True
        elif getattr(inference_config, "enable_sequential_cpu_offload", False):
            flow.offload_model = True
        return

    if flow_name == "StepVideoModelFlow":
        flow.enable_sequential_cpu_offload = bool(
            getattr(inference_config, "enable_sequential_cpu_offload", False)
        )
        flow.enable_model_cpu_offload = bool(
            getattr(inference_config, "enable_model_cpu_offload", True)
        )


def _apply_hunyuan_pipeline_offload(flow: Any, pipeline: Any, inference_config: Any) -> None:
    device = resolve_inference_device(getattr(inference_config, "device", None))
    if getattr(flow, "use_cpu_offload", False) or getattr(
        inference_config, "enable_sequential_cpu_offload", False
    ):
        pipeline.enable_sequential_cpu_offload()
    elif getattr(flow, "use_model_cpu_offload", False) or getattr(
        inference_config, "enable_model_cpu_offload", False
    ):
        pipeline.enable_model_cpu_offload()
    elif gpu_is_available():
        pipeline.to(device)


def transformer_cache_context(pipe: Any):
    """Return a cache context manager when the transformer supports it."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and hasattr(transformer, "cache_context"):
        return transformer.cache_context()
    return nullcontext()
