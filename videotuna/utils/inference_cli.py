"""Shared CLI helpers for VideoTuna inference entrypoints."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from videotuna.cli.inference_options import InferenceRunConfig
from videotuna.settings import get_settings
from videotuna.utils.inference_profile import resolve_inference_profile


def apply_compile_env(compile_flag: bool) -> None:
    """Deprecated: use :func:`inference_settings_session` instead."""
    if get_settings().cpu_mode == "smoke":
        return
    from videotuna.settings import settings_session

    with settings_session(torch_compile=compile_flag):
        pass


def apply_cpu_smoke_env(run_config: InferenceRunConfig | Any) -> None:
    """Deprecated: use :func:`videotuna.settings.inference_settings_session`."""
    _ = run_config


def validate_cpu_offload_flags(run_config: InferenceRunConfig | Any) -> None:
    """Reject GPU VRAM offload on CPU inference; resolve dual offload flag conflicts."""
    from videotuna.utils.device_utils import (
        detect_compute_backend,
        gpu_is_available,
        resolve_cpu_mode,
    )

    if getattr(run_config, "enable_sequential_cpu_offload", False) and getattr(
        run_config, "enable_model_cpu_offload", False
    ):
        logger.warning(
            "Both --enable_sequential_cpu_offload and --enable_model_cpu_offload "
            "were set; using sequential CPU offload (ignoring model offload)."
        )
        run_config.enable_model_cpu_offload = False

    cpu_mode = resolve_cpu_mode(cli_smoke=getattr(run_config, "cpu_smoke", False))
    device = (getattr(run_config, "device", None) or "").strip().lower()
    cpu_inference = (
        cpu_mode in ("smoke", "force")
        or device == "cpu"
        or detect_compute_backend() == "cpu"
        or not gpu_is_available()
    )
    if not cpu_inference:
        return

    offload = (
        getattr(run_config, "enable_sequential_cpu_offload", False)
        or getattr(run_config, "enable_model_cpu_offload", False)
        or getattr(run_config, "memory_preset", None) == "low_vram"
    )
    if offload:
        raise RuntimeError(
            "CPU offload flags (--enable_model_cpu_offload, "
            "--enable_sequential_cpu_offload, --memory-preset low_vram) "
            "require a GPU accelerator to stage weights. "
            "They are not CPU-only inference modes.\n"
            "Install a GPU stack (poetry install --extras cuda) or run "
            "without offload flags."
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


def resolve_offload_mode(run_config: InferenceRunConfig | Any) -> str:
    """Return offload mode string from parsed run config."""
    return resolve_inference_profile(run_config, apply_preset=False).offload_mode


def prepare_cli_inference_config(
    run_config: InferenceRunConfig,
) -> InferenceRunConfig:
    """Validate parallel degrees before config merge."""
    ulysses = run_config.ulysses_degree
    ring = run_config.ring_degree
    if ulysses is not None or ring is not None:
        from videotuna.utils.device_utils import validate_sequence_parallel_degrees

        validate_sequence_parallel_degrees(ulysses, ring)
    return run_config


def prepare_cli_inference_args(args: Any) -> Any:
    """Deprecated: use :func:`prepare_cli_inference_config`."""
    if isinstance(args, InferenceRunConfig):
        return prepare_cli_inference_config(args)
    ulysses = getattr(args, "ulysses_degree", None)
    ring = getattr(args, "ring_degree", None)
    if ulysses is not None or ring is not None:
        from videotuna.utils.device_utils import validate_sequence_parallel_degrees

        validate_sequence_parallel_degrees(ulysses, ring)
    return args


__all__ = [
    "apply_compile_env",
    "apply_cpu_smoke_env",
    "apply_cpu_smoke_limits",
    "prepare_cli_inference_args",
    "prepare_cli_inference_config",
    "resolve_offload_mode",
    "validate_cpu_offload_flags",
]
