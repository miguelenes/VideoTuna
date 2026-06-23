"""Multi-GPU launch validation and safe-command generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

MultiGpuMode = Literal["device_map", "xfuser", "wan_lightning", "flux_accelerate"]
DiagnosticSeverity = Literal["info", "warning", "fatal"]


@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    message: str


@dataclass(frozen=True)
class MultiGpuValidationResult:
    success: bool
    diagnostics: tuple[Diagnostic, ...]
    generated_command: str | None = None


@dataclass(frozen=True)
class MultiGpuSpec:
    mode: MultiGpuMode
    gpu_ids: tuple[int, ...] = ()
    ulysses_degree: int = 1
    ring_degree: int = 1
    config_path: str | None = None
    inference_flow: str = "DiffusersVideoFlow"
    extra_args: dict[str, str] = field(default_factory=dict)
    devices: str = "0,"
    num_processes: int = 1
    max_memory_per_gpu: str = "22GiB"
    offload_mode: str = "none"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CUDA_DETECTED: bool | None = None
_GPU_COUNT: int | None = None


def _reset_device_cache() -> None:
    global _CUDA_DETECTED, _GPU_COUNT
    _CUDA_DETECTED = None
    _GPU_COUNT = None


def _cuda_available() -> bool:
    global _CUDA_DETECTED
    if _CUDA_DETECTED is None:
        try:
            import torch  # fmt: skip
            _CUDA_DETECTED = torch.cuda.is_available()
        except Exception:
            _CUDA_DETECTED = False
    return _CUDA_DETECTED


def _gpu_count() -> int:
    global _GPU_COUNT
    if _GPU_COUNT is None:
        if _cuda_available():
            import torch  # fmt: skip
            _GPU_COUNT = torch.cuda.device_count()
        else:
            _GPU_COUNT = 0
    return _GPU_COUNT


def _compute_backend() -> str:
    try:
        from videotuna.utils.device_utils import detect_compute_backend

        return detect_compute_backend()
    except Exception:
        return "unknown"


def _deepspeed_available() -> bool:
    try:
        import deepspeed  # noqa: F401

        return True
    except ImportError:
        return False


def _xfuser_available() -> bool:
    try:
        import xfuser  # noqa: F401

        return True
    except ImportError:
        return False


def _accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


def _nccl_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.nccl.version() is not None
    except Exception:
        return False


def _detected_attn_backend() -> str:
    try:
        from videotuna.utils.attention import get_attn_backend

        return get_attn_backend()
    except Exception:
        return "unknown"


def _visible_gpu_count() -> int:
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ids.strip():
        return _gpu_count()
    return len([x for x in ids.split(",") if x.strip()])


def _gpu_from_devices(devices_str: str) -> int:
    return len([x for x in devices_str.split(",") if x.strip()])


def _format_env_vars(env: dict[str, str]) -> str:
    parts = [f"{k}={v}" for k, v in env.items()]
    return " ".join(parts)


def _fatal(m: str) -> Diagnostic:
    return Diagnostic("fatal", m)


def _warn(m: str) -> Diagnostic:
    return Diagnostic("warning", m)


def _info(m: str) -> Diagnostic:
    return Diagnostic("info", m)


# ---------------------------------------------------------------------------
# Per-mode validators
# ---------------------------------------------------------------------------


def _validate_device_map(spec: MultiGpuSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    if not _cuda_available():
        diags.append(
            _fatal("CUDA is not available. device_map=auto requires a CUDA GPU.")
        )
        return diags

    n_gpu = len(spec.gpu_ids) or _visible_gpu_count()
    if n_gpu < 2:
        diags.append(
            _warn(
                f"Only {n_gpu} GPU{'s' if n_gpu == 1 else ''} visible. "
                "device_map=auto benefits from 2+ GPUs."
            )
        )
    else:
        diags.append(
            _info(
                f"Detected {n_gpu} GPU{'s' if n_gpu > 1 else ''} "
                "for device_map=auto."
            )
        )

    if spec.offload_mode != "none":
        diags.append(
            _fatal(
                "device_map=auto requires offload_mode='none' "
                f"but '{spec.offload_mode}' was set. "
                "CPU offload and device_map=auto are mutually exclusive."
            )
        )

    backend = _compute_backend()
    if backend == "rocm":
        diags.append(
            _info(
                "device_map=auto works on AMD ROCm "
                "(xfuser is blocked, but Diffusers dispatch is fine)."
            )
        )

    if not _accelerate_available():
        diags.append(
            _fatal(
                "device_map=auto requires accelerate. "
                "Install with: poetry install -E cuda"
            )
        )

    attn = _detected_attn_backend()
    if attn == "eager":
        diags.append(
            _warn(
                "Attention backend is 'eager'. For better performance, "
                "set VIDEOTUNA_ATTN_BACKEND=flash or sdpa."
            )
        )

    try:
        int(spec.max_memory_per_gpu.replace("GiB", ""))
    except (ValueError, AttributeError):
        diags.append(
            _warn(
                f"max_memory_per_gpu={spec.max_memory_per_gpu!r} "
                "does not look like a valid size (e.g. '22GiB')."
            )
        )

    return diags


def _validate_xfuser(spec: MultiGpuSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    if not _cuda_available():
        diags.append(_fatal("CUDA is not available. xfuser USP requires NVIDIA CUDA."))
        return diags

    backend = _compute_backend()
    if backend == "rocm":
        diags.append(
            _fatal(
                "xfuser USP is not supported on AMD ROCm. "
                "Use single-GPU Diffusers inference "
                "with VIDEOTUNA_ATTN_BACKEND=sdpa instead."
            )
        )

    n_gpu = len(spec.gpu_ids) or _visible_gpu_count()
    if n_gpu < 2:
        diags.append(
            _fatal(
                f"xfuser USP requires at least 2 GPUs " f"but only {n_gpu} are visible."
            )
        )
        return diags

    product = spec.ulysses_degree * spec.ring_degree
    if product != n_gpu:
        diags.append(
            _fatal(
                f"ulysses_degree ({spec.ulysses_degree}) "
                f"× ring_degree ({spec.ring_degree}) = {product} "
                f"but {n_gpu} GPU{'s' if n_gpu > 1 else ''} available. "
                "Launch with torchrun --nproc_per_node=N "
                "where N equals the product."
            )
        )

    if spec.offload_mode != "none":
        diags.append(
            _fatal(
                f"xfuser USP does not support CPU offload "
                f"(offload_mode='{spec.offload_mode}'). "
                "The model must fit entirely in GPU VRAM across all ranks."
            )
        )

    if not _xfuser_available():
        diags.append(
            _fatal("xfuser is not installed. " "Install with: poetry install -E cuda")
        )

    if not _nccl_available():
        diags.append(
            _warn(
                "NCCL check failed or unavailable. "
                "xfuser USP depends on NCCL for collective communication. "
                "Set NCCL_DEBUG=INFO to diagnose."
            )
        )
    else:
        diags.append(_info("NCCL is available for xfuser collective communication."))

    attn = _detected_attn_backend()
    if attn == "eager":
        diags.append(
            _warn(
                "Attention backend is 'eager'. "
                "For sequence-parallel performance, "
                "set VIDEOTUNA_ATTN_BACKEND=flash."
            )
        )

    return diags


def _validate_wan_lightning(spec: MultiGpuSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    if not _cuda_available():
        diags.append(
            _fatal(
                "CUDA is not available. " "Wan Lightning training requires a CUDA GPU."
            )
        )
        return diags

    backend = _compute_backend()
    if backend == "rocm":
        diags.append(
            _fatal(
                "Wan training requires NVIDIA CUDA "
                "(ROCm is inference + Flux training only). "
                "See AGENTS.md for supported profiles."
            )
        )

    n_gpu = len(spec.gpu_ids) or _visible_gpu_count()
    requested = _gpu_from_devices(spec.devices)
    if requested > n_gpu:
        diags.append(
            _fatal(
                f"--devices '{spec.devices}' requests {requested} "
                f"GPU{'s' if requested > 1 else ''} "
                f"but only {n_gpu} GPU{'s' if n_gpu > 1 else ''} "
                "are visible."
            )
        )
    else:
        diags.append(
            _info(
                f"Training requested on {requested} of {n_gpu} "
                f"visible GPU{'s' if n_gpu > 1 else ''}."
            )
        )

    if not _deepspeed_available():
        diags.append(
            _warn(
                "DeepSpeed is not installed. "
                "The config uses strategy=deepspeed_stage_3_offload. "
                "Install with: poetry run install-deepspeed"
            )
        )
    else:
        diags.append(_info("DeepSpeed is available for ZeRO-3 offload."))

    if not _nccl_available():
        diags.append(
            _warn(
                "NCCL unavailable or check failed. "
                "Multi-GPU training depends on NCCL for gradient sync."
            )
        )

    attn = _detected_attn_backend()
    if attn == "eager":
        diags.append(
            _warn(
                "Attention backend is 'eager'. For training speed, "
                "set VIDEOTUNA_ATTN_BACKEND=flash or sdpa."
            )
        )

    return diags


def _validate_flux_accelerate(spec: MultiGpuSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    if not _cuda_available():
        diags.append(
            _fatal(
                "CUDA is not available. "
                "Flux Accelerate training requires a CUDA GPU."
            )
        )
        return diags

    backend = _compute_backend()
    if backend == "rocm":
        diags.append(
            _info(
                "Flux training with Accelerate works on ROCm " "(bf16 mixed precision)."
            )
        )

    n_gpu = len(spec.gpu_ids) or _visible_gpu_count()
    if spec.num_processes > n_gpu:
        diags.append(
            _fatal(
                f"--num_processes={spec.num_processes} exceeds {n_gpu} "
                f"visible GPU{'s' if n_gpu > 1 else ''}."
            )
        )
    else:
        diags.append(
            _info(
                f"Flux training requested with {spec.num_processes} "
                f"process{'es' if spec.num_processes > 1 else ''} on "
                f"{n_gpu} visible GPU{'s' if n_gpu > 1 else ''}."
            )
        )

    if not _accelerate_available():
        diags.append(
            _fatal(
                "accelerate is not installed. " "Install with: poetry install -E cuda"
            )
        )

    if not _nccl_available() and spec.num_processes > 1:
        diags.append(
            _warn(
                "NCCL unavailable or check failed. "
                "Multi-process accelerate depends on NCCL."
            )
        )

    attn = _detected_attn_backend()
    if attn == "eager" and spec.num_processes > 1:
        diags.append(
            _warn(
                "Attention backend is 'eager'. "
                "Consider VIDEOTUNA_ATTN_BACKEND=sdpa for training speed."
            )
        )

    return diags


# ---------------------------------------------------------------------------
# Command generators
# ---------------------------------------------------------------------------


def _gpu_ids_str(spec: MultiGpuSpec) -> str:
    if spec.gpu_ids:
        return ",".join(str(i) for i in spec.gpu_ids)
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if ids.strip():
        return ids
    return ",".join(str(i) for i in range(_gpu_count()))


def generate_device_map_command(spec: MultiGpuSpec) -> str:
    gpu_str = _gpu_ids_str(spec)
    config = spec.config_path or "configs/inference/presets/balanced_wan2_2_720p.yaml"
    cmd = (
        f"poetry run inference-wan2.2-t2v-720p"
        f" --config {config}"
        f" --device-map auto"
        f" --max-memory-per-gpu {spec.max_memory_per_gpu}"
    )
    for k, v in sorted(spec.extra_args.items()):
        cmd += f" --{k} {v}"
    env = {"CUDA_VISIBLE_DEVICES": gpu_str}
    backend = _compute_backend()
    if backend == "rocm":
        env["VIDEOTUNA_ATTN_BACKEND"] = "sdpa"
    return f"{_format_env_vars(env)} {cmd}"


def generate_xfuser_command(spec: MultiGpuSpec) -> str:
    gpu_str = _gpu_ids_str(spec)
    nproc = len(spec.gpu_ids) or _visible_gpu_count()
    config = spec.config_path or "configs/inference/presets/wan_domain_lora_smoke.yaml"
    cmd = (
        f"torchrun --nproc_per_node={nproc} scripts/inference_new.py"
        f" --config {config}"
        f" --ulysses_degree {spec.ulysses_degree}"
        f" --ring_degree {spec.ring_degree}"
    )
    for k, v in sorted(spec.extra_args.items()):
        cmd += f" --{k} {v}"
    env = {
        "CUDA_VISIBLE_DEVICES": gpu_str,
        "NCCL_DEBUG": "INFO",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    return f"{_format_env_vars(env)} {cmd}"


def generate_wan_lightning_command(spec: MultiGpuSpec) -> str:
    gpu_str = _gpu_ids_str(spec)
    config = spec.config_path or "configs/domain/wan_t2v_lora.yaml"
    cmd = (
        f"poetry run train-domain-t2v"
        f" --config {config}"
        f" --devices '{spec.devices}'"
    )
    for k, v in sorted(spec.extra_args.items()):
        cmd += f" --{k} {v}"
    env = {"CUDA_VISIBLE_DEVICES": gpu_str}
    requested = _gpu_from_devices(spec.devices)
    if requested > 1:
        env["NCCL_DEBUG"] = "INFO"
    return f"{_format_env_vars(env)} {cmd}"


def generate_flux_accelerate_command(spec: MultiGpuSpec) -> str:
    gpu_str = _gpu_ids_str(spec)
    config = spec.config_path or "configs/domain/flux_t2i.json"
    data_config = "configs/domain/flux_t2i_data.json"
    cmd = (
        f"accelerate launch"
        f" --mixed_precision=bf16"
        f" --num_processes={spec.num_processes}"
        f" --num_machines=1"
        f" scripts/train_flux_lora.py"
        f" --config_path {config}"
        f" --data_config_path {data_config}"
    )
    for k, v in sorted(spec.extra_args.items()):
        cmd += f" --{k} {v}"
    env = {"CUDA_VISIBLE_DEVICES": gpu_str}
    return f"{_format_env_vars(env)} {cmd}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALIDATORS: dict[MultiGpuMode, Any] = {
    "device_map": _validate_device_map,
    "xfuser": _validate_xfuser,
    "wan_lightning": _validate_wan_lightning,
    "flux_accelerate": _validate_flux_accelerate,
}

_GENERATORS: dict[MultiGpuMode, Any] = {
    "device_map": generate_device_map_command,
    "xfuser": generate_xfuser_command,
    "wan_lightning": generate_wan_lightning_command,
    "flux_accelerate": generate_flux_accelerate_command,
}


def validate_multi_gpu_setup(
    spec: MultiGpuSpec,
) -> MultiGpuValidationResult:
    """Run all validation checks for the given multi-GPU spec."""
    validator = _VALIDATORS.get(spec.mode)
    if validator is None:
        return MultiGpuValidationResult(
            success=False,
            diagnostics=(_fatal(f"Unknown multi-GPU mode: {spec.mode!r}"),),
        )

    diags = validator(spec)
    fatal_count = sum(1 for d in diags if d.severity == "fatal")
    success = fatal_count == 0

    command: str | None = None
    if success:
        generator = _GENERATORS.get(spec.mode)
        if generator is not None:
            command = generator(spec)

    for d in diags:
        log_method = (
            logger.error
            if d.severity == "fatal"
            else (logger.warning if d.severity == "warning" else logger.info)
        )
        log_method("[{}] {}", d.severity.upper(), d.message)

    return MultiGpuValidationResult(
        success=success,
        diagnostics=tuple(diags),
        generated_command=command,
    )


def generate_launch_command(spec: MultiGpuSpec) -> str:
    """Generate a safe multi-GPU launch command without validation."""
    generator = _GENERATORS.get(spec.mode)
    if generator is None:
        raise ValueError(f"Unknown multi-GPU mode: {spec.mode!r}")
    return generator(spec)


_FAILURE_TABLE: dict[str, list[str]] = {
    "hang": [
        "ulysses_degree × ring_degree does not match WORLD_SIZE.",
        "Set NCCL_DEBUG=INFO before launching to see NCCL diagnostics.",
        "Ensure torchrun --nproc_per_node=N matches "
        "the product of --ulysses_degree and --ring_degree.",
    ],
    "oom": [
        "Model loaded on all ranks without broadcast.",
        "Reduce batch size or enable gradient checkpointing.",
        "Check that offload is disabled for xfuser USP " "(offload is incompatible).",
    ],
    "xfuser_import_error": [
        "xfuser is an optional CUDA dependency.",
        "Install with: poetry install -E cuda",
    ],
    "xfuser_rocm": [
        "xfuser requires NVIDIA CUDA (blocked on AMD ROCm).",
        "Use single-GPU Diffusers inference:",
        "CUDA_VISIBLE_DEVICES=0 VIDEOTUNA_ATTN_BACKEND=sdpa "
        "poetry run inference-wan2.2-t2v-720p --config <preset>",
    ],
    "deepspeed_import_error": [
        "DeepSpeed is an optional dependency for Wan LoRA training.",
        "Install with: poetry run install-deepspeed",
    ],
    "nccl_timeout": [
        "NCCL operations are timing out.",
        "Set NCCL_TIMEOUT=1800 (seconds) for large-model " "first-time initialization.",
        "Check NCCL_DEBUG=INFO output for peer-to-peer " "connectivity issues.",
    ],
    "device_map_cpu_offload": [
        "device_map=auto and CPU offload are mutually exclusive.",
        "Remove --enable_model_cpu_offload / --enable_sequential_cpu_offload "
        "when using --device-map auto.",
    ],
}


def diagnose_failure(symptom: str) -> list[str]:
    """Return troubleshooting steps for a known failure symptom."""
    symptom_lower = symptom.lower().replace(" ", "_")
    for keyword, steps in _FAILURE_TABLE.items():
        if keyword in symptom_lower:
            return steps[:]
    return [
        f"No known diagnostic for symptom: {symptom}",
        "Run 'validate-multi-gpu diagnose <symptom>' with "
        "a more specific term "
        "(e.g. 'hang', 'oom', 'xfuser', 'rocm', "
        "'deepspeed', 'nccl', 'device_map').",
    ]


__all__ = [
    "Diagnostic",
    "MultiGpuMode",
    "MultiGpuSpec",
    "MultiGpuValidationResult",
    "validate_multi_gpu_setup",
    "generate_launch_command",
    "diagnose_failure",
]
