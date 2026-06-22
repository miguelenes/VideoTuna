"""Device detection and inference hardware requirements."""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.version
from loguru import logger

ComputeBackend = Literal["cuda", "rocm", "cpu", "mps"]

_COMPUTE_BACKEND_ENV = "VIDEOTUNA_COMPUTE_BACKEND"

_STEPVIDEO_FLOW = "videotuna.flow.stepvideo.StepVideoModelFlow"

# Flows that need a GPU for practical 720p video generation.
_GPU_REQUIRED_FLOW_TARGETS = (
    "videotuna.flow.hunyuanvideo.HunyuanVideoFlow",
    "videotuna.flow.wanvideo.WanVideoModelFlow",
    _STEPVIDEO_FLOW,
)


def _torch_hip_version() -> str | None:
    hip = getattr(torch.version, "hip", None)
    if hip is None:
        return None
    return str(hip)


def _detect_compute_backend_raw() -> ComputeBackend:
    if not torch.cuda.is_available():
        return "cpu"
    if _torch_hip_version() is not None:
        return "rocm"
    return "cuda"


def detect_compute_backend() -> ComputeBackend:
    """Return the active compute backend (cuda, rocm, cpu, or mps)."""
    requested = os.environ.get(_COMPUTE_BACKEND_ENV, "auto").strip().lower()
    if requested == "auto":
        return _detect_compute_backend_raw()
    if requested not in ("cuda", "rocm", "cpu", "mps"):
        raise ValueError(
            f"Invalid {_COMPUTE_BACKEND_ENV}={requested!r}. "
            "Expected auto, cuda, rocm, cpu, or mps."
        )
    if requested == "mps":
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested == "rocm":
        if _torch_hip_version() is None:
            raise RuntimeError(
                f"VIDEOTUNA_COMPUTE_BACKEND=rocm but PyTorch was not built with HIP. "
                f"Detected: {describe_compute_environment()}\n"
                "Install with: poetry install --extras rocm"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VIDEOTUNA_COMPUTE_BACKEND=rocm but no ROCm GPU is visible. "
                "Check ROCm driver and HIP_VISIBLE_DEVICES."
            )
        return "rocm"
    # requested == "cuda"
    if _torch_hip_version() is not None:
        raise RuntimeError(
            f"VIDEOTUNA_COMPUTE_BACKEND=cuda but PyTorch reports HIP ({_torch_hip_version()}). "
            "Use VIDEOTUNA_COMPUTE_BACKEND=rocm or install the CUDA PyTorch wheel."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VIDEOTUNA_COMPUTE_BACKEND=cuda but torch.cuda.is_available() is False."
        )
    return "cuda"


def gpu_is_available() -> bool:
    """True when an accelerator GPU is available (NVIDIA CUDA or AMD ROCm)."""
    return torch.cuda.is_available()


def cuda_is_available() -> bool:
    """Deprecated alias for gpu_is_available()."""
    return gpu_is_available()


def accelerator_device_string() -> str:
    """PyTorch device type string for GPU autocast/offload ('cuda' for CUDA and ROCm)."""
    return "cuda" if gpu_is_available() else "cpu"


def resolve_inference_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device for inference."""
    if prefer:
        preferred = torch.device(prefer)
        if preferred.type == "cuda" and not gpu_is_available():
            raise RuntimeError(
                f"Requested device {prefer!r} but no GPU accelerator is available."
            )
        return preferred
    if gpu_is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def empty_accelerator_cache() -> None:
    if gpu_is_available():
        torch.cuda.empty_cache()


def synchronize_accelerator() -> None:
    if gpu_is_available():
        torch.cuda.synchronize()


def describe_compute_environment() -> str:
    backend = _detect_compute_backend_raw()
    if backend == "rocm":
        name = torch.cuda.get_device_name(0)
        hip = _torch_hip_version() or "unknown"
        return f"ROCm available ({name}, torch {torch.__version__}, HIP {hip})"
    if backend == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"CUDA available ({name}, torch {torch.__version__})"
    return "No GPU accelerator (CPU-only PyTorch or no GPU driver)"


def require_accelerator_for_flow(
    flow_target: str,
    *,
    min_vram_gb: float | None = None,
    allow_cpu: bool = False,
) -> None:
    """
    Fail fast when a GPU-backed video flow is started without an accelerator.

    Passes when a CUDA or ROCm GPU is available, or when allow_cpu is True.
    """
    if allow_cpu:
        logger.warning(
            "allow_cpu=True: skipping GPU requirement check for {}",
            flow_target,
        )
        return

    if flow_target not in _GPU_REQUIRED_FLOW_TARGETS:
        return

    backend = detect_compute_backend()

    if flow_target == _STEPVIDEO_FLOW and backend == "rocm":
        raise RuntimeError(
            "StepVideo inference is not supported on AMD ROCm.\n"
            f"  Flow: {flow_target}\n"
            f"  Detected: {describe_compute_environment()}\n"
            "StepVideo depends on proprietary CUDA liboptimus libraries and xfuser "
            "tensor parallel.\n"
            "Alternatives on ROCm:\n"
            "  - Wan 2.2 Diffusers: poetry run inference-wan2.2-t2v-720p\n"
            "  - Hunyuan 1.5 Diffusers: poetry run inference-hunyuan1.5-t2v\n"
            "See docs/install-rocm.md for Tier-A/B model compatibility."
        )

    if gpu_is_available():
        logger.info("Inference device: {}", describe_compute_environment())
        if min_vram_gb is not None:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            if total_gb < min_vram_gb:
                logger.warning(
                    "GPU VRAM {:.1f} GB is below recommended {:.1f} GB for {}",
                    total_gb,
                    min_vram_gb,
                    flow_target,
                )
        return

    raise RuntimeError(
        "This inference command requires a GPU accelerator (NVIDIA CUDA or AMD ROCm).\n"
        f"  Flow: {flow_target}\n"
        f"  Detected: {describe_compute_environment()}\n"
        "Install options:\n"
        "  - NVIDIA: poetry install --extras cuda\n"
        "  - AMD ROCm: poetry install --extras rocm (see docs/install-rocm.md)\n"
        "What you can do without a GPU:\n"
        "  - Run unit/smoke tests: poetry run pytest tests/test_inference_optimization.py\n"
        "  - Validate CLI/config parsing only (no model load)\n"
        "To bypass this check for debugging init on CPU only: "
        "VIDEOTUNA_ALLOW_CPU_INFERENCE=1 poetry run inference-..."
    )


def require_nvidia_cuda_for_flow(flow_target: str, *, allow_cpu: bool = False) -> None:
    """Deprecated alias for require_accelerator_for_flow."""
    require_accelerator_for_flow(flow_target, allow_cpu=allow_cpu)


def require_xfuser_sequence_parallel(flow_name: str) -> None:
    """Fail when xfuser USP is requested on ROCm (CUDA-only dependency)."""
    if detect_compute_backend() == "rocm":
        raise RuntimeError(
            f"Sequence parallel (ulysses_degree / ring_degree) is not supported on "
            f"AMD ROCm for {flow_name}. xfuser requires NVIDIA CUDA.\n"
            "Use single-GPU inference with VIDEOTUNA_ATTN_BACKEND=sdpa instead."
        )


def checkpoints_exist(path: str | None) -> bool:
    if not path:
        return False
    from pathlib import Path

    p = Path(path)
    return p.exists() and (p.is_dir() or p.is_file())


def looks_like_hf_model_id(path: str) -> bool:
    """True for org/model repo ids that are not local paths."""
    if not path or path.startswith(("/", "./", "../")):
        return False
    from pathlib import Path

    if Path(path).exists():
        return False
    parts = path.replace("\\", "/").split("/")
    return len(parts) == 2 and all(parts) and " " not in path


def checkpoint_available(path: str | None, *, flow_target: str = "") -> bool:
    """Local checkpoint exists, or path is a Hugging Face model id."""
    if not path:
        return True
    if checkpoints_exist(path):
        return True
    if "diffusers_video" in flow_target and looks_like_hf_model_id(path):
        return True
    return looks_like_hf_model_id(path)
