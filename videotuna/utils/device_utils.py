"""Device detection and inference hardware requirements."""

from __future__ import annotations

import torch
from loguru import logger


def cuda_is_available() -> bool:
    return torch.cuda.is_available()


def resolve_inference_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device for inference."""
    if prefer:
        preferred = torch.device(prefer)
        if preferred.type == "cuda" and not cuda_is_available():
            raise RuntimeError(
                f"Requested device {prefer!r} but torch.cuda.is_available() is False."
            )
        return preferred
    if cuda_is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_compute_environment() -> str:
    if cuda_is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA available ({name})"
    return "CUDA not available (CPU-only PyTorch or no NVIDIA driver)"


# Flows that need a GPU for practical 720p video generation.
_GPU_REQUIRED_FLOW_TARGETS = (
    "videotuna.flow.hunyuanvideo.HunyuanVideoFlow",
    "videotuna.flow.wanvideo.WanVideoModelFlow",
    "videotuna.flow.stepvideo.StepVideoModelFlow",
)


def require_nvidia_cuda_for_flow(flow_target: str, *, allow_cpu: bool = False) -> None:
    """
    Fail fast when a GPU-backed video flow is started without CUDA.

    VideoTuna's default Poetry install pins PyTorch to the CUDA 12.6 wheel
    (pytorch-cu126). AMD ROCm is not supported out of the box; an AMD GPU
    will not be used unless you rebuild the stack for ROCm yourself.
    """
    if allow_cpu:
        logger.warning(
            "allow_cpu=True: skipping GPU requirement check for {}",
            flow_target,
        )
        return

    if flow_target not in _GPU_REQUIRED_FLOW_TARGETS:
        return

    if cuda_is_available():
        logger.info("Inference device: {}", describe_compute_environment())
        return

    raise RuntimeError(
        "This inference command requires an NVIDIA GPU with a working CUDA driver.\n"
        f"  Flow: {flow_target}\n"
        f"  Detected: {describe_compute_environment()}\n"
        "VideoTuna's default install uses PyTorch built for NVIDIA CUDA (cu126). "
        "AMD GPUs are not used by that build.\n"
        "What you can do locally without NVIDIA CUDA:\n"
        "  - Run unit/smoke tests: poetry run pytest tests/test_inference_optimization.py\n"
        "  - Validate CLI/config parsing only (no model load)\n"
        "For full Hunyuan/Wan/StepVideo generation, use a machine with NVIDIA GPU + "
        "downloaded checkpoints under checkpoints/.\n"
        "To bypass this check for debugging init on CPU only: "
        "VIDEOTUNA_ALLOW_CPU_INFERENCE=1 poetry run inference-..."
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
