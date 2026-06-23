"""Hardware-aware inference preset planner and preflight checker.

Reads VIDEOTUNA_* environment settings, GPU/VRAM information, and optional
user overrides to recommend the correct inference preset YAML or raise a
structured error with next-step hints. Covers CUDA, ROCm, and CPU smoke modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

from videotuna.settings import (
    AttnBackendSetting,
    ComputeBackendSetting,
    get_settings,
    settings_session,
)
from videotuna.utils.attention import get_attn_backend
from videotuna.utils.device_utils import (
    _DIFFUSERS_FLOW,
    GpuInfo,
    detect_compute_backend,
    get_flow_tier,
    get_visible_gpus,
    gpu_is_available,
)

MemoryPreset = Literal["low_vram", "balanced", "max_speed"]
OffloadMode = Literal["sequential", "model", "none"]
FlowName = Literal[
    "wan_t2v",
    "wan_domain_lora_t2v",
    "wan_i2v",
    "wan_domain_lora_i2v",
    "flux_t2i",
    "flux_domain_lora_t2i",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "configs" / "inference" / "presets"

_FP8_MIN_SM = (8, 9)
_LOW_VRAM_GB = 16.0
_BALANCED_VRAM_GB = 32.0


class PresetTier(str, Enum):
    """VRAM-based preset tiers."""

    CPU_SMOKE = "cpu_smoke"
    LOW_VRAM = "low_vram"
    LOW_VRAM_INT8 = "low_vram_int8"
    LOW_VRAM_FP8 = "low_vram_fp8"
    BALANCED = "balanced"
    MAX_SPEED = "max_speed"


@dataclass(frozen=True)
class PresetRecommendation:
    """Recommended inference preset and supporting settings."""

    preset_path: str
    flow: FlowName
    tier: PresetTier
    memory_preset: MemoryPreset | None
    dtype: str | None
    offload_mode: OffloadMode
    attn_backend: AttnBackendSetting
    transformer_quant: str | None
    quant_backend: str | None
    compile_enabled: bool
    vram_gb: float | None
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_path": self.preset_path,
            "flow": self.flow,
            "tier": self.tier.value,
            "memory_preset": self.memory_preset,
            "dtype": self.dtype,
            "offload_mode": self.offload_mode,
            "attn_backend": self.attn_backend,
            "transformer_quant": self.transformer_quant,
            "quant_backend": self.quant_backend,
            "compile_enabled": self.compile_enabled,
            "vram_gb": self.vram_gb,
            "warnings": self.warnings,
            "hints": self.hints,
        }


class PresetPlanningError(RuntimeError):
    """Structured error from the preset planner with next-step hints."""

    def __init__(
        self,
        message: str,
        hints: list[str] | None = None,
        detected_backend: ComputeBackendSetting | None = None,
        detected_vram_gb: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hints = hints or []
        self.detected_backend = detected_backend
        self.detected_vram_gb = detected_vram_gb

    def format(self) -> str:
        lines = [self.message]
        if self.hints:
            lines.append("Next steps:")
            lines.extend(f"  - {hint}" for hint in self.hints)
        return "\n".join(lines)


_PRESET_REGISTRY: dict[FlowName, dict[PresetTier, str]] = {
    "wan_t2v": {
        PresetTier.CPU_SMOKE: "wan2_2_cpu_smoke.yaml",
        PresetTier.LOW_VRAM: "low_vram_wan2_2_720p.yaml",
        PresetTier.LOW_VRAM_INT8: "low_vram_wan2_2_720p_int8.yaml",
        PresetTier.LOW_VRAM_FP8: "low_vram_wan2_2_720p_fp8.yaml",
        PresetTier.BALANCED: "balanced_wan2_2_720p.yaml",
        PresetTier.MAX_SPEED: "max_speed_wan2_2_720p.yaml",
    },
    "wan_domain_lora_t2v": {
        PresetTier.LOW_VRAM: "wan_domain_lora_smoke_22_low_vram.yaml",
        PresetTier.BALANCED: "wan_domain_lora_smoke_22.yaml",
        PresetTier.MAX_SPEED: "wan_domain_lora_smoke_22.yaml",
    },
    "wan_i2v": {
        PresetTier.BALANCED: "wan_domain_i2v_smoke_22.yaml",
    },
    "wan_domain_lora_i2v": {
        PresetTier.BALANCED: "wan_domain_i2v_smoke_22.yaml",
    },
    "flux_t2i": {
        PresetTier.BALANCED: "flux1_dev.yaml",
        PresetTier.MAX_SPEED: "flux1_dev.yaml",
    },
    "flux_domain_lora_t2i": {
        PresetTier.CPU_SMOKE: "flux_domain_lora_smoke.yaml",
        PresetTier.BALANCED: "flux_domain_lora_smoke.yaml",
        PresetTier.MAX_SPEED: "flux_domain_lora_smoke.yaml",
    },
}


_FLOW_DEFAULTS: dict[FlowName, dict[str, Any]] = {
    "wan_t2v": {"family": "wan", "height": 720, "width": 1280, "mode": "t2v"},
    "wan_domain_lora_t2v": {
        "family": "wan",
        "height": 720,
        "width": 1280,
        "mode": "t2v",
    },
    "wan_i2v": {"family": "wan", "height": 720, "width": 1280, "mode": "i2v"},
    "wan_domain_lora_i2v": {
        "family": "wan",
        "height": 720,
        "width": 1280,
        "mode": "i2v",
    },
    "flux_t2i": {"family": "flux", "height": 768, "width": 1360, "mode": "t2i"},
    "flux_domain_lora_t2i": {
        "family": "flux",
        "height": 512,
        "width": 512,
        "mode": "t2i",
    },
}


def _resolve_preset_path(flow: FlowName, tier: PresetTier) -> str:
    flow_registry = _PRESET_REGISTRY.get(flow)
    if flow_registry is None:
        raise PresetPlanningError(
            f"Unknown inference flow {flow!r}.",
            hints=[
                "Use one of: " + ", ".join(_PRESET_REGISTRY.keys()),
            ],
        )
    filename = flow_registry.get(tier)
    if filename is None:
        raise PresetPlanningError(
            f"Preset tier {tier.value!r} is not available for flow {flow!r}.",
            hints=[
                "Available tiers: " + ", ".join(t.value for t in flow_registry),
                "Try a different tier or use --cpu-smoke if supported.",
            ],
        )
    return str(PRESETS_DIR / filename)


def _get_primary_gpu(gpus: list[GpuInfo]) -> GpuInfo | None:
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.total_vram_gb)


def _override_gpus_from_vram(
    gpus: list[GpuInfo],
    vram_gb: float | None,
) -> list[GpuInfo]:
    """Return a synthetic GPU list when a VRAM override is provided."""
    if vram_gb is None:
        return gpus
    return list(gpus) or [GpuInfo(0, "override", vram_gb, vram_gb, (8, 0), True)]


def detect_vram_tier(gpus: list[GpuInfo]) -> PresetTier:
    """Map visible GPU VRAM to a preset tier."""
    primary = _get_primary_gpu(gpus)
    if primary is None:
        return PresetTier.CPU_SMOKE
    vram = primary.total_vram_gb
    if vram < _LOW_VRAM_GB:
        return PresetTier.LOW_VRAM
    if vram < _BALANCED_VRAM_GB:
        return PresetTier.BALANCED
    return PresetTier.MAX_SPEED


def _compute_capability(gpus: list[GpuInfo]) -> tuple[int, int] | None:
    primary = _get_primary_gpu(gpus)
    if primary is None:
        return None
    return primary.compute_capability


def _supports_fp8(gpus: list[GpuInfo]) -> bool:
    cc = _compute_capability(gpus)
    if cc is None:
        return False
    return cc >= _FP8_MIN_SM


def _supports_bf16(gpus: list[GpuInfo]) -> bool:
    primary = _get_primary_gpu(gpus)
    if primary is None:
        return False
    return primary.supports_bf16


def resolve_quant_tier(
    *,
    backend: ComputeBackendSetting,
    gpus: list[GpuInfo],
    requested_quant: str | None,
    vram_tier: PresetTier,
) -> tuple[str, str | None]:
    """Return normalized transformer_quant and quant_backend.

    Returns (quant, backend) where quant is one of 'none', 'int8_wo', 'fp8_wo'.
    """
    quant = (requested_quant or "none").lower().strip()
    if quant == "none" or not quant:
        return "none", None

    if backend == "cpu":
        raise PresetPlanningError(
            f"transformer_quant={quant!r} is not supported on CPU.",
            hints=[
                "Use transformer_quant: none or run without quantization.",
                "CPU inference is limited to smoke validation with eager attention.",
            ],
            detected_backend=backend,
        )
    if backend == "rocm":
        raise PresetPlanningError(
            f"transformer_quant={quant!r} is not supported on AMD ROCm.",
            hints=[
                "Use memory_preset low_vram with CPU offload instead.",
                "Set VIDEOTUNA_ATTN_BACKEND=sdpa on ROCm.",
            ],
            detected_backend=backend,
        )
    if backend != "cuda":
        raise PresetPlanningError(
            f"transformer_quant={quant!r} requires NVIDIA CUDA; "
            f"detected backend {backend!r}.",
            hints=["Install the CUDA stack: poetry install -E cuda"],
            detected_backend=backend,
        )

    if quant not in {"int8_wo", "int4_wo", "fp8_wo"}:
        raise PresetPlanningError(
            f"Unsupported transformer_quant {quant!r}.",
            hints=[
                "Expected one of: none, int8_wo, int4_wo, fp8_wo.",
                "Note: int4_wo is only usable when the quant backend is installed.",
            ],
            detected_backend=backend,
        )

    if quant == "fp8_wo":
        if not _supports_fp8(gpus):
            cc = _compute_capability(gpus)
            detected = ".".join(map(str, cc)) if cc else "unknown"
            raise PresetPlanningError(
                f"transformer_quant=fp8_wo requires NVIDIA sm >= "
                f"{_FP8_MIN_SM[0]}.{_FP8_MIN_SM[1]} (Ada/Hopper); "
                f"detected sm {detected}.",
                hints=[
                    "Use int8_wo on older GPUs.",
                    "Pick a GPU with sm >= 8.9 (e.g. RTX 4090, L40S, Hopper).",
                ],
                detected_backend=backend,
            )

    if vram_tier in {PresetTier.BALANCED, PresetTier.MAX_SPEED}:
        logger.warning(
            "transformer_quant={} is usually unnecessary with {} VRAM; "
            "quantization is intended for low-VRAM GPUs.",
            quant,
            vram_tier.value,
        )

    return quant, "torchao"


def _resolve_auto_quant(
    backend: ComputeBackendSetting,
    gpus: list[GpuInfo],
    vram_tier: PresetTier,
) -> tuple[str, str | None]:
    """Pick a default quant scheme when the user did not request one."""
    if backend != "cuda" or vram_tier != PresetTier.LOW_VRAM:
        return "none", None
    if _supports_fp8(gpus):
        return "fp8_wo", "torchao"
    return "int8_wo", "torchao"


def _resolve_offload_mode(
    memory_preset: MemoryPreset | None,
    enable_sequential: bool,
    enable_model: bool,
) -> OffloadMode:
    if enable_sequential:
        return "sequential"
    if enable_model:
        return "model"
    return "none"


def _offload_mode_from_preset(inference: dict[str, Any]) -> OffloadMode:
    if inference.get("enable_sequential_cpu_offload", False):
        return "sequential"
    if inference.get("enable_model_cpu_offload", False):
        return "model"
    return "none"


def _resolve_attention_backend(
    backend: ComputeBackendSetting,
    gpus: list[GpuInfo],
) -> AttnBackendSetting:
    """Validate and resolve the requested attention backend via get_attn_backend."""
    try:
        resolved = get_attn_backend()
    except RuntimeError as exc:
        raise PresetPlanningError(
            str(exc),
            hints=[
                "Check VIDEOTUNA_ATTN_BACKEND is compatible with your backend.",
                "CPU: eager; ROCm: sdpa; NVIDIA: auto, flash, or sdpa.",
            ],
            detected_backend=backend,
        ) from exc
    except ValueError as exc:
        raise PresetPlanningError(
            str(exc),
            hints=[
                "Expected VIDEOTUNA_ATTN_BACKEND values: auto, flash, sdpa, eager.",
            ],
            detected_backend=backend,
        ) from exc

    if not gpus and resolved == "flash":
        # get_attn_backend may fall back, but guard against a flash-only result.
        raise PresetPlanningError(
            "VIDEOTUNA_ATTN_BACKEND=flash requires a GPU.",
            hints=[
                "Use VIDEOTUNA_ATTN_BACKEND=eager for CPU-only runs.",
            ],
            detected_backend=backend,
        )
    return resolved


def _resolve_compile_flag(
    *,
    compile_flag: bool,
    backend: ComputeBackendSetting,
    offload_mode: OffloadMode,
    vram_tier: PresetTier,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if not compile_flag:
        return False, warnings

    if backend == "cpu":
        raise PresetPlanningError(
            "torch.compile is not supported on CPU.",
            hints=[
                "Keep VIDEOTUNA_TORCH_COMPILE=0 for CPU inference.",
            ],
            detected_backend=backend,
        )

    if offload_mode != "none":
        raise PresetPlanningError(
            "torch.compile requires offload_mode=none (max_speed tier).",
            hints=[
                "Use --memory-preset max_speed or omit CPU offload flags.",
                "Compile is disabled automatically when offloading.",
            ],
            detected_backend=backend,
        )

    if vram_tier != PresetTier.MAX_SPEED:
        raise PresetPlanningError(
            "torch.compile is recommended only with the max_speed tier.",
            hints=[
                "Use --memory-preset max_speed or a GPU with >= 32 GB VRAM.",
            ],
            detected_backend=backend,
        )

    if backend == "rocm":
        warnings.append(
            "torch.compile on AMD ROCm is experimental in PyTorch 2.6; "
            "warm up once before timing."
        )

    return True, warnings


def _check_cpu_smoke_allowed(
    flow: FlowName,
    backend: ComputeBackendSetting,
    cpu_smoke: bool,
) -> None:
    """Raise if CPU smoke is requested for a flow that does not support it."""
    if not cpu_smoke and backend == "cpu":
        return  # will be handled later via tier logic
    if not cpu_smoke:
        return

    # If a dedicated CPU smoke preset exists, allow it (the preset itself uses
    # tiny resolution/steps suitable for CPU validation).
    if PresetTier.CPU_SMOKE in _PRESET_REGISTRY.get(flow, {}):
        return

    defaults = _FLOW_DEFAULTS[flow]
    tier = get_flow_tier(
        _DIFFUSERS_FLOW,
        model_family=defaults["family"],
        height=defaults["height"],
        width=defaults["width"],
    )
    if tier == "gpu_required":
        raise PresetPlanningError(
            f"--cpu-smoke is not supported for {flow} at "
            f"{defaults['height']}x{defaults['width']}.",
            hints=[
                "This resolution requires a GPU accelerator.",
                "Use a lower-resolution smoke preset if available, "
                "or run on CPU-only flows.",
            ],
            detected_backend=backend,
        )


def plan_preset(
    flow: FlowName,
    *,
    compute_backend: ComputeBackendSetting | None = None,
    vram_gb: float | None = None,
    attn_backend: AttnBackendSetting | None = None,
    transformer_quant: str | None = None,
    compile_flag: bool | None = None,
    cpu_smoke: bool | None = None,
    gpus: list[GpuInfo] | None = None,
) -> PresetRecommendation:
    """Recommend an inference preset for the given hardware context.

    Any unspecified parameter is read from the current VIDEOTUNA_* settings
    and visible GPU enumeration.
    """
    settings = get_settings()
    backend = compute_backend or detect_compute_backend()

    if gpus is None:
        gpus = get_visible_gpus() if backend != "cpu" and gpu_is_available() else []

    gpus = _override_gpus_from_vram(gpus, vram_gb)

    cpu_smoke_flag = (
        cpu_smoke if cpu_smoke is not None else (settings.cpu_mode == "smoke")
    )
    compile_flag = compile_flag if compile_flag is not None else settings.torch_compile
    requested_attn = attn_backend or settings.attn_backend

    # Temporarily adjust settings for attention resolution so we can reuse
    # get_attn_backend without mutating global state.
    with settings_session(
        compute_backend=backend,
        attn_backend=requested_attn,
        cpu_mode="smoke" if cpu_smoke_flag else settings.cpu_mode,
        torch_compile=compile_flag,
    ):
        resolved_attn = _resolve_attention_backend(backend, gpus)

    if backend == "cpu" and not cpu_smoke_flag:
        # CPU only makes sense for smoke; otherwise recommend smoke or error.
        registry = _PRESET_REGISTRY.get(flow, {})
        if PresetTier.CPU_SMOKE not in registry:
            raise PresetPlanningError(
                f"Flow {flow!r} requires a GPU accelerator.",
                hints=[
                    "Install NVIDIA: poetry install -E cuda",
                    "Install AMD ROCm: poetry install -E rocm"
                    " (see docs/install-rocm.md)",
                    "Or run with --cpu-smoke on a flow that supports CPU smoke.",
                ],
                detected_backend=backend,
            )
        cpu_smoke_flag = True

    if cpu_smoke_flag:
        _check_cpu_smoke_allowed(flow, backend, cpu_smoke_flag)
        tier = PresetTier.CPU_SMOKE
    else:
        tier = detect_vram_tier(gpus)

    # Adjust tier based on quant preference for low-VRAM CUDA.
    quant, quant_backend = _resolve_auto_quant(backend, gpus, tier)
    if transformer_quant:
        quant, quant_backend = resolve_quant_tier(
            backend=backend,
            gpus=gpus,
            requested_quant=transformer_quant,
            vram_tier=tier,
        )
    flow_registry = _PRESET_REGISTRY.get(flow, {})
    if tier == PresetTier.LOW_VRAM and backend == "cuda":
        if quant == "fp8_wo" and PresetTier.LOW_VRAM_FP8 in flow_registry:
            tier = PresetTier.LOW_VRAM_FP8
        elif quant == "int8_wo" and PresetTier.LOW_VRAM_INT8 in flow_registry:
            tier = PresetTier.LOW_VRAM_INT8
        else:
            # Keep base low_vram preset; quant is still surfaced in the
            # recommendation for explicit CLI overrides.
            pass

    preset_path = _resolve_preset_path(flow, tier)
    inference = _load_preset(preset_path)["inference"]
    memory_preset = inference.get("memory_preset")
    dtype = inference.get("dtype")
    offload_mode = _offload_mode_from_preset(inference)

    warnings: list[str] = []
    if memory_preset == "low_vram" and dtype == "fp16" and not _supports_bf16(gpus):
        warnings.append(
            "fp16 is used for low_vram because bf16 is not supported on this GPU."
        )

    compile_enabled, compile_warnings = _resolve_compile_flag(
        compile_flag=compile_flag,
        backend=backend,
        offload_mode=offload_mode,
        vram_tier=tier,
    )
    warnings.extend(compile_warnings)

    primary = _get_primary_gpu(gpus)
    vram_detected = primary.total_vram_gb if primary else None

    hints: list[str] = []
    if backend == "cpu":
        hints.append(
            "CPU smoke is for pipeline validation only; do not use for production."
        )
    elif backend == "rocm":
        hints.append(
            "Set VIDEOTUNA_ATTN_BACKEND=sdpa on ROCm and avoid flash attention."
        )

    return PresetRecommendation(
        preset_path=preset_path,
        flow=flow,
        tier=tier,
        memory_preset=memory_preset,
        dtype=dtype,
        offload_mode=offload_mode,
        attn_backend=resolved_attn,
        transformer_quant=quant,
        quant_backend=quant_backend,
        compile_enabled=compile_enabled,
        vram_gb=vram_detected,
        warnings=warnings,
        hints=hints,
    )


def _load_preset(preset_path: str) -> dict[str, Any]:
    path = Path(preset_path)
    if not path.exists():
        raise PresetPlanningError(
            f"Preset file not found: {preset_path}",
            hints=[
                "Check that configs/inference/presets/ contains the expected YAML.",
                "Run from the repository root.",
            ],
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def preflight_check(
    preset_path: str,
    *,
    compute_backend: ComputeBackendSetting | None = None,
    vram_gb: float | None = None,
    gpus: list[GpuInfo] | None = None,
    cpu_smoke: bool | None = None,
    compile_flag: bool | None = None,
) -> PresetRecommendation:
    """Validate a chosen preset against the current hardware context.

    Returns the same recommendation shape as :func:`plan_preset`, but infers the
    flow and tier from the YAML file. Raises :class:`PresetPlanningError` with
    precise hints if the preset is incompatible.
    """
    settings = get_settings()
    backend = compute_backend or detect_compute_backend()
    if gpus is None:
        gpus = get_visible_gpus() if backend != "cpu" and gpu_is_available() else []
    gpus = _override_gpus_from_vram(gpus, vram_gb)

    cpu_smoke_flag = (
        cpu_smoke if cpu_smoke is not None else (settings.cpu_mode == "smoke")
    )
    compile_flag = compile_flag if compile_flag is not None else settings.torch_compile

    preset = _load_preset(preset_path)
    inference = preset.get("inference", {})
    flow_params = preset.get("flow", {}).get("params", {})
    family = flow_params.get("model_family", "")
    mode = inference.get("mode") or flow_params.get("mode", "")
    height = inference.get("height") or flow_params.get("height")
    width = inference.get("width") or flow_params.get("width")
    min_vram_gb = inference.get("min_vram_gb")
    preset_quant = inference.get("transformer_quant")
    preset_quant_backend = inference.get("quant_backend", "torchao")
    preset_dtype = inference.get("dtype")
    preset_memory_preset = inference.get("memory_preset")
    offload_mode = _offload_mode_from_preset(inference)
    device = inference.get("device")

    # Determine the flow name from the YAML family + mode.
    flow = _flow_name_from_preset(family, mode, height, width)

    if device == "cpu" or cpu_smoke_flag:
        if backend != "cpu" and not cpu_smoke_flag:
            cpu_smoke_flag = True
        if backend == "cpu" and not cpu_smoke_flag:
            cpu_smoke_flag = True
        _check_cpu_smoke_allowed(flow, backend, cpu_smoke_flag)
        tier = PresetTier.CPU_SMOKE
    else:
        tier = detect_vram_tier(gpus)

    if min_vram_gb is not None and gpus:
        primary = _get_primary_gpu(gpus)
        if primary is not None and primary.total_vram_gb < min_vram_gb:
            raise PresetPlanningError(
                f"GPU VRAM {primary.total_vram_gb:.1f} GB is below preset "
                f"requirement {min_vram_gb:.1f} GB for {preset_path}.",
                hints=[
                    "Use a lower-VRAM preset: --memory-preset low_vram",
                    "Lower resolution or frame count in the config",
                    "Select a GPU with more VRAM via --device / CUDA_VISIBLE_DEVICES",
                ],
                detected_backend=backend,
                detected_vram_gb=primary.total_vram_gb,
            )

    if backend == "cpu" and preset_quant not in (None, "none"):
        raise PresetPlanningError(
            f"Preset {preset_path} requests transformer_quant={preset_quant!r} "
            f"but the active backend is CPU.",
            hints=[
                "Remove transformer_quant from the preset or use a CUDA backend.",
                "Run with --cpu-smoke on a preset without quantization.",
            ],
            detected_backend=backend,
        )
    if backend == "rocm" and preset_quant not in (None, "none"):
        raise PresetPlanningError(
            f"Preset {preset_path} requests transformer_quant={preset_quant!r} "
            f"but the active backend is AMD ROCm.",
            hints=[
                "Use a non-quantized preset with low_vram + CPU offload on ROCm.",
                "Set VIDEOTUNA_ATTN_BACKEND=sdpa.",
            ],
            detected_backend=backend,
        )
    if preset_quant == "fp8_wo" and not _supports_fp8(gpus):
        cc = _compute_capability(gpus)
        detected = ".".join(map(str, cc)) if cc else "unknown"
        min_sm = f"{_FP8_MIN_SM[0]}.{_FP8_MIN_SM[1]}"
        raise PresetPlanningError(
            f"Preset {preset_path} requests transformer_quant=fp8_wo but "
            f"detected GPU sm {detected} is below sm {min_sm}.",
            hints=[
                "Use the int8 preset instead.",
                "Pick an Ada/Hopper GPU (sm >= 8.9) for fp8.",
            ],
            detected_backend=backend,
        )

    compile_enabled, compile_warnings = _resolve_compile_flag(
        compile_flag=compile_flag,
        backend=backend,
        offload_mode=offload_mode,
        vram_tier=tier,
    )

    with settings_session(
        compute_backend=backend,
        cpu_mode="smoke" if cpu_smoke_flag else settings.cpu_mode,
        torch_compile=compile_flag,
    ):
        resolved_attn = _resolve_attention_backend(backend, gpus)

    warnings: list[str] = list(compile_warnings)
    if cpu_smoke_flag:
        warnings.append(
            "CPU smoke is limited to tiny resolution/steps; "
            "production inference requires a GPU."
        )

    hints: list[str] = []
    if backend == "cpu":
        hints.append(
            "CPU smoke is for pipeline validation only; do not use for production."
        )
    elif backend == "rocm":
        hints.append(
            "Set VIDEOTUNA_ATTN_BACKEND=sdpa on ROCm and avoid flash attention."
        )

    primary = _get_primary_gpu(gpus)
    return PresetRecommendation(
        preset_path=preset_path,
        flow=flow,
        tier=tier,
        memory_preset=preset_memory_preset,
        dtype=preset_dtype,
        offload_mode=offload_mode,
        attn_backend=resolved_attn,
        transformer_quant=preset_quant,
        quant_backend=preset_quant_backend if preset_quant else None,
        compile_enabled=compile_enabled,
        vram_gb=primary.total_vram_gb if primary else None,
        warnings=warnings,
        hints=hints,
    )


def _flow_name_from_preset(family: str, mode: str, height: Any, width: Any) -> FlowName:
    family = family.lower()
    mode = str(mode).lower()
    if family == "wan" and mode in ("t2v", ""):
        if height == 720 and width == 1280:
            return "wan_t2v"
        return "wan_t2v"
    if family == "wan" and mode == "i2v":
        return "wan_i2v"
    if family == "flux":
        return "flux_t2i"
    return "wan_t2v"


__all__ = [
    "PresetTier",
    "PresetRecommendation",
    "PresetPlanningError",
    "FlowName",
    "plan_preset",
    "preflight_check",
    "detect_vram_tier",
]
