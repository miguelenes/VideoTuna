"""Tests for the hardware-aware preset planner and preflight checker."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest import mock

import pytest

from videotuna.utils import attention, preset_planner
from videotuna.utils.device_utils import GpuInfo
from videotuna.utils.preset_planner import (
    PresetPlanningError,
    detect_vram_tier,
    plan_preset,
    preflight_check,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS = REPO_ROOT / "configs" / "inference" / "presets"


def _make_gpu(vram_gb: float, sm: tuple[int, int]) -> GpuInfo:
    return GpuInfo(
        index=0,
        name=f"Mock GPU {vram_gb}GB",
        total_vram_gb=vram_gb,
        free_vram_gb=vram_gb,
        compute_capability=sm,
        supports_bf16=sm[0] >= 8,
    )


@contextmanager
def _hardware_mocks(
    backend: str,
    gpus: list[GpuInfo] | None = None,
    *,
    flash_available: bool = False,
) -> Generator[None, None, None]:
    """Patch device and attention detection for deterministic tests."""
    gpus = gpus or []
    gpu_available = backend != "cpu" and bool(gpus)
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(
            preset_planner, "detect_compute_backend", return_value=backend
        )
    )
    stack.enter_context(
        mock.patch.object(
            preset_planner, "gpu_is_available", return_value=gpu_available
        )
    )
    stack.enter_context(
        mock.patch.object(preset_planner, "get_visible_gpus", return_value=gpus)
    )
    stack.enter_context(
        mock.patch.object(attention, "detect_compute_backend", return_value=backend)
    )
    stack.enter_context(
        mock.patch.object(attention, "gpu_is_available", return_value=gpu_available)
    )
    stack.enter_context(
        mock.patch.object(
            attention, "is_flash_attn_available", return_value=flash_available
        )
    )
    stack.enter_context(
        mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", flash_available)
    )
    try:
        yield
    finally:
        stack.close()


class TestDetectVramTier:
    def test_no_gpu_returns_cpu_smoke(self):
        assert detect_vram_tier([]) == preset_planner.PresetTier.CPU_SMOKE

    def test_12_gb_returns_low_vram(self):
        tier = detect_vram_tier([_make_gpu(12, (8, 6))])
        assert tier == preset_planner.PresetTier.LOW_VRAM

    def test_24_gb_returns_balanced(self):
        tier = detect_vram_tier([_make_gpu(24, (8, 9))])
        assert tier == preset_planner.PresetTier.BALANCED

    def test_48_gb_returns_max_speed(self):
        tier = detect_vram_tier([_make_gpu(48, (8, 9))])
        assert tier == preset_planner.PresetTier.MAX_SPEED


class TestPlanPresetCpu:
    def test_cpu_smoke_for_wan_t2v(self):
        with _hardware_mocks("cpu"):
            rec = plan_preset("wan_t2v", cpu_smoke=True)
        assert rec.tier == preset_planner.PresetTier.CPU_SMOKE
        assert rec.attn_backend == "eager"
        assert rec.transformer_quant == "none"
        assert rec.compile_enabled is False
        assert "wan2_2_cpu_smoke.yaml" in rec.preset_path

    def test_cpu_without_smoke_auto_smoke_for_wan_t2v(self):
        with _hardware_mocks("cpu"):
            rec = plan_preset("wan_t2v", cpu_smoke=False)
        assert rec.tier == preset_planner.PresetTier.CPU_SMOKE
        assert "wan2_2_cpu_smoke.yaml" in rec.preset_path

    def test_cpu_without_smoke_errors_for_gpu_only_flow(self):
        with _hardware_mocks("cpu"):
            with pytest.raises(PresetPlanningError, match="requires a GPU"):
                plan_preset("wan_domain_lora_t2v", cpu_smoke=False)

    def test_cpu_quant_errors(self):
        with _hardware_mocks("cpu"):
            with pytest.raises(PresetPlanningError, match="not supported on CPU"):
                plan_preset("wan_t2v", cpu_smoke=True, transformer_quant="int8_wo")

    def test_cpu_compile_errors(self):
        with _hardware_mocks("cpu"):
            with pytest.raises(
                PresetPlanningError, match="torch.compile is not supported"
            ):
                plan_preset("wan_t2v", cpu_smoke=True, compile_flag=True)


class TestPlanPresetCuda:
    def test_low_vram_base(self):
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus)
        # Auto-quant selects int8 on sm 8.6 low-VRAM, so tier maps to int8 preset.
        assert rec.tier == preset_planner.PresetTier.LOW_VRAM_INT8
        assert rec.memory_preset == "low_vram"
        assert rec.dtype == "fp16"
        assert rec.transformer_quant == "int8_wo"

    def test_low_vram_no_quant(self):
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus, transformer_quant="none")
        assert rec.tier == preset_planner.PresetTier.LOW_VRAM
        assert rec.transformer_quant == "none"

    def test_low_vram_int8_explicit(self):
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus, transformer_quant="int8_wo")
        assert rec.tier == preset_planner.PresetTier.LOW_VRAM_INT8
        assert rec.transformer_quant == "int8_wo"
        assert rec.quant_backend == "torchao"

    def test_low_vram_fp8_auto(self):
        gpus = [_make_gpu(12, (8, 9))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus)
        assert rec.tier == preset_planner.PresetTier.LOW_VRAM_FP8
        assert rec.transformer_quant == "fp8_wo"

    def test_balanced_24gb(self):
        gpus = [_make_gpu(24, (8, 9))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus)
        assert rec.tier == preset_planner.PresetTier.BALANCED
        assert rec.memory_preset == "balanced"
        assert rec.dtype == "bf16"
        assert rec.offload_mode == "model"
        assert rec.transformer_quant == "none"

    def test_max_speed_48gb(self):
        gpus = [_make_gpu(48, (8, 9))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus, compile_flag=True)
        assert rec.tier == preset_planner.PresetTier.MAX_SPEED
        assert rec.compile_enabled is True
        assert rec.offload_mode == "none"

    def test_compile_rejected_with_offload(self):
        gpus = [_make_gpu(24, (8, 9))]
        with _hardware_mocks("cuda", gpus):
            with pytest.raises(PresetPlanningError, match="offload_mode=none"):
                plan_preset("wan_t2v", gpus=gpus, compile_flag=True)

    def test_fp8_rejected_on_old_gpu(self):
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            with pytest.raises(PresetPlanningError, match="Ada/Hopper"):
                plan_preset("wan_t2v", gpus=gpus, transformer_quant="fp8_wo")

    def test_flash_backend_resolves_on_cuda(self):
        gpus = [_make_gpu(48, (8, 9))]
        with _hardware_mocks("cuda", gpus, flash_available=True):
            rec = plan_preset("wan_t2v", gpus=gpus, attn_backend="flash")
        assert rec.attn_backend == "flash"


class TestPlanPresetRocm:
    def test_balanced_24gb(self):
        gpus = [_make_gpu(24, (9, 0))]
        with _hardware_mocks("rocm", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus)
        assert rec.attn_backend == "sdpa"
        assert rec.tier == preset_planner.PresetTier.BALANCED

    def test_flash_on_rocm_errors(self):
        gpus = [_make_gpu(24, (9, 0))]
        with _hardware_mocks("rocm", gpus):
            with pytest.raises(PresetPlanningError, match="not supported on AMD ROCm"):
                plan_preset("wan_t2v", gpus=gpus, attn_backend="flash")

    def test_quant_on_rocm_errors(self):
        gpus = [_make_gpu(12, (9, 0))]
        with _hardware_mocks("rocm", gpus):
            with pytest.raises(PresetPlanningError, match="not supported on AMD ROCm"):
                plan_preset("wan_t2v", gpus=gpus, transformer_quant="int8_wo")

    def test_compile_on_rocm_warns(self):
        gpus = [_make_gpu(48, (9, 0))]
        with _hardware_mocks("rocm", gpus):
            rec = plan_preset("wan_t2v", gpus=gpus, compile_flag=True)
        assert rec.compile_enabled is True
        assert any("experimental" in w for w in rec.warnings)


class TestOtherFlows:
    def test_flux_t2i(self):
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("flux_t2i", gpus=gpus)
        assert "flux1_dev.yaml" in rec.preset_path

    def test_flux_domain_lora_t2i(self):
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("flux_domain_lora_t2i", gpus=gpus)
        assert "flux_domain_lora_smoke.yaml" in rec.preset_path

    def test_wan_domain_lora_t2v_low_vram(self):
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_domain_lora_t2v", gpus=gpus)
        assert "wan_domain_lora_smoke_22_low_vram.yaml" in rec.preset_path

    def test_wan_domain_lora_t2v_balanced(self):
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_domain_lora_t2v", gpus=gpus)
        assert "wan_domain_lora_smoke_22.yaml" in rec.preset_path

    def test_wan_i2v(self):
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = plan_preset("wan_i2v", gpus=gpus)
        assert "wan_domain_i2v_smoke_22.yaml" in rec.preset_path

    def test_unknown_flow(self):
        with _hardware_mocks("cuda", [_make_gpu(24, (8, 6))]):
            with pytest.raises(PresetPlanningError, match="Unknown inference flow"):
                plan_preset("unknown_flow", gpus=[_make_gpu(24, (8, 6))])  # type: ignore[arg-type]


class TestPreflightCheck:
    def test_balanced_preset_ok_on_cuda(self):
        preset = str(PRESETS / "balanced_wan2_2_720p.yaml")
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            rec = preflight_check(preset, gpus=gpus)
        assert rec.tier == preset_planner.PresetTier.BALANCED

    def test_max_speed_preset_vram_insufficient(self):
        preset = str(PRESETS / "max_speed_wan2_2_720p.yaml")
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            with pytest.raises(PresetPlanningError, match="below preset requirement"):
                preflight_check(preset, gpus=gpus)

    def test_int8_preset_on_cpu_errors(self):
        preset = str(PRESETS / "low_vram_wan2_2_720p_int8.yaml")
        with _hardware_mocks("cpu"):
            with pytest.raises(PresetPlanningError, match="requests transformer_quant"):
                preflight_check(preset)

    def test_fp8_preset_on_old_gpu_errors(self):
        preset = str(PRESETS / "low_vram_wan2_2_720p_fp8.yaml")
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            with pytest.raises(PresetPlanningError, match="below sm"):
                preflight_check(preset, gpus=gpus)

    def test_cpu_smoke_preset_ok(self):
        preset = str(PRESETS / "wan2_2_cpu_smoke.yaml")
        with _hardware_mocks("cpu"):
            rec = preflight_check(preset, cpu_smoke=True)
        assert rec.tier == preset_planner.PresetTier.CPU_SMOKE


class TestCli:
    def test_recommend_json_output(self):
        from videotuna.cli.preset_planner_app import PlannerOptions, _recommend

        options = PlannerOptions(
            flow="wan_t2v",
            vram_gb=24.0,
            json=True,
        )
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            assert _recommend(options) == 0

    def test_validate_json_output(self):
        from videotuna.cli.preset_planner_app import PlannerOptions, _validate

        preset = str(PRESETS / "balanced_wan2_2_720p.yaml")
        options = PlannerOptions(preset=preset, vram_gb=24.0, json=True)
        gpus = [_make_gpu(24, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            assert _validate(options) == 0

    def test_list_flows_output(self):
        from videotuna.cli.preset_planner_app import list_flows

        assert list_flows() == 0

    def test_error_includes_hints(self):
        from videotuna.cli.preset_planner_app import PlannerOptions, _recommend

        options = PlannerOptions(
            flow="wan_t2v",
            vram_gb=12.0,
            transformer_quant="fp8_wo",
            json=True,
        )
        gpus = [_make_gpu(12, (8, 6))]
        with _hardware_mocks("cuda", gpus):
            assert _recommend(options) == 2
