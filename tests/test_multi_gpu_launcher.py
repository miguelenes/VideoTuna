"""Tests for multi-GPU launcher/validator."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from videotuna.utils import multi_gpu_launcher as mgl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_caches():
    mgl._reset_device_cache()
    yield
    mgl._reset_device_cache()


def _cuda_env(n_gpu: int = 4) -> mock.Mock:
    p = mock.patch.object(mgl, "_gpu_count", return_value=n_gpu)
    p.start()
    return p


def _cuda_available_patch(available: bool = True) -> mock.Mock:
    p = mock.patch.object(mgl, "_cuda_available", return_value=available)
    p.start()
    return p


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def _fatal_count(diags: tuple[mgl.Diagnostic, ...]) -> int:
    return sum(1 for d in diags if d.severity == "fatal")


def _find(diags: tuple[mgl.Diagnostic, ...], text: str) -> mgl.Diagnostic | None:
    for d in diags:
        if text in d.message:
            return d
    return None


# ---------------------------------------------------------------------------
# device_map validation
# ---------------------------------------------------------------------------


def test_device_map_ok():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_accelerate_available", return_value=True):
                with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                    with mock.patch.object(
                        mgl, "_detected_attn_backend", return_value="sdpa"
                    ):
                        spec = mgl.MultiGpuSpec(
                            mode="device_map",
                            gpu_ids=(0, 1),
                            offload_mode="none",
                        )
                        result = mgl.validate_multi_gpu_setup(spec)
                        assert result.success is True
                        assert result.generated_command is not None


def test_device_map_fails_with_offload():
    spec = mgl.MultiGpuSpec(
        mode="device_map",
        gpu_ids=(0, 1),
        offload_mode="sequential",
    )
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_accelerate_available", return_value=True):
            result = mgl.validate_multi_gpu_setup(spec)
            assert result.success is False
            assert _fatal_count(result.diagnostics) >= 1
            assert _find(result.diagnostics, "mutually exclusive") is not None


def test_device_map_warns_single_gpu():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=1):
            with mock.patch.object(mgl, "_accelerate_available", return_value=True):
                spec = mgl.MultiGpuSpec(
                    mode="device_map",
                    gpu_ids=(0,),
                    offload_mode="none",
                )
                result = mgl.validate_multi_gpu_setup(spec)
                assert result.success is True
                assert _find(result.diagnostics, "benefits from 2+ GPUs") is not None


def test_device_map_no_cuda_fatal():
    spec = mgl.MultiGpuSpec(mode="device_map", offload_mode="none")
    with _cuda_available_patch(False):
        result = mgl.validate_multi_gpu_setup(spec)
        assert result.success is False
        assert _find(result.diagnostics, "CUDA is not available") is not None


def test_device_map_on_rocm_ok():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_accelerate_available", return_value=True):
            with mock.patch.object(mgl, "_compute_backend", return_value="rocm"):
                spec = mgl.MultiGpuSpec(
                    mode="device_map",
                    gpu_ids=(0, 1),
                    offload_mode="none",
                )
                result = mgl.validate_multi_gpu_setup(spec)
                assert result.success is True
                assert _find(result.diagnostics, "works on AMD ROCm") is not None


def test_device_map_no_accelerate_fatal():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_accelerate_available", return_value=False):
            spec = mgl.MultiGpuSpec(
                mode="device_map", gpu_ids=(0, 1), offload_mode="none"
            )
            result = mgl.validate_multi_gpu_setup(spec)
            assert result.success is False
            assert _find(result.diagnostics, "requires accelerate") is not None


# ---------------------------------------------------------------------------
# xfuser validation
# ---------------------------------------------------------------------------


def test_xfuser_ok():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_xfuser_available", return_value=True):
                    with mock.patch.object(mgl, "_nccl_available", return_value=True):
                        with mock.patch.object(
                            mgl,
                            "_detected_attn_backend",
                            return_value="flash",
                        ):
                            spec = mgl.MultiGpuSpec(
                                mode="xfuser",
                                gpu_ids=(0, 1, 2, 3),
                                ulysses_degree=2,
                                ring_degree=2,
                                offload_mode="none",
                            )
                            result = mgl.validate_multi_gpu_setup(spec)
                            assert result.success is True
                            assert result.generated_command is not None
                            assert "torchrun" in result.generated_command


def test_xfuser_fails_on_rocm():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_compute_backend", return_value="rocm"):
            spec = mgl.MultiGpuSpec(mode="xfuser", gpu_ids=(0, 1))
            result = mgl.validate_multi_gpu_setup(spec)
            assert result.success is False
            assert _find(result.diagnostics, "not supported on AMD ROCm") is not None


def test_xfuser_degree_product_mismatch():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_xfuser_available", return_value=True):
                    spec = mgl.MultiGpuSpec(
                        mode="xfuser",
                        gpu_ids=(0, 1, 2),
                        ulysses_degree=2,
                        ring_degree=2,
                        offload_mode="none",
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is False
                    d = _find(result.diagnostics, "ulysses_degree")
                    assert d is not None
                    assert "4" in d.message and "3" in d.message


def test_xfuser_fails_with_offload():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_xfuser_available", return_value=True):
                    spec = mgl.MultiGpuSpec(
                        mode="xfuser",
                        gpu_ids=(0, 1),
                        ulysses_degree=2,
                        ring_degree=1,
                        offload_mode="model",
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is False
                    assert (
                        _find(result.diagnostics, "does not support CPU offload")
                        is not None
                    )


def test_xfuser_fails_fewer_than_two_gpus():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=1):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                spec = mgl.MultiGpuSpec(mode="xfuser", gpu_ids=(0,))
                result = mgl.validate_multi_gpu_setup(spec)
                assert result.success is False
                assert _find(result.diagnostics, "at least 2 GPUs") is not None


def test_xfuser_no_cuda_fatal():
    spec = mgl.MultiGpuSpec(mode="xfuser")
    with _cuda_available_patch(False):
        result = mgl.validate_multi_gpu_setup(spec)
        assert result.success is False
        assert _find(result.diagnostics, "CUDA is not available") is not None


def test_xfuser_no_xfuser_installed_fatal():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_xfuser_available", return_value=False):
                    spec = mgl.MultiGpuSpec(
                        mode="xfuser",
                        gpu_ids=(0, 1),
                        offload_mode="none",
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is False
                    assert (
                        _find(result.diagnostics, "xfuser is not installed") is not None
                    )


def test_xfuser_nccl_warning():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_xfuser_available", return_value=True):
                    with mock.patch.object(mgl, "_nccl_available", return_value=False):
                        spec = mgl.MultiGpuSpec(
                            mode="xfuser",
                            gpu_ids=(0, 1),
                            ulysses_degree=2,
                            ring_degree=1,
                            offload_mode="none",
                        )
                        result = mgl.validate_multi_gpu_setup(spec)
                        assert result.success is True
                        assert (
                            _find(result.diagnostics, "NCCL check failed") is not None
                        )


# ---------------------------------------------------------------------------
# Wan Lightning training validation
# ---------------------------------------------------------------------------


def test_wan_lightning_ok():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_deepspeed_available", return_value=True):
                    with mock.patch.object(mgl, "_nccl_available", return_value=True):
                        spec = mgl.MultiGpuSpec(
                            mode="wan_lightning",
                            gpu_ids=(0, 1, 2, 3),
                            devices="0,1,2,3",
                        )
                        result = mgl.validate_multi_gpu_setup(spec)
                        assert result.success is True
                        assert result.generated_command is not None


def test_wan_lightning_fails_devices_exceed_gpus():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                spec = mgl.MultiGpuSpec(
                    mode="wan_lightning",
                    gpu_ids=(0, 1),
                    devices="0,1,2,3",
                )
                result = mgl.validate_multi_gpu_setup(spec)
                assert result.success is False
                assert _find(result.diagnostics, "requests 4") is not None


def test_wan_lightning_fails_on_rocm():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_compute_backend", return_value="rocm"):
            spec = mgl.MultiGpuSpec(mode="wan_lightning")
            result = mgl.validate_multi_gpu_setup(spec)
            assert result.success is False
            assert _find(result.diagnostics, "requires NVIDIA CUDA") is not None


def test_wan_lightning_no_cuda_fatal():
    with _cuda_available_patch(False):
        result = mgl.validate_multi_gpu_setup(mgl.MultiGpuSpec(mode="wan_lightning"))
        assert result.success is False
        assert _find(result.diagnostics, "CUDA is not available") is not None


def test_wan_lightning_warns_no_deepspeed():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_deepspeed_available", return_value=False):
                    spec = mgl.MultiGpuSpec(
                        mode="wan_lightning",
                        gpu_ids=(0, 1),
                        devices="0,1",
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is True
                    assert (
                        _find(result.diagnostics, "DeepSpeed is not installed")
                        is not None
                    )


def test_wan_lightning_nccl_warning():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_deepspeed_available", return_value=True):
                    with mock.patch.object(mgl, "_nccl_available", return_value=False):
                        spec = mgl.MultiGpuSpec(
                            mode="wan_lightning",
                            gpu_ids=(0, 1),
                            devices="0,1",
                        )
                        result = mgl.validate_multi_gpu_setup(spec)
                        assert result.success is True
                        assert _find(result.diagnostics, "NCCL unavailable") is not None


# ---------------------------------------------------------------------------
# Flux Accelerate training validation
# ---------------------------------------------------------------------------


def test_flux_accelerate_ok():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                with mock.patch.object(mgl, "_accelerate_available", return_value=True):
                    spec = mgl.MultiGpuSpec(
                        mode="flux_accelerate",
                        gpu_ids=(0, 1, 2, 3),
                        num_processes=4,
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is True
                    assert result.generated_command is not None


def test_flux_accelerate_fails_processes_exceed_gpus():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                spec = mgl.MultiGpuSpec(
                    mode="flux_accelerate",
                    gpu_ids=(0, 1),
                    num_processes=8,
                )
                result = mgl.validate_multi_gpu_setup(spec)
                assert result.success is False
                assert _find(result.diagnostics, "exceeds 2") is not None


def test_flux_accelerate_no_cuda_fatal():
    with _cuda_available_patch(False):
        result = mgl.validate_multi_gpu_setup(mgl.MultiGpuSpec(mode="flux_accelerate"))
        assert result.success is False
        assert _find(result.diagnostics, "CUDA is not available") is not None


def test_flux_accelerate_works_on_rocm():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="rocm"):
                with mock.patch.object(mgl, "_accelerate_available", return_value=True):
                    spec = mgl.MultiGpuSpec(
                        mode="flux_accelerate",
                        gpu_ids=(0, 1),
                        num_processes=1,
                    )
                    result = mgl.validate_multi_gpu_setup(spec)
                    assert result.success is True
                    assert _find(result.diagnostics, "works on ROCm") is not None


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------


def test_generate_device_map_command():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                spec = mgl.MultiGpuSpec(
                    mode="device_map",
                    gpu_ids=(0, 1),
                    config_path="configs/inference/presets/max_speed_wan2_2_720p.yaml",
                    offload_mode="none",
                    max_memory_per_gpu="40GiB",
                )
                cmd = mgl.generate_launch_command(spec)
                assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
                assert "device-map auto" in cmd
                assert "max-memory-per-gpu 40GiB" in cmd
                assert "max_speed_wan2_2_720p" in cmd


def test_generate_device_map_command_rocm():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            with mock.patch.object(mgl, "_compute_backend", return_value="rocm"):
                spec = mgl.MultiGpuSpec(
                    mode="device_map",
                    gpu_ids=(0, 1),
                    offload_mode="none",
                )
                cmd = mgl.generate_launch_command(spec)
                assert "VIDEOTUNA_ATTN_BACKEND=sdpa" in cmd


def test_generate_xfuser_command():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                spec = mgl.MultiGpuSpec(
                    mode="xfuser",
                    gpu_ids=(0, 1, 2, 3),
                    ulysses_degree=2,
                    ring_degree=2,
                    config_path="configs/inference/presets/wan2_2_native_t2v_14b.yaml",
                    offload_mode="none",
                )
                cmd = mgl.generate_launch_command(spec)
                assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in cmd
                assert "torchrun --nproc_per_node=4" in cmd
                assert "NCCL_DEBUG=INFO" in cmd
                assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in cmd
                assert "ulysses_degree 2" in cmd
                assert "ring_degree 2" in cmd


def test_generate_wan_lightning_command():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            spec = mgl.MultiGpuSpec(
                mode="wan_lightning",
                gpu_ids=(0, 1, 2, 3),
                devices="0,1,2,3",
            )
            cmd = mgl.generate_launch_command(spec)
            assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in cmd
            assert "train-domain-t2v" in cmd
            assert "--devices" in cmd
            assert "NCCL_DEBUG=INFO" in cmd


def test_generate_wan_lightning_single_gpu_no_nccl():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=1):
            spec = mgl.MultiGpuSpec(
                mode="wan_lightning",
                gpu_ids=(0,),
                devices="0,",
            )
            cmd = mgl.generate_launch_command(spec)
            assert "NCCL_DEBUG=INFO" not in cmd


def test_generate_flux_accelerate_command():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            spec = mgl.MultiGpuSpec(
                mode="flux_accelerate",
                gpu_ids=(0, 1, 2, 3),
                num_processes=4,
            )
            cmd = mgl.generate_launch_command(spec)
            assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in cmd
            assert "accelerate launch" in cmd
            assert "num_processes=4" in cmd
            assert "train_flux_lora.py" in cmd


def test_generate_command_with_extra_args():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=2):
            spec = mgl.MultiGpuSpec(
                mode="device_map",
                gpu_ids=(0, 1),
                offload_mode="none",
                extra_args={"validation_prompt": "test", "seed": "42"},
            )
            cmd = mgl.generate_launch_command(spec)
            assert "validation_prompt test" in cmd
            assert "seed 42" in cmd


# ---------------------------------------------------------------------------
# Unknown mode
# ---------------------------------------------------------------------------


def test_unknown_mode():
    spec = mgl.MultiGpuSpec(mode="invalid_mode")  # type: ignore[arg-type]
    result = mgl.validate_multi_gpu_setup(spec)
    assert result.success is False
    assert _find(result.diagnostics, "Unknown multi-GPU mode") is not None


# ---------------------------------------------------------------------------
# Diagnose failure
# ---------------------------------------------------------------------------


def test_diagnose_hang():
    steps = mgl.diagnose_failure("Hang at init")
    assert any("ulysses_degree" in s for s in steps)
    assert any("NCCL_DEBUG=INFO" in s for s in steps)


def test_diagnose_oom():
    steps = mgl.diagnose_failure("OOM")
    assert any("loaded on all ranks" in s for s in steps)


def test_diagnose_xfuser_import():
    steps = mgl.diagnose_failure("xfuser_import_error")
    assert any("poetry install -E cuda" in s for s in steps)


def test_diagnose_xfuser_rocm():
    steps = mgl.diagnose_failure("xfuser_rocm")
    assert any("requires NVIDIA CUDA" in s for s in steps)
    assert any("VIDEOTUNA_ATTN_BACKEND=sdpa" in s for s in steps)


def test_diagnose_deepspeed():
    steps = mgl.diagnose_failure("deepspeed_import_error")
    assert any("poetry run install-deepspeed" in s for s in steps)


def test_diagnose_nccl_timeout():
    steps = mgl.diagnose_failure("nccl_timeout")
    assert any("NCCL_TIMEOUT=1800" in s for s in steps)


def test_diagnose_device_map_cpu_offload():
    steps = mgl.diagnose_failure("device_map_cpu_offload")
    assert any("mutually exclusive" in s for s in steps)


def test_diagnose_unknown():
    steps = mgl.diagnose_failure("some gibberish symptom")
    assert any("No known diagnostic" in s for s in steps)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_gpu_ids_uses_env():
    with _cuda_available_patch(True):
        with mock.patch.object(mgl, "_gpu_count", return_value=4):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
                with mock.patch.object(mgl, "_compute_backend", return_value="cuda"):
                    with mock.patch.object(
                        mgl, "_accelerate_available", return_value=True
                    ):
                        spec = mgl.MultiGpuSpec(
                            mode="device_map",
                            gpu_ids=(),
                            offload_mode="none",
                        )
                        result = mgl.validate_multi_gpu_setup(spec)
                        assert result.success is True
