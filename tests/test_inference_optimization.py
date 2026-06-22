"""Tests for inference CLI, metrics, and FP8 validation."""

import argparse
import json
import os
import tempfile
from unittest import mock

import pytest

from videotuna.utils.common_utils import monitor_resources, save_metrics
from videotuna.utils.fp8_utils import (
    fp8_map_path,
    precision_from_dtype_flag,
    validate_fp8_inference,
)
from videotuna.utils.inference_cli import (
    add_standard_inference_flags,
    apply_compile_env,
    prepare_cli_inference_args,
    resolve_offload_mode,
)
from videotuna.utils.memory_presets import apply_memory_preset


def test_add_standard_inference_flags():
    parser = argparse.ArgumentParser()
    add_standard_inference_flags(parser)
    args = parser.parse_args(
        [
            "--device",
            "cuda:1",
            "--min-vram-gb",
            "24",
            "--memory-preset",
            "low_vram",
            "--enable_vae_tiling",
            "--enable_sequential_cpu_offload",
            "--dtype",
            "bf16",
            "--ulysses_degree",
            "2",
            "--ring_degree",
            "2",
            "--compile",
            "--enable_fp8",
        ]
    )
    assert args.device == "cuda:1"
    assert args.min_vram_gb == 24.0
    assert args.memory_preset == "low_vram"
    assert args.enable_vae_tiling is True
    assert args.enable_sequential_cpu_offload is True
    assert args.dtype == "bf16"
    assert args.ulysses_degree == 2
    assert args.compile is True
    assert args.enable_fp8 is True


def test_apply_memory_preset_low_vram():
    args = argparse.Namespace(
        memory_preset="low_vram",
        enable_model_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_vae_tiling=False,
        dtype=None,
    )
    apply_memory_preset(args)
    assert args.enable_sequential_cpu_offload is True
    assert args.enable_vae_tiling is True
    assert args.dtype == "fp16"


def test_apply_memory_preset_max_speed():
    args = argparse.Namespace(
        memory_preset="max_speed",
        enable_model_cpu_offload=True,
        enable_sequential_cpu_offload=True,
        dtype=None,
    )
    apply_memory_preset(args)
    assert args.enable_model_cpu_offload is False
    assert args.enable_sequential_cpu_offload is False
    assert args.dtype == "bf16"


def test_prepare_cli_inference_args_validates_parallel():
    args = argparse.Namespace(
        memory_preset=None,
        ulysses_degree=2,
        ring_degree=2,
        cpu_smoke=False,
        device=None,
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
    )
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "3"}):
        with pytest.raises(ValueError, match="ulysses_degree"):
            prepare_cli_inference_args(args)


def test_validate_cpu_offload_rejected_on_cpu_smoke():
    from videotuna.utils.inference_cli import validate_cpu_offload_flags

    args = argparse.Namespace(
        cpu_smoke=True,
        device=None,
        enable_sequential_cpu_offload=True,
        enable_model_cpu_offload=False,
        memory_preset=None,
    )
    with pytest.raises(RuntimeError, match="CPU offload flags"):
        validate_cpu_offload_flags(args)


def test_apply_cpu_smoke_env():
    from videotuna.utils.inference_cli import apply_cpu_smoke_env

    args = argparse.Namespace(cpu_smoke=True)
    with mock.patch.dict(os.environ, {}, clear=True):
        apply_cpu_smoke_env(args)
        assert os.environ["VIDEOTUNA_CPU_MODE"] == "smoke"
        assert os.environ["VIDEOTUNA_ATTN_BACKEND"] == "eager"
        assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "0"


@mock.patch.dict(
    os.environ,
    {"VIDEOTUNA_ATTN_BACKEND": "flash", "VIDEOTUNA_ATTN_BACKEND_STRICT": "0"},
)
def test_attn_flash_fallback_to_sdpa():
    from videotuna.utils import attention

    with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", False):
        with mock.patch.object(
            attention, "detect_compute_backend", return_value="cuda"
        ):
            with mock.patch.object(attention, "gpu_is_available", return_value=True):
                assert attention.get_attn_backend() == "sdpa"


@mock.patch.dict(
    os.environ,
    {"VIDEOTUNA_ATTN_BACKEND": "flash", "VIDEOTUNA_ATTN_BACKEND_STRICT": "1"},
)
def test_attn_flash_strict_raises():
    from videotuna.utils import attention

    with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", False):
        with mock.patch.object(
            attention, "detect_compute_backend", return_value="cuda"
        ):
            with pytest.raises(RuntimeError, match="flash-attn"):
                attention.get_attn_backend()


@pytest.mark.gpu
def test_attn_auto_resolves_on_cuda():
    from videotuna.utils.attention import get_attn_backend

    backend = get_attn_backend()
    assert backend in ("flash", "sdpa", "eager")


def test_resolve_offload_mode():
    args = argparse.Namespace(
        enable_sequential_cpu_offload=True,
        enable_model_cpu_offload=False,
    )
    assert resolve_offload_mode(args) == "sequential"
    args = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=True,
    )
    assert resolve_offload_mode(args) == "model"


def test_apply_compile_env():
    apply_compile_env(True)
    assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "1"
    apply_compile_env(False)
    assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "0"


def test_fp8_map_path():
    assert fp8_map_path("model.pt").endswith("model_map.pt")


def test_precision_from_dtype_flag():
    assert precision_from_dtype_flag("fp16") == "fp16"
    assert precision_from_dtype_flag(None, default="bf16") == "bf16"


def test_validate_fp8_inference_rejected_on_cpu():
    with mock.patch(
        "videotuna.utils.fp8_utils.detect_compute_backend", return_value="cpu"
    ):
        with pytest.raises(RuntimeError, match="not supported on CPU"):
            validate_fp8_inference("model.pt")


def test_validate_fp8_inference_missing_map():
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        with mock.patch(
            "videotuna.utils.fp8_utils.detect_compute_backend", return_value="cuda"
        ):
            with mock.patch(
                "videotuna.utils.fp8_utils.gpu_is_available", return_value=False
            ):
                with mock.patch(
                    "videotuna.utils.fp8_utils.fp8_dtype_available", return_value=True
                ):
                    mock_torchao = mock.MagicMock()
                    with mock.patch.dict("sys.modules", {"torchao": mock_torchao}):
                        with pytest.raises(FileNotFoundError):
                            validate_fp8_inference(tmp.name)


@mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "eager"})
def test_monitor_resources_returns_extended_keys():
    @monitor_resources(return_metrics=True, frames=10)
    def dummy():
        return "ok"

    out = dummy()
    assert out["result"] == "ok"
    assert "peak_vram_gb" in out
    assert "seconds_per_frame" in out
    assert out["attention_backend"] == "eager"
    assert "torch_compile" in out


def test_hyvideo_cfgdistill_no_duplicate_guidance_embed():
    from videotuna.models.hunyuan.hyvideo_i2v.modules.models import (
        HYVideoDiffusionTransformerWrapper,
    )

    wrapper = HYVideoDiffusionTransformerWrapper(
        device="cpu",
        precision="bf16",
        i2v_mode=False,
        embedded_cfg_scale=6.0,
        model="HYVideo-T/2-cfgdistill",
        ckpt_path="checkpoints/hunyuanvideo/HunyuanVideo",
        dit_weight="dummy.pt",
    )
    assert wrapper.model.guidance_embed is True


def test_require_accelerator_for_flow_raises_without_gpu():
    import torch

    from videotuna.utils.device_utils import require_accelerator_for_flow

    if torch.cuda.is_available():
        require_accelerator_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")
        return
    with pytest.raises(RuntimeError, match="requires a GPU"):
        require_accelerator_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")


def test_require_nvidia_cuda_alias_raises_without_gpu():
    import torch

    from videotuna.utils.device_utils import require_nvidia_cuda_for_flow

    if torch.cuda.is_available():
        require_nvidia_cuda_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")
        return
    with pytest.raises(RuntimeError, match="requires a GPU"):
        require_nvidia_cuda_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")


def test_save_metrics_writes_metrics_json():
    with tempfile.TemporaryDirectory() as tmp:
        save_metrics(
            savedir=tmp,
            gpu=[1.5],
            time=[10.0],
            frames=5,
        )
        path = os.path.join(tmp, "metrics.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "per_sample" in data
        assert os.path.exists(os.path.join(tmp, "metric.json"))
