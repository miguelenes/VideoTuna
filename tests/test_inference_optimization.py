"""Tests for inference CLI, metrics, and FP8 validation."""

import argparse
import json
import os
import tempfile
from unittest import mock

import pytest

from videotuna.utils.inference_cli import (
    add_standard_inference_flags,
    apply_compile_env,
    resolve_offload_mode,
)
from videotuna.utils.fp8_utils import (
    fp8_map_path,
    precision_from_dtype_flag,
    validate_fp8_inference,
)
from videotuna.utils.common_utils import monitor_resources, save_metrics


def test_add_standard_inference_flags():
    parser = argparse.ArgumentParser()
    add_standard_inference_flags(parser)
    args = parser.parse_args(
        [
            "--enable_vae_tiling",
            "--enable_sequential_cpu_offload",
            "--dtype",
            "bf16",
            "--ulysses_degree",
            "2",
            "--ring_degree",
            "1",
            "--compile",
            "--enable_fp8",
        ]
    )
    assert args.enable_vae_tiling is True
    assert args.enable_sequential_cpu_offload is True
    assert args.dtype == "bf16"
    assert args.ulysses_degree == 2
    assert args.compile is True
    assert args.enable_fp8 is True


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


def test_validate_fp8_inference_missing_map():
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
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


def test_require_nvidia_cuda_raises_without_gpu():
    from videotuna.utils.device_utils import require_nvidia_cuda_for_flow
    import torch

    if torch.cuda.is_available():
        require_nvidia_cuda_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")
        return
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
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
