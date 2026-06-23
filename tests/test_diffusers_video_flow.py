"""Unit tests for the PrivTune Diffusers inference flow (Flux + Wan 2.2)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

from videotuna.flow.diffusers_video import (
    MODEL_REGISTRY,
    DiffusersVideoFlow,
    resolve_model_id,
    resolve_torch_dtype,
)
from videotuna.utils.diffusers_optimizations import (
    apply_diffusers_optimizations,
    transformer_cache_context,
)


def test_resolve_model_id_defaults():
    assert (
        resolve_model_id("flux", "t2i", None, model_variant="1-dev")
        == "black-forest-labs/FLUX.1-dev"
    )
    assert (
        resolve_model_id("wan", "t2v", None, model_variant="2.2")
        == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    assert resolve_model_id("wan", "t2v", "custom/model") == "custom/model"


def test_resolve_model_id_variant_fallback():
    assert (
        resolve_model_id("wan", "t2v", None, model_variant="2.1")
        == "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )
    assert (
        resolve_model_id("wan", "t2v", None, model_variant="unknown")
        == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )


def test_resolve_model_id_i2v_defaults():
    assert (
        resolve_model_id("wan", "i2v", None, model_variant="2.2")
        == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    )
    assert (
        resolve_model_id("wan", "i2v", None, model_variant="2.1")
        == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    )


def test_resolve_model_id_unknown_family_raises():
    with pytest.raises(ValueError, match="Unsupported diffusers model"):
        resolve_model_id("cogvideox", "t2v", None)


def test_resolve_model_id_no_variant_uses_default():
    assert resolve_model_id("flux", "t2i", None) == "black-forest-labs/FLUX.1-dev"


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("fp16") == torch.float16
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    assert resolve_torch_dtype(None) == torch.bfloat16


def test_resolve_torch_dtype_float16_alias():
    assert resolve_torch_dtype("float16") == torch.float16


def test_model_registry_covers_domain_families():
    assert ("flux", "t2i") in MODEL_REGISTRY
    assert ("wan", "t2v") in MODEL_REGISTRY
    assert ("wan", "i2v") in MODEL_REGISTRY
    assert ("cogvideox", "t2v") not in MODEL_REGISTRY


def test_model_registry_entries_have_pipeline_cls():
    for key, entry in MODEL_REGISTRY.items():
        assert "pipeline_cls" in entry
        assert "default_id" in entry


class TestDiffusersVideoFlowConfig:
    """Config-only tests for DiffusersVideoFlow (no weights loaded)."""

    def test_init_sets_pipeline_only_true(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        assert flow.pipeline_only is True
        assert flow.pipeline is None
        assert flow.model_family == "wan"
        assert flow.mode == "t2v"
        assert flow._model_id is None

    def test_init_stores_all_kwargs(self):
        flow = DiffusersVideoFlow(
            model_family="flux",
            mode="t2i",
            pretrained_model_name_or_path="some/model",
            model_variant="schnell",
            lora_rank=64,
            fuse_qkv=True,
            enable_attention_cache=True,
        )
        assert flow.pretrained_model_name_or_path == "some/model"
        assert flow.model_variant == "schnell"
        assert flow.lora_rank == 64
        assert flow.fuse_qkv is True
        assert flow.enable_attention_cache is True

    def test_from_pretrained_resolves_model_id_with_pretrained_path(self):
        flow = DiffusersVideoFlow(
            model_family="flux",
            mode="t2i",
            pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell",
        )
        flow.from_pretrained()
        assert flow._model_id == "black-forest-labs/FLUX.1-schnell"

    def test_from_pretrained_resolves_model_id_with_ckpt_path(self):
        flow = DiffusersVideoFlow(
            model_family="wan",
            mode="t2v",
            pretrained_model_name_or_path="default/model",
        )
        flow.from_pretrained(ckpt_path="explicit/ckpt")
        assert flow._model_id == "explicit/ckpt"

    def test_from_pretrained_resolves_variant_when_no_path(self):
        flow = DiffusersVideoFlow(
            model_family="flux",
            mode="t2i",
            model_variant="schnell",
        )
        flow.from_pretrained()
        assert flow._model_id == "black-forest-labs/FLUX.1-schnell"

    def test_from_pretrained_sets_lora_path(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(lora_ckpt_path="/tmp/my_lora.safetensors")
        assert flow._lora_path == "/tmp/my_lora.safetensors"
        assert flow._model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    def test_from_pretrained_sets_denoiser_as_lora_for_wan(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(denoiser_ckpt_path="/tmp/denoiser.ckpt")
        assert flow._lora_path == "/tmp/denoiser.ckpt"

    def test_from_pretrained_no_denoiser_as_lora_for_flux(self):
        flow = DiffusersVideoFlow(model_family="flux", mode="t2i")
        flow.from_pretrained(denoiser_ckpt_path="/tmp/denoiser.ckpt")
        assert flow._lora_path is None

    def test_from_pretrained_lora_overrides_denoiser(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(
            lora_ckpt_path="/tmp/lora.safetensors",
            denoiser_ckpt_path="/tmp/denoiser.ckpt",
        )
        assert flow._lora_path == "/tmp/lora.safetensors"

    def test_from_pretrained_stores_device(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(device="cuda:0")
        assert flow._inference_device == "cuda:0"

    def test_resolve_model_id_unsupported_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported diffusers model"):
            resolve_model_id("unknown", "t2v", None)

    def test_resolve_model_id_case_insensitive(self):
        assert resolve_model_id("Flux", "T2I", "my/model") == "my/model"
        assert (
            resolve_model_id("WAN", "T2V", None, model_variant="2.2")
            == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )

    def test_resolve_model_id_no_variant_returns_default(self):
        assert resolve_model_id("flux", "t2i", None) == "black-forest-labs/FLUX.1-dev"

    def test_resolve_model_id_variant_no_variants_entry(self):
        entry = MODEL_REGISTRY[("wan", "t2v")]
        saved = entry.pop("variants", None)
        try:
            assert (
                resolve_model_id("wan", "t2v", None, model_variant="2.2")
                == entry["default_id"]
            )
        finally:
            if saved is not None:
                entry["variants"] = saved

    def test_init_default_lora_rank_and_weight_name(self):
        flow = DiffusersVideoFlow(model_family="flux", mode="t2i")
        assert flow.lora_rank == 128
        assert flow.lora_weight_name == "pytorch_lora_weights.safetensors"

    def test_init_default_internal_state(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        assert flow._dtype == torch.bfloat16
        assert flow._transformer_quant == "none"
        assert flow._quant_backend == "torchao"
        assert flow._model_id is None
        assert flow._lora_path is None
        assert flow._inference_device is None

    def test_init_stores_name_or_path_as_none_when_omitted(self):
        flow = DiffusersVideoFlow(model_family="flux", mode="t2i")
        assert flow.pretrained_model_name_or_path is None

    def test_init_normalizes_model_family_and_mode(self):
        flow = DiffusersVideoFlow(model_family="FLUX", mode="T2I")
        assert flow.model_family == "flux"
        assert flow.mode == "t2i"

    def test_from_pretrained_extra_kwargs_accepted(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(ckpt_path="some/model", extra_flag=True)
        assert flow._model_id == "some/model"

    def test_from_pretrained_denoiser_not_set_for_non_wan(self):
        flow = DiffusersVideoFlow(model_family="flux", mode="t2i")
        flow.from_pretrained(denoiser_ckpt_path="/tmp/denoiser.ckpt")
        assert flow._lora_path is None

    def test_from_pretrained_pretrained_name_or_path_as_fallback(self):
        flow = DiffusersVideoFlow(
            model_family="flux",
            mode="t2i",
            pretrained_model_name_or_path="stabilityai/stable-diffusion",
        )
        flow.from_pretrained()
        assert flow._model_id == "stabilityai/stable-diffusion"


class TestDiffusersVideoFlowConfigExtra:
    """Additional CPU-only, state-only config coverage for DiffusersVideoFlow."""

    def test_init_pipeline_only_is_forced_true(self):
        flow = DiffusersVideoFlow(model_family="flux", mode="t2i", pipeline_only=False)
        assert flow.pipeline_only is True

    def test_from_pretrained_ignore_missing_ckpts_is_noop(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
        flow.from_pretrained(ignore_missing_ckpts=True)
        assert flow._model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert flow._lora_path is None

    def test_from_pretrained_denoiser_for_wan_i2v(self):
        flow = DiffusersVideoFlow(model_family="wan", mode="i2v")
        flow.from_pretrained(denoiser_ckpt_path="/tmp/denoiser.ckpt")
        assert flow._lora_path == "/tmp/denoiser.ckpt"
        assert flow._model_id == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    def test_resolve_model_id_wan_i2v_unknown_variant(self):
        assert (
            resolve_model_id("wan", "i2v", None, model_variant="unknown")
            == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        )


def test_apply_diffusers_optimizations_mock_pipe():
    pipe = mock.MagicMock()
    pipe.vae = mock.MagicMock()
    del pipe.enable_vae_tiling
    args = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=True,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
        fuse_qkv=True,
        enable_attention_cache=False,
    )
    apply_diffusers_optimizations(pipe, args)
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.vae.enable_slicing.assert_called_once()
    pipe.vae.enable_tiling.assert_called_once()
    pipe.fuse_qkv_projections.assert_called_once()
    pipe.set_progress_bar_config.assert_called_once()


def test_transformer_cache_context_noop_without_transformer():
    pipe = SimpleNamespace(transformer=None)
    with transformer_cache_context(pipe):
        pass


def test_diffusers_video_flow_instantiate_pipeline_only():
    flow = DiffusersVideoFlow(
        model_family="wan",
        mode="t2v",
        pretrained_model_name_or_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    assert flow.pipeline_only is True
    assert flow.pipeline is None


@mock.patch("videotuna.flow.diffusers_video.get_settings")
@mock.patch.object(DiffusersVideoFlow, "_generate_sample")
@mock.patch.object(DiffusersVideoFlow, "save_output")
@mock.patch.object(DiffusersVideoFlow, "save_manifest")
@mock.patch.object(DiffusersVideoFlow, "save_metrics")
def test_inference_t2v_saves_video(
    mock_save_metrics,
    mock_save_manifest,
    mock_save_output,
    mock_generate,
    mock_get_settings,
):
    mock_get_settings.return_value.metrics_owner = "flow"
    mock_generate.return_value = {
        "result": [{"frame": 0}],
        "peak_vram_gb": 1.0,
        "wall_time_s": 2.0,
    }
    flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
    flow.pipeline = mock.MagicMock()
    args = OmegaConf.create(
        {
            "savedir": "/tmp/privtune-test",
            "prompt_file": "hello world",
            "frames": 49,
            "num_inference_steps": 4,
            "unconditional_guidance_scale": 6.0,
            "seed": 1,
            "savefps": 8,
        }
    )
    metrics = flow.inference(args)
    assert len(metrics["per_sample"]) == 1
    mock_save_output.assert_called_once()
    mock_save_manifest.assert_called_once()
    mock_save_metrics.assert_called_once()


def test_yaml_wan22_instantiates_flow():
    from videotuna.flow.factories import build_flow

    cfg = OmegaConf.load("configs/inference/presets/balanced_wan2_2_720p.yaml")
    flow = build_flow(cfg.flow)
    assert isinstance(flow, DiffusersVideoFlow)
    assert flow.model_variant == "2.2"


@pytest.mark.gpu
def test_wan22_4step_deterministic_gpu_smoke(tmp_path):
    """GPU regression: two Wan 2.2 4-step generations with seed=42 must be identical."""
    import torch
    from diffusers import WanPipeline

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def _run():
        generator = torch.Generator(device="cuda").manual_seed(42)
        return pipe(
            prompt="a beautiful sunset over the ocean, cinematic",
            num_frames=5,
            height=256,
            width=448,
            num_inference_steps=4,
            guidance_scale=5.0,
            generator=generator,
        )

    result_a = _run()
    result_b = _run()

    assert result_a is not None
    assert result_b is not None
    assert len(result_a.frames[0]) == 5
    assert len(result_b.frames[0]) == 5

    va = torch.stack([torch.as_tensor(f) for f in result_a.frames[0]])
    vb = torch.stack([torch.as_tensor(f) for f in result_b.frames[0]])
    assert torch.equal(va, vb), "4-step GPU generations diverged — regression detected"

    from diffusers.utils import export_to_video

    savedir = tmp_path / "smoke"
    savedir.mkdir()
    export_to_video(result_a.frames[0], str(savedir / "run_a.mp4"), fps=8)
    export_to_video(result_b.frames[0], str(savedir / "run_b.mp4"), fps=8)
    assert (savedir / "run_a.mp4").exists()
    assert (savedir / "run_b.mp4").exists()
