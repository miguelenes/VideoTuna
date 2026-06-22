"""Tests for Wan 2.1 native LoRA → Wan 2.2 Diffusers bridge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from videotuna.utils.wan_lora_bridge import (
    WAN_DIFFUSERS_LORA_TARGETS,
    _infer_lora_rank,
    _remap_native_to_diffusers_keys,
    _remap_single_native_key,
    analyze_native_wan_lora_ckpt,
    apply_native_wan_lora_to_pipeline,
    export_diffusers_lora_state_dicts,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
)


def _production_native_keys(
    *, block: int = 0, rank: int = 16
) -> dict[str, torch.Tensor]:
    dim_in, dim_mid, dim_out = 5120, 13824, 5120
    state: dict[str, torch.Tensor] = {}
    for p in ("q", "k", "v", "o"):
        out_dim = dim_in if p != "o" else dim_in
        state[f"blocks.{block}.self_attn.{p}.lora_A.weight"] = torch.zeros(
            rank, dim_in
        )
        state[f"blocks.{block}.self_attn.{p}.lora_B.weight"] = torch.zeros(
            out_dim, rank
        )
    state[f"blocks.{block}.ffn.0.lora_A.weight"] = torch.zeros(rank, dim_in)
    state[f"blocks.{block}.ffn.0.lora_B.weight"] = torch.zeros(dim_mid, rank)
    state[f"blocks.{block}.ffn.2.lora_A.weight"] = torch.zeros(rank, dim_mid)
    state[f"blocks.{block}.ffn.2.lora_B.weight"] = torch.zeros(dim_out, rank)
    return state


def _tiny_transformer() -> WanTransformer3DModel:
    cfg = WanTransformer3DModel.load_config(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer"
    )
    cfg["num_layers"] = 1
    return WanTransformer3DModel.from_config(cfg)


def test_load_native_wan_lora_state_dict_filters_non_lora(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = {
        "denoiser.blocks.0.self_attn.q.lora_A.weight": torch.zeros(16, 5120),
        "denoiser.blocks.0.self_attn.q.lora_B.weight": torch.zeros(5120, 16),
        "denoiser.blocks.0.self_attn.q.weight": torch.zeros(4, 4),
    }
    torch.save({"state_dict": state}, ckpt)
    loaded = load_native_wan_lora_state_dict(ckpt)
    assert len(loaded) == 2
    assert all("lora" in k for k in loaded)
    assert loaded["blocks.0.self_attn.q.lora_A.weight"].shape == (16, 5120)


def test_is_native_wan_lora_ckpt(tmp_path):
    ckpt = tmp_path / "lora.ckpt"
    torch.save(
        {
            "state_dict": {
                "blocks.0.self_attn.q.lora_A.weight": torch.zeros(16, 5120)
            }
        },
        ckpt,
    )
    assert is_native_wan_lora_ckpt(ckpt)
    assert not is_native_wan_lora_ckpt(tmp_path / "missing.ckpt")


def test_infer_lora_rank():
    state = {"blocks.0.self_attn.q.lora_A.weight": torch.zeros(16, 8)}
    assert _infer_lora_rank(state) == 16


def test_remap_production_self_attn_keys():
    assert (
        _remap_single_native_key("blocks.0.self_attn.q.lora_A.weight")
        == "blocks.0.attn1.to_q.lora_A.weight"
    )
    assert (
        _remap_single_native_key("blocks.3.self_attn.o.lora_B.weight")
        == "blocks.3.attn1.to_out.0.lora_B.weight"
    )


def test_remap_production_ffn_keys():
    assert (
        _remap_single_native_key("blocks.1.ffn.0.lora_A.weight")
        == "blocks.1.ffn.net.0.proj.lora_A.weight"
    )
    assert (
        _remap_single_native_key("blocks.1.ffn.2.lora_B.weight")
        == "blocks.1.ffn.net.2.lora_B.weight"
    )


def test_remap_legacy_attn_shorthand():
    native = {"blocks.0.attn.q.lora_A.weight": torch.zeros(1)}
    remapped = _remap_native_to_diffusers_keys(native)
    assert "blocks.0.attn1.to_q.lora_A.weight" in remapped


def test_analyze_native_wan_lora_ckpt(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": {f"denoiser.{k}": v for k, v in state.items()}}, ckpt)
    info = analyze_native_wan_lora_ckpt(ckpt)
    assert info["native_key_count"] == 12
    assert info["rank"] == 16


def test_export_diffusers_lora_state_dicts(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": state}, ckpt)
    exports = export_diffusers_lora_state_dicts(ckpt)
    assert "high_noise" in exports and "low_noise" in exports
    assert "blocks.0.attn1.to_q.lora_A.weight" in exports["high_noise"]


def test_apply_native_wan_lora_to_single_transformer():
    ckpt_state = _production_native_keys()
    ckpt_path = MagicMock()
    transformer = _tiny_transformer()
    pipeline = SimpleNamespace(transformer=transformer, transformer_2=None)

    with patch(
        "videotuna.utils.wan_lora_bridge.load_native_wan_lora_state_dict",
        return_value=ckpt_state,
    ):
        reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt_path)

    assert len(reports) == 1
    assert reports[0].expert == "transformer"
    assert reports[0].loaded_lora_params > 0
    assert _count_lora(pipeline.transformer) > 0


def test_apply_native_wan_lora_to_dual_transformer():
    ckpt_state = _production_native_keys()
    ckpt_path = MagicMock()
    pipeline = SimpleNamespace(
        transformer=_tiny_transformer(),
        transformer_2=_tiny_transformer(),
        set_adapters=MagicMock(),
    )

    with patch(
        "videotuna.utils.wan_lora_bridge.load_native_wan_lora_state_dict",
        return_value=ckpt_state,
    ):
        reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt_path)

    assert len(reports) == 2
    assert {r.expert for r in reports} == {"transformer", "transformer_2"}
    pipeline.set_adapters.assert_called_once()
    adapters, scales = pipeline.set_adapters.call_args[0]
    assert len(adapters) == 2
    assert scales == [1.0, 1.0]


def _count_lora(module: WanTransformer3DModel) -> int:
    return sum(1 for n, _ in module.named_parameters() if "lora" in n.lower())


@pytest.mark.gpu
def test_validate_domain_t2v_gpu_smoke(tmp_path):
    """GPU integration: bridge + pipeline load (skipped without CUDA/ROCm)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys(block=0)
    # duplicate for block 1 to match 1-layer tiny model expectations minimally
    state.update(_production_native_keys(block=0))
    torch.save({"state_dict": {f"denoiser.{k}": v for k, v in state.items()}}, ckpt)

    from diffusers import WanPipeline

    pipeline = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype=torch.bfloat16,
    )
    reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt)
    assert all(r.loaded_lora_params > 0 for r in reports)
    assert WAN_DIFFUSERS_LORA_TARGETS  # referenced for coverage
