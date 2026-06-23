"""Tests for Wan 2.1 native LoRA → Wan 2.2 Diffusers bridge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from videotuna.testing.wan_lora_ckpt import build_synthetic_wan_lora_ckpt
from videotuna.utils.wan_lora_bridge import (
    MIN_REMAP_COVERAGE,
    WAN_DIFFUSERS_LORA_TARGETS,
    WanBridgeConfig,
    _infer_lora_rank,
    _remap_native_to_diffusers_keys,
    _remap_single_native_key,
    analyze_native_wan_lora_ckpt,
    apply_native_wan_lora_to_pipeline,
    build_bridge_key_map,
    compute_remap_coverage,
    export_diffusers_lora_state_dicts,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
    validate_remap_coverage,
    verify_runtime_export_parity,
)


def _production_native_keys(
    *, block: int = 0, rank: int = 16
) -> dict[str, torch.Tensor]:
    dim_in, dim_mid, dim_out = 5120, 13824, 5120
    state: dict[str, torch.Tensor] = {}
    for p in ("q", "k", "v", "o"):
        out_dim = dim_in if p != "o" else dim_in
        state[f"blocks.{block}.self_attn.{p}.lora_A.weight"] = torch.zeros(rank, dim_in)
        state[f"blocks.{block}.self_attn.{p}.lora_B.weight"] = torch.zeros(
            out_dim, rank
        )
        state[f"blocks.{block}.cross_attn.{p}.lora_A.weight"] = torch.zeros(
            rank, dim_in
        )
        state[f"blocks.{block}.cross_attn.{p}.lora_B.weight"] = torch.zeros(
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
        {"state_dict": {"blocks.0.self_attn.q.lora_A.weight": torch.zeros(16, 5120)}},
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


def test_remap_cross_attn_keys():
    assert (
        _remap_single_native_key("blocks.0.cross_attn.q.lora_A.weight")
        == "blocks.0.attn2.to_q.lora_A.weight"
    )
    assert (
        _remap_single_native_key("blocks.2.cross_attn.o.lora_B.weight")
        == "blocks.2.attn2.to_out.0.lora_B.weight"
    )


def test_analyze_native_wan_lora_ckpt(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": {f"denoiser.{k}": v for k, v in state.items()}}, ckpt)
    info = analyze_native_wan_lora_ckpt(ckpt)
    assert info["native_key_count"] == 20
    assert info["remap_coverage"] == 1.0
    assert info["unmapped_keys"] == []
    assert info["rank"] == 16


def test_export_diffusers_lora_state_dicts(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": state}, ckpt)
    exports = export_diffusers_lora_state_dicts(ckpt)
    assert "high_noise" in exports and "low_noise" in exports
    assert "blocks.0.attn1.to_q.lora_A.weight" in exports["high_noise"]
    assert "blocks.0.attn2.to_q.lora_A.weight" in exports["high_noise"]


def test_exported_lora_loads_via_diffusers_adapter(tmp_path):
    """Offline export path: safetensors → WanTransformer3DModel.load_lora_adapter."""
    from safetensors.torch import save_file

    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": state}, ckpt)
    exports = export_diffusers_lora_state_dicts(ckpt)
    lora_path = tmp_path / "high_noise.safetensors"
    save_file(exports["high_noise"], lora_path)

    transformer = _tiny_transformer()
    transformer.load_lora_adapter(str(lora_path), adapter_name="exported", prefix=None)
    transformer.set_adapters(["exported"], weights=[1.0])
    assert _count_lora(transformer) == 20


def test_remap_coverage_on_production_fixture():
    native = _production_native_keys()
    transformed, total, coverage = compute_remap_coverage(native)
    assert total == 20
    assert transformed == 20
    assert coverage >= MIN_REMAP_COVERAGE


def test_validate_remap_coverage_below_threshold_raises():
    native = _production_native_keys()
    native["blocks.0.unmapped.extra.weight"] = torch.zeros(1)
    native["blocks.0.unmapped.another.weight"] = torch.zeros(1)
    native["blocks.0.unmapped.third.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        validate_remap_coverage(native)


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
    assert reports[0].remap_ratio >= MIN_REMAP_COVERAGE
    assert reports[0].missing_keys == []
    assert reports[0].unmapped_keys == []
    assert reports[0].renamed_keys
    assert reports[0].loaded_lora_params == 20
    assert _count_lora(pipeline.transformer) == 20


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
    for report in reports:
        assert report.remap_ratio >= MIN_REMAP_COVERAGE
        assert report.missing_keys == []
        assert report.unmapped_keys == []
        assert report.renamed_keys
    pipeline.set_adapters.assert_called_once()
    call = pipeline.set_adapters.call_args
    adapters = call.args[0] if call.args else call.kwargs["adapter_names"]
    weights = call.kwargs.get("adapter_weights")
    if weights is None and len(call.args) > 1:
        weights = call.args[1]
    assert len(adapters) == 2
    assert weights == [1.0, 1.0]


def test_apply_native_wan_lora_to_pipeline_raises_on_low_coverage():
    ckpt_state = _production_native_keys()
    ckpt_state["blocks.0.unmapped.extra.weight"] = torch.zeros(1)
    ckpt_state["blocks.0.unmapped.another.weight"] = torch.zeros(1)
    ckpt_state["blocks.0.unmapped.third.weight"] = torch.zeros(1)
    ckpt_path = MagicMock()
    pipeline = SimpleNamespace(transformer=_tiny_transformer(), transformer_2=None)

    with patch(
        "videotuna.utils.wan_lora_bridge.load_native_wan_lora_state_dict",
        return_value=ckpt_state,
    ):
        with pytest.raises(RuntimeError):
            apply_native_wan_lora_to_pipeline(pipeline, ckpt_path)


def test_renamed_keys_content():
    """renamed_keys pairs must match expected old→new mapping for each pattern."""
    from videotuna.utils.wan_lora_bridge import _remap_state_with_meta

    native = _production_native_keys()
    _, _, renamed = _remap_state_with_meta(native)
    assert len(renamed) == 20
    mapping = dict(renamed)

    # self_attn.q → attn1.to_q
    assert (
        mapping["blocks.0.self_attn.q.lora_A.weight"]
        == "blocks.0.attn1.to_q.lora_A.weight"
    )
    # self_attn.o → attn1.to_out.0
    assert (
        mapping["blocks.0.self_attn.o.lora_B.weight"]
        == "blocks.0.attn1.to_out.0.lora_B.weight"
    )
    # cross_attn.q → attn2.to_q
    assert (
        mapping["blocks.0.cross_attn.q.lora_A.weight"]
        == "blocks.0.attn2.to_q.lora_A.weight"
    )
    # cross_attn.o → attn2.to_out.0
    assert (
        mapping["blocks.0.cross_attn.o.lora_B.weight"]
        == "blocks.0.attn2.to_out.0.lora_B.weight"
    )
    # ffn.0 → ffn.net.0.proj
    assert (
        mapping["blocks.0.ffn.0.lora_A.weight"]
        == "blocks.0.ffn.net.0.proj.lora_A.weight"
    )
    # ffn.2 → ffn.net.2
    assert mapping["blocks.0.ffn.2.lora_B.weight"] == "blocks.0.ffn.net.2.lora_B.weight"


def test_bridge_config_from_env(monkeypatch):
    """VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE env var overrides default threshold."""
    config = WanBridgeConfig.from_env()
    assert config.min_coverage == MIN_REMAP_COVERAGE  # no env → default

    monkeypatch.setenv("VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE", "0.85")
    config = WanBridgeConfig.from_env()
    assert config.min_coverage == 0.85

    monkeypatch.setenv("VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE", "0.95")
    config = WanBridgeConfig.from_env()
    assert config.min_coverage == 0.95


def test_configurable_coverage_threshold(tmp_path):
    """Bridge accepts configurable min_coverage via WanBridgeConfig."""
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    # Add 2 unmapped keys → coverage ≈ 20/22 ≈ 0.909
    state["blocks.0.unknown.layer.lora_A.weight"] = torch.zeros(16, 5120)
    state["blocks.0.unknown.layer.lora_B.weight"] = torch.zeros(5120, 16)
    torch.save({"state_dict": state}, ckpt)

    # Strict threshold above coverage should raise.
    strict_config = WanBridgeConfig(min_coverage=0.95)
    with pytest.raises(RuntimeError, match="remap coverage"):
        export_diffusers_lora_state_dicts(ckpt, bridge_config=strict_config)

    # Lenient threshold below coverage should pass.
    lenient_config = WanBridgeConfig(min_coverage=0.85)
    exports = export_diffusers_lora_state_dicts(ckpt, bridge_config=lenient_config)
    assert "high_noise" in exports


def test_runtime_export_parity_pass(tmp_path):
    """Runtime bridge remap and export remap produce identical key sets."""
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": state}, ckpt)

    parity = verify_runtime_export_parity(ckpt)
    assert parity.keys_match is True
    assert parity.runtime_key_count == parity.export_key_count
    assert parity.only_in_export == []


def test_key_diff_structure(tmp_path):
    """build_bridge_key_map returns one entry per native key with correct status."""
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    state["blocks.0.unknown.extra.weight"] = torch.zeros(1)
    torch.save({"state_dict": state}, ckpt)

    native_state = load_native_wan_lora_state_dict(ckpt)
    key_map = build_bridge_key_map(native_state)

    assert len(key_map) == len(native_state)
    for entry in key_map:
        assert entry.native_key in native_state
        if entry.status == "remapped":
            assert entry.diffusers_key is not None
            assert entry.pattern is not None
        elif entry.status == "unmapped":
            assert entry.diffusers_key is None
            assert entry.pattern is None


def test_export_include_key_diff(tmp_path):
    """export_diffusers_lora_state_dicts with include_key_diff=True returns diff."""
    ckpt = tmp_path / "denoiser.ckpt"
    state = _production_native_keys()
    torch.save({"state_dict": state}, ckpt)

    exports = export_diffusers_lora_state_dicts(ckpt, include_key_diff=True)
    assert "_parity" in exports
    assert exports["_parity"]["keys_match"] is True
    assert "_key_diff" in exports
    assert len(exports["_key_diff"]) == 20
    for entry in exports["_key_diff"]:
        assert "native_key" in entry
        assert "status" in entry


def _count_lora(module: WanTransformer3DModel) -> int:
    return sum(1 for n, _ in module.named_parameters() if "lora" in n.lower())


@pytest.mark.gpu
def test_validate_domain_t2v_gpu_smoke(tmp_path):
    """GPU integration: bridge + pipeline load (skipped without CUDA/ROCm)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    ckpt = build_synthetic_wan_lora_ckpt(tmp_path / "denoiser.ckpt", num_blocks=2)

    from diffusers import WanPipeline

    pipeline = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype=torch.bfloat16,
    )
    reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt)
    assert len(reports) == 2
    assert {r.expert for r in reports} == {"transformer", "transformer_2"}
    for report in reports:
        assert report.remap_ratio >= MIN_REMAP_COVERAGE
        assert report.missing_keys == []
        assert report.unmapped_keys == []
        assert report.loaded_lora_params > 0
    assert WAN_DIFFUSERS_LORA_TARGETS
