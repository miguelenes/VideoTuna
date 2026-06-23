"""Tests for Wan 2.1 native I2V LoRA → Wan 2.2 I2V Diffusers bridge."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from videotuna.utils.wan_lora_bridge import (
    MIN_REMAP_COVERAGE,
    apply_native_wan_lora_to_i2v_pipeline,
)


def test_i2v_bridge_delegates_to_t2v_bridge():
    pipeline = MagicMock()
    with patch(
        "videotuna.utils.wan_lora_bridge.apply_native_wan_lora_to_pipeline"
    ) as mock_apply:
        mock_apply.return_value = []
        apply_native_wan_lora_to_i2v_pipeline(
            pipeline, "/tmp/denoiser.ckpt", lora_scale=0.8
        )
        mock_apply.assert_called_once_with(
            pipeline,
            "/tmp/denoiser.ckpt",
            lora_scale=0.8,
            lora_scale_2=None,
            mode="i2v",
        )


def _tiny_i2v_transformer() -> WanTransformer3DModel:
    cfg = WanTransformer3DModel.load_config(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer"
    )
    cfg["num_layers"] = 1
    return WanTransformer3DModel.from_config(cfg)


def _i2v_native_keys() -> dict[str, torch.Tensor]:
    dim_in, dim_mid, dim_out = 5120, 13824, 5120
    rank = 16
    state: dict[str, torch.Tensor] = {}
    for p in ("q", "k", "v", "o"):
        state[f"blocks.0.self_attn.{p}.lora_A.weight"] = torch.zeros(rank, dim_in)
        state[f"blocks.0.self_attn.{p}.lora_B.weight"] = torch.zeros(dim_in, rank)
        state[f"blocks.0.cross_attn.{p}.lora_A.weight"] = torch.zeros(rank, dim_in)
        state[f"blocks.0.cross_attn.{p}.lora_B.weight"] = torch.zeros(dim_in, rank)
    state["blocks.0.ffn.0.lora_A.weight"] = torch.zeros(rank, dim_in)
    state["blocks.0.ffn.0.lora_B.weight"] = torch.zeros(dim_mid, rank)
    state["blocks.0.ffn.2.lora_A.weight"] = torch.zeros(rank, dim_mid)
    state["blocks.0.ffn.2.lora_B.weight"] = torch.zeros(dim_out, rank)
    return state


def test_i2v_bridge_loads_cross_attn():
    ckpt_state = _i2v_native_keys()
    ckpt_path = MagicMock()
    transformer = _tiny_i2v_transformer()
    pipeline = SimpleNamespace(transformer=transformer, transformer_2=None)

    with patch(
        "videotuna.utils.wan_lora_bridge.load_native_wan_lora_state_dict",
        return_value=ckpt_state,
    ):
        reports = apply_native_wan_lora_to_i2v_pipeline(pipeline, ckpt_path)

    assert len(reports) == 1
    assert reports[0].expert == "i2v_transformer"
    assert reports[0].remap_ratio >= MIN_REMAP_COVERAGE
    assert reports[0].unmapped_keys == []
    assert reports[0].loaded_lora_params > 0
    assert any(
        "attn2" in name and "lora" in name.lower()
        for name, _ in pipeline.transformer.named_parameters()
    )
