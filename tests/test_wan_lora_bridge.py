"""Tests for Wan 2.1 native LoRA → Wan 2.2 Diffusers bridge helpers."""

from __future__ import annotations

import torch

from videotuna.utils.wan_lora_bridge import (
    _infer_lora_rank,
    _remap_native_to_diffusers_keys,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
)


def test_load_native_wan_lora_state_dict_filters_non_lora(tmp_path):
    ckpt = tmp_path / "denoiser.ckpt"
    state = {
        "denoiser.blocks.0.attn.q.lora_A.weight": torch.zeros(16, 4),
        "denoiser.blocks.0.attn.q.lora_B.weight": torch.zeros(4, 16),
        "denoiser.blocks.0.attn.q.weight": torch.zeros(4, 4),
    }
    torch.save({"state_dict": state}, ckpt)
    loaded = load_native_wan_lora_state_dict(ckpt)
    assert len(loaded) == 2
    assert all("lora" in k for k in loaded)
    assert loaded["blocks.0.attn.q.lora_A.weight"].shape == (16, 4)


def test_is_native_wan_lora_ckpt(tmp_path):
    ckpt = tmp_path / "lora.ckpt"
    torch.save(
        {"state_dict": {"blocks.0.attn.q.lora_A.weight": torch.zeros(16, 4)}},
        ckpt,
    )
    assert is_native_wan_lora_ckpt(ckpt)
    assert not is_native_wan_lora_ckpt(tmp_path / "missing.ckpt")


def test_infer_lora_rank():
    state = {"blocks.0.attn.q.lora_A.weight": torch.zeros(16, 8)}
    assert _infer_lora_rank(state) == 16


def test_remap_blocks_to_transformer_blocks():
    native = {"blocks.0.attn.q.lora_A.weight": torch.zeros(1)}
    remapped = _remap_native_to_diffusers_keys(native)
    assert "transformer_blocks.0.attn.q.lora_A.weight" in remapped
