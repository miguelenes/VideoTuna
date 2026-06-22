"""Tests for PEFT LoRA helpers."""

import torch.nn as nn

from videotuna.utils.lora_utils import resolve_lora_target_modules


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_resolve_all_linear():
    model = _TinyModel()
    assert resolve_lora_target_modules(model, "all-linear") == "all-linear"


def test_resolve_explicit_list():
    model = _TinyModel()
    targets = resolve_lora_target_modules(model, ["linear"])
    assert targets == ["linear"]
