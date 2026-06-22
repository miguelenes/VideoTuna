"""PEFT LoRA target-module resolution helpers."""

from __future__ import annotations

from typing import List, Union

import torch.nn as nn


def resolve_lora_target_modules(
    model: nn.Module,
    target_modules: Union[str, List[str], None],
) -> Union[str, List[str]]:
    """Resolve LoRA target modules from explicit lists or PEFT shortcuts."""
    if target_modules is None:
        raise ValueError("target_modules must be provided for LoRA configuration")

    if target_modules == "all-linear":
        return "all-linear"

    if isinstance(target_modules, str):
        if target_modules == "kappa":
            return _kappa_targets(model)
        return [target_modules]

    if isinstance(target_modules, list):
        return target_modules

    raise TypeError(f"Unsupported target_modules type: {type(target_modules)}")


def _kappa_targets(model: nn.Module) -> List[str]:
    try:
        from peft.helpers import find_kappa_target_modules
    except ImportError as exc:
        raise ImportError(
            "kappa target discovery requires peft.helpers.find_kappa_target_modules"
        ) from exc

    targets = find_kappa_target_modules(model, top_p=0.2)
    resolved = targets.get("target_modules") or []
    if not resolved:
        raise ValueError("find_kappa_target_modules returned no target_modules")
    return resolved


def collect_lora_parameter_names(model: nn.Module) -> set[str]:
    """Return parameter names that belong to LoRA adapters."""
    return {name for name, _ in model.named_parameters() if "lora" in name.lower()}
