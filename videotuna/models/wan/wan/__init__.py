# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import importlib
from typing import Any, List

from . import configs

__all__ = ["WanT2V", "WanI2V", "configs"]

_LAZY_IMPORTS: dict[str, str] = {
    "WanT2V": ".text2video",
    "WanI2V": ".image2video",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name, module)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals()) + list(_LAZY_IMPORTS) + ["configs"]))
