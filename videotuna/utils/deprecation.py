"""Shared deprecation helpers for legacy CLI entry points."""

from __future__ import annotations

import sys
import warnings

REMOVAL_VERSION = "0.3.0"


def _ensure_deprecation_warnings_visible() -> None:
    if not sys.warnoptions:
        warnings.simplefilter("default", DeprecationWarning)


def warn_deprecated_cli_alias(
    legacy: str,
    canonical: str,
    *,
    kind: str = "Poetry script",
) -> None:
    """Emit a visible DeprecationWarning for a legacy CLI alias."""
    _ensure_deprecation_warnings_visible()
    warnings.warn(
        f"{kind} `{legacy}` is deprecated; use `{canonical}` instead. "
        f"Removal planned in v{REMOVAL_VERSION}.",
        DeprecationWarning,
        stacklevel=3,
    )


def warn_deprecated_inference_script() -> None:
    """Warn when invoking ``python scripts/inference_new.py`` directly."""
    _ensure_deprecation_warnings_visible()
    warnings.warn(
        "Direct invocation of `scripts/inference_new.py` is deprecated; "
        "use `poetry run inference-run` for generic YAML inference, or a "
        "preset command such as `validate-domain-t2v` / `inference-domain-t2i`. "
        f"Removal planned in v{REMOVAL_VERSION}.",
        DeprecationWarning,
        stacklevel=2,
    )
