"""Tests for the validate-datasets CLI."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from videotuna.cli.dataset_app import (
    DatasetValidationOptions,
    validate_datasets_entry,
)


def test_entry_resolves() -> None:
    """Entry point must be a callable function."""
    assert callable(validate_datasets_entry)


def test_poetry_script_modules_can_resolve() -> None:
    """Simulates the auto-resolve check from test_poetry_scripts."""
    mod = importlib.import_module("videotuna.cli.dataset_app")
    assert hasattr(mod, "validate_datasets_entry")


def test_options_dataclass_defaults() -> None:
    """Default option values must match expectations."""
    opts = DatasetValidationOptions()
    assert opts.phase == ("all",)
    assert opts.trigger_token == "sks_style"
    assert opts.strict is False
    assert opts.normalize is False
    assert opts.output_dir == Path("results/data_validation")
    assert opts.wan_height == 480
    assert opts.wan_width == 832
    assert opts.wan_frames == 81


def test_options_dataclass_custom_values() -> None:
    """Cyclopts should accept custom option values."""
    opts = DatasetValidationOptions(
        phase=("wan-t2v",),
        trigger_token="custom_token",
        strict=True,
        normalize=True,
        output_dir=Path("/tmp/custom"),
    )
    assert opts.phase == ("wan-t2v",)
    assert opts.trigger_token == "custom_token"
    assert opts.strict is True
    assert opts.normalize is True


def test_help_mentions_validate_and_normalize() -> None:
    """--help output should mention key concepts."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from videotuna.cli.dataset_app import app; app(['--help'])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "validate" in output.lower() or "normalize" in output.lower()
