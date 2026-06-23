"""Tests for Rich CLI console helpers."""

from __future__ import annotations

from io import StringIO

from omegaconf import OmegaConf
from rich.console import Console

from videotuna.utils.cli_console import (
    install_pretty_tracebacks,
    render_inference_config_panel,
)


def test_render_inference_config_panel_includes_key_fields() -> None:
    inference_config = OmegaConf.create(
        {
            "mode": "t2v",
            "savedir": "/tmp/outputs/run",
            "height": 480,
            "width": 832,
            "frames": 81,
            "fps": 16,
            "seed": 42,
            "bs": 1,
            "n_samples_prompt": 2,
        }
    )
    buffer = StringIO()
    console = Console(file=buffer, width=80, force_terminal=True, legacy_windows=False)

    render_inference_config_panel(inference_config, console=console)

    output = buffer.getvalue()
    assert "Inference Configuration" in output
    assert "Mode" in output
    assert "t2v" in output
    assert "Save Directory" in output
    assert "/tmp/outputs/run" in output
    assert "Height" in output
    assert "480" in output
    assert "Seed" in output
    assert "42" in output


def test_render_inference_config_panel_skips_none_values() -> None:
    inference_config = OmegaConf.create(
        {
            "mode": "t2v",
            "savedir": "/tmp/outputs/run",
            "height": None,
            "width": 832,
            "frames": 81,
            "fps": None,
            "seed": 42,
            "bs": 1,
            "n_samples_prompt": 2,
        }
    )
    buffer = StringIO()
    console = Console(file=buffer, width=80, force_terminal=True, legacy_windows=False)

    render_inference_config_panel(inference_config, console=console)

    output = buffer.getvalue()
    assert "Height" not in output
    assert "FPS" not in output
    assert "Width" in output


def test_install_pretty_tracebacks_is_idempotent() -> None:
    install_pretty_tracebacks()
    install_pretty_tracebacks()
