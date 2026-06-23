"""Tests for cyclopts training CLI groups and Poetry entrypoints."""

from __future__ import annotations

import subprocess
import sys

import pytest

from videotuna.cli.train_app import (
    PRESET_TRAIN_T2I,
    PRESET_TRAIN_T2V,
    app,
)
from videotuna.cli.train_options import (
    FLUX_T2I_CONFIG,
    FLUX_T2I_DATA_CONFIG,
    WAN_I2V_LORA_CONFIG,
    WAN_T2V_LORA_CONFIG,
    FluxTrainOptions,
    WanTrainOptions,
    build_flux_train_argv,
    build_wan_train_argv,
)


def _help_text(command: list[str]) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (f"from videotuna.cli.train_app import app; app({command!r})"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    ("command", "expected_paths"),
    [
        (
            ["train-domain-t2i", "--help"],
            (FLUX_T2I_CONFIG, FLUX_T2I_DATA_CONFIG),
        ),
        (
            ["train-domain-t2v", "--help"],
            (WAN_T2V_LORA_CONFIG, "checkpoints/wan/Wan2.1-T2V-14B"),
        ),
        (
            ["train-domain-i2v", "--help"],
            (WAN_I2V_LORA_CONFIG, "checkpoints/wan/Wan2.1-I2V-14B-480P"),
        ),
    ],
)
def test_train_help_lists_config_paths(
    command: list[str],
    expected_paths: tuple[str, ...],
) -> None:
    help_text = _help_text(command)
    for path in expected_paths:
        assert path in help_text


def test_flux_preset_default_configs() -> None:
    argv = build_flux_train_argv(PRESET_TRAIN_T2I)
    assert argv[argv.index("--config_path") + 1] == FLUX_T2I_CONFIG
    assert argv[argv.index("--data_config_path") + 1] == FLUX_T2I_DATA_CONFIG
    assert argv[0] == "accelerate"
    assert "scripts/train_flux_lora.py" in argv


def test_wan_preset_default_config_and_ckpt() -> None:
    argv = build_wan_train_argv(
        PRESET_TRAIN_T2V,
        timestamp="20260101120000",
    )
    assert argv[argv.index("--base") + 1] == WAN_T2V_LORA_CONFIG
    assert argv[argv.index("--ckpt") + 1] == "checkpoints/wan/Wan2.1-T2V-14B"
    assert argv[argv.index("--name") + 1] == "train_wan_domain_t2v_lora_20260101120000"
    assert argv[argv.index("--devices") + 1] == "0,"
    assert "scripts/train_new.py" in argv


def test_config_override_flux() -> None:
    custom = "configs/domain/flux_t2i_cloud_smoke.json"
    argv = build_flux_train_argv(
        PRESET_TRAIN_T2I,
        FluxTrainOptions(config=custom),
    )
    assert argv[argv.index("--config_path") + 1] == custom
    assert argv[argv.index("--data_config_path") + 1] == FLUX_T2I_DATA_CONFIG


def test_config_override_wan() -> None:
    custom = "configs/domain/wan_t2v_lora_cloud_smoke.yaml"
    argv = build_wan_train_argv(
        PRESET_TRAIN_T2V,
        WanTrainOptions(config=custom),
        timestamp="fixed",
    )
    assert argv[argv.index("--base") + 1] == custom


def test_wan_extra_cli_passthrough() -> None:
    argv = build_wan_train_argv(
        PRESET_TRAIN_T2V,
        timestamp="fixed",
        limit_train_batches="1",
    )
    assert argv[argv.index("--limit_train_batches") + 1] == "1"


def test_cyclopts_parses_train_flags() -> None:
    captured: dict[str, object] = {}

    def handler(
        wan: WanTrainOptions | None = None,
        **extra_cli: str,
    ) -> None:
        captured["wan"] = wan
        captured["extra"] = extra_cli

    probe = app.__class__(name="probe")
    probe.command(name="probe")(handler)
    probe(
        [
            "probe",
            "--config",
            "configs/domain/wan_t2v_lora_cloud_smoke.yaml",
            "--devices",
            "0,1,",
            "--limit_train_batches",
            "2",
        ]
    )
    wan = captured["wan"]
    assert isinstance(wan, WanTrainOptions)
    assert wan.config == "configs/domain/wan_t2v_lora_cloud_smoke.yaml"
    assert wan.devices == "0,1,"
    extra = captured["extra"]
    assert isinstance(extra, dict)
    assert extra.get("limit_train_batches") == "2"
