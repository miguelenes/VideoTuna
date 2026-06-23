"""Tests for cyclopts inference CLI groups and Poetry entrypoints."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from videotuna.cli.inference_app import (
    PRESET_DOMAIN_T2I,
    PRESET_VALIDATE_T2V,
    PRESET_WAN2_2_T2V_720P,
    app,
)
from videotuna.cli.inference_options import (
    InferenceRunConfig,
    InferenceRunOptions,
    StandardInferenceOptions,
    inference_options_to_config,
    validate_preset_requirements,
)


def _help_text(command: list[str]) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (f"from videotuna.cli.inference_app import app; app({command!r})"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
        ["inference-wan2.2-t2v-720p", "--help"],
    ],
)
def test_inference_help_lists_shared_flags(command: list[str]) -> None:
    help_text = _help_text(command)
    for flag in (
        "--lorackpt",
        "--memory-preset",
        "--enable_vae_tiling",
        "--num-inference-steps",
    ):
        assert flag in help_text


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
    ],
)
def test_inference_help_omits_legacy_hunyuan_fp8(command: list[str]) -> None:
    help_text = _help_text(command).lower()
    for forbidden in ("enable_fp8", "enable-fp8", "dit-weight", "hunyuan"):
        assert forbidden not in help_text


def test_flag_parity_across_presets() -> None:
    run = InferenceRunOptions(lorackpt="/tmp/lora", num_inference_steps=8)
    standard = StandardInferenceOptions(memory_preset="balanced", device="cuda:0")
    t2i = inference_options_to_config(
        run=run,
        standard=standard,
        preset=PRESET_DOMAIN_T2I,
    ).model_dump()
    wan = inference_options_to_config(
        run=run,
        standard=standard,
        preset=PRESET_WAN2_2_T2V_720P,
    ).model_dump()

    shared_keys = {k for k in t2i if k not in {"config", "enable_model_cpu_offload"}}
    for key in shared_keys:
        assert t2i[key] == wan[key], key

    assert t2i["config"] == PRESET_DOMAIN_T2I.config
    assert wan["config"] == PRESET_WAN2_2_T2V_720P.config
    assert t2i["enable_model_cpu_offload"] is True
    assert wan["enable_model_cpu_offload"] is False


def test_domain_t2i_preset_applies_without_user_config() -> None:
    run_config = inference_options_to_config(preset=PRESET_DOMAIN_T2I)
    assert run_config.config == PRESET_DOMAIN_T2I.config
    assert run_config.enable_model_cpu_offload is True


def test_validate_domain_t2v_requires_checkpoint() -> None:
    with pytest.raises(SystemExit) as exc:
        validate_preset_requirements(InferenceRunOptions(), PRESET_VALIDATE_T2V)
    assert exc.value.code == 2


def test_cyclopts_parses_standard_flags() -> None:
    captured: dict[str, object] = {}

    def handler(
        run: InferenceRunOptions | None = None,
        *,
        standard: StandardInferenceOptions | None = None,
    ) -> None:
        captured["run_config"] = inference_options_to_config(run=run, standard=standard)

    probe = app.__class__(name="probe")
    probe.command(name="probe")(handler)
    probe(
        [
            "probe",
            "--lorackpt",
            "/tmp/lora",
            "--memory-preset",
            "low_vram",
            "--enable_vae_tiling",
            "--config",
            "configs/inference/presets/flux_domain_lora_smoke.yaml",
        ]
    )
    run_config = captured["run_config"]
    assert isinstance(run_config, InferenceRunConfig)
    assert run_config.lorackpt == "/tmp/lora"
    assert run_config.memory_preset == "low_vram"
    assert run_config.enable_vae_tiling is True


def test_inference_run_config_round_trip() -> None:
    run_config = inference_options_to_config(preset=PRESET_DOMAIN_T2I)
    round_tripped = InferenceRunConfig.model_validate(
        run_config.model_dump(mode="json")
    )
    assert round_tripped.config == run_config.config
    assert round_tripped.enable_model_cpu_offload == run_config.enable_model_cpu_offload


# ---------------------------------------------------------------------------
# End-to-end typed inference path — no argparse.Namespace bridge
# ---------------------------------------------------------------------------


_REMOVED_BRIDGE = [
    ("videotuna.cli.inference_options", "inference_options_to_namespace"),
    ("videotuna.utils.args_utils", "prepare_inference_args"),
    ("videotuna.utils.inference_cli", "prepare_cli_inference_args"),
    ("videotuna.utils.inference_cli", "apply_compile_env"),
    ("videotuna.utils.inference_cli", "apply_cpu_smoke_env"),
]


@pytest.mark.parametrize(("module_name", "attr"), _REMOVED_BRIDGE)
def test_removed_bridge_functions_are_gone(module_name: str, attr: str) -> None:
    """Guard against reintroducing the deprecated Namespace bridge."""
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, attr), f"{module_name}.{attr} should have been removed"


def test_inference_new_has_no_namespace() -> None:
    """scripts/inference_new.py must not import argparse or use Namespace."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "inference_new.py").read_text(
        encoding="utf-8"
    )
    assert "import argparse" not in source
    assert "argparse.Namespace" not in source


def test_run_inference_consumes_run_config() -> None:
    """run_inference's first parameter must be annotated InferenceRunConfig."""
    from scripts.inference_new import run_inference

    sig = inspect.signature(run_inference)
    first_param = next(iter(sig.parameters.values()))
    assert first_param.annotation is InferenceRunConfig


def test_prepare_inference_config_propagates_preset_side_effects() -> None:
    """Regression: preset side effects (low_vram → dtype=fp16, sequential offload)
    must propagate into the merged YAML inference_config, not be lost."""
    from unittest import mock

    from omegaconf import OmegaConf

    from videotuna.utils.args_utils import prepare_inference_config

    run_config = InferenceRunConfig(
        config="configs/inference/presets/flux_domain_lora_smoke.yaml",
        memory_preset="low_vram",
        savedir="/tmp/videotuna-test-preset",
    )
    yaml_config = OmegaConf.load(run_config.config)

    with (
        mock.patch(
            "videotuna.utils.device_utils.gpu_is_available", return_value=True
        ),
        mock.patch(
            "videotuna.utils.device_utils.detect_compute_backend", return_value="cuda"
        ),
        mock.patch(
            "videotuna.utils.device_utils.resolve_cpu_mode", return_value="off"
        ),
        mock.patch("videotuna.utils.args_utils.process_savedir", side_effect=lambda d: d),
    ):
        merged = prepare_inference_config(run_config, yaml_config)

    inference = merged.inference
    assert inference.dtype == "fp16"
    assert inference.enable_sequential_cpu_offload is True
    assert inference.enable_model_cpu_offload is False


def test_typed_end_to_end_merge() -> None:
    """inference_options_to_config → prepare_inference_config produces a DictConfig
    whose inference section reflects CLI overrides."""
    from unittest import mock

    from omegaconf import OmegaConf

    from videotuna.utils.args_utils import prepare_inference_config

    run_config = inference_options_to_config(
        run=InferenceRunOptions(
            lorackpt="/tmp/lora",
            num_inference_steps=4,
            seed=123,
        ),
        preset=PRESET_DOMAIN_T2I,
    )
    yaml_config = OmegaConf.load(run_config.config)

    with (
        mock.patch(
            "videotuna.utils.device_utils.gpu_is_available", return_value=True
        ),
        mock.patch(
            "videotuna.utils.device_utils.detect_compute_backend", return_value="cuda"
        ),
        mock.patch(
            "videotuna.utils.device_utils.resolve_cpu_mode", return_value="off"
        ),
        mock.patch("videotuna.utils.args_utils.process_savedir", side_effect=lambda d: d),
    ):
        merged = prepare_inference_config(run_config, yaml_config)

    inference = merged.inference
    assert inference.lorackpt == "/tmp/lora"
    assert int(inference.num_inference_steps) == 4
    assert int(inference.seed) == 123
