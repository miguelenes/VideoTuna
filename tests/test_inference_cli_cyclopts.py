"""Tests for typed inference option groups (Pydantic) and config merge."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from videotuna.cli.inference_options import (
    InferencePreset,
    InferenceRunConfig,
    InferenceRunOptions,
    StandardInferenceOptions,
    inference_options_to_config,
)


def test_inference_options_to_config_merges_correctly():
    run = InferenceRunOptions(
        config="test.yaml",
        mode="t2v",
        savedir="/tmp/out",
        num_inference_steps=20,
    )
    standard = StandardInferenceOptions(
        cpu_smoke=True,
        device="cpu",
    )
    config = inference_options_to_config(run=run, standard=standard)
    assert isinstance(config, InferenceRunConfig)
    assert config.mode == "t2v"
    assert config.savedir == "/tmp/out"
    assert config.num_inference_steps == 20
    assert config.cpu_smoke is True
    assert config.device == "cpu"
    assert config.config == "test.yaml"


def test_inference_options_to_config_requires_config():
    run = InferenceRunOptions()
    standard = StandardInferenceOptions()
    with pytest.raises(ValueError, match="Inference requires a YAML config path"):
        inference_options_to_config(run=run, standard=standard)


def test_preset_hardcodes_config_path():
    preset = InferencePreset(
        cli_name="test-preset",
        config="configs/inference/presets/test.yaml",
    )
    config = inference_options_to_config(preset=preset)
    assert config.config == "configs/inference/presets/test.yaml"


def test_preset_enable_cpu_offload():
    preset = InferencePreset(
        cli_name="offload-preset",
        config="config.yaml",
        enable_model_cpu_offload=True,
    )
    config = inference_options_to_config(preset=preset)
    assert config.enable_model_cpu_offload is True


def test_run_options_extra_forbid():
    with pytest.raises(ValidationError, match="extra"):
        InferenceRunOptions.model_validate(
            {
                "mode": "t2v",
                "unknown_field": "should_fail",
            }
        )


def test_standard_options_extra_forbid():
    with pytest.raises(ValidationError, match="extra"):
        StandardInferenceOptions.model_validate(
            {
                "cpu_smoke": True,
                "bogus_param": 42,
            }
        )


def test_run_config_extra_forbid():
    with pytest.raises(ValidationError, match="extra"):
        InferenceRunConfig.model_validate(
            {
                "config": "test.yaml",
                "invalid_key": "rejected",
            }
        )


def test_preset_keeps_dataclass_behavior():
    preset = InferencePreset(
        cli_name="test",
        config="cfg.yaml",
    )
    assert preset.cli_name == "test"
    assert preset.config == "cfg.yaml"
    assert preset.enable_model_cpu_offload is False


def test_bool_defaults_applied():
    run = InferenceRunOptions(config="cfg.yaml")
    standard = StandardInferenceOptions()
    config = inference_options_to_config(run=run, standard=standard)
    assert config.enable_vae_tiling is False
    assert config.enable_vae_slicing is False
    assert config.compile is False
    assert config.cpu_smoke is False
