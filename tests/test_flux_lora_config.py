"""Unit tests for Flux domain LoRA Pydantic config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from videotuna.training.flux_lora.config import (
    FluxLoraTrainConfig,
    load_train_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


def test_invalid_model_family_raises():
    with pytest.raises(ValidationError, match="model_family"):
        FluxLoraTrainConfig(
            pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
            output_dir="results/train/test",
            instance_data_dir="data/t2i/domain",
            model_family="sd",
        )


def test_invalid_extra_key_raises():
    with pytest.raises(ValidationError, match="extra"):
        FluxLoraTrainConfig.model_validate(
            {
                "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "output_dir": "results/train/test",
                "instance_data_dir": "data/t2i/domain",
                "unknown_key": "should_fail",
            }
        )


def test_round_trip_revalidates():
    train_cfg, _ = load_train_config(FLUX_CONFIG, FLUX_DATA)
    round_tripped = FluxLoraTrainConfig.model_validate(
        train_cfg.model_dump(mode="json")
    )
    assert round_tripped.max_train_steps == train_cfg.max_train_steps
    assert round_tripped.pretrained_model_name_or_path == (
        "black-forest-labs/FLUX.1-dev"
    )


def test_coerce_string_to_int():
    cfg = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir="results/train/test",
        instance_data_dir="data/t2i/domain",
        lora_rank="8",
    )
    assert cfg.lora_rank == 8
    assert isinstance(cfg.lora_rank, int)


def test_coerce_string_to_float():
    cfg = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir="results/train/test",
        instance_data_dir="data/t2i/domain",
        learning_rate="1e-4",
    )
    assert cfg.learning_rate == 0.0001
    assert isinstance(cfg.learning_rate, float)


def test_coerce_int_field_from_json(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--lora_rank":"16"}',
        encoding="utf-8",
    )
    train_cfg, _ = load_train_config(config_path, data_path)
    assert train_cfg.lora_rank == 16
    assert isinstance(train_cfg.lora_rank, int)


def test_coerce_float_field_from_json(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--learning_rate":"5e-5"}',
        encoding="utf-8",
    )
    train_cfg, _ = load_train_config(config_path, data_path)
    assert abs(train_cfg.learning_rate - 5e-5) < 1e-10


def test_default_field_values():
    """Basic smoke: instantiate config with just required fields."""
    cfg = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir="results/train/test",
        instance_data_dir="data/t2i/domain",
    )
    assert cfg.model_family == "flux"
    assert cfg.lora_rank == 4
    assert cfg.seed == 42
