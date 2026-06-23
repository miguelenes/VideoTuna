"""Unit tests for Flux domain LoRA Pydantic config loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from videotuna.training.flux_lora.config import FluxLoraTrainConfig, load_train_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


def test_invalid_key_raises_validation_error():
    train_cfg, _data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    payload = train_cfg.model_dump(mode="json")
    payload["unknown_flux_key"] = "nope"
    with pytest.raises(ValidationError):
        FluxLoraTrainConfig.model_validate(payload)


def test_round_trip_revalidates():
    train_cfg, _data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    round_tripped = FluxLoraTrainConfig.model_validate(
        train_cfg.model_dump(mode="json")
    )
    assert (
        round_tripped.pretrained_model_name_or_path
        == train_cfg.pretrained_model_name_or_path
    )
    assert round_tripped.lora_rank == train_cfg.lora_rank
    assert round_tripped.max_train_steps == train_cfg.max_train_steps


def test_invalid_model_family_raises():
    train_cfg, _data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    payload = train_cfg.model_dump(mode="json")
    payload["model_family"] = "sdxl"
    with pytest.raises(ValidationError):
        FluxLoraTrainConfig.model_validate(payload)
