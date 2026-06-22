"""Load and normalize `configs/domain/` SimpleTuner-style JSON configs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videotuna.utils.logging_config import bound_logger

logger = bound_logger(phase="t2i", flow="flux_lora")

_UNSUPPORTED_KEYS = frozenset(
    {
        "disable_benchmark",
        "resume_from_checkpoint",
        "checkpoints_total_limit",
        "caption_dropout_probability",
        "disable_tf32",
        "validation_guidance_rescale",
        "validation_num_inference_steps",
        "aspect_bucket_rounding",
        "minimum_image_size",
        "write_batch_size",
        "lora_type",
        "gradient_checkpointing",
    }
)


def _normalize_key(key: str) -> str:
    return key[2:] if key.startswith("--") else key


def _coerce_value(key: str, value: Any) -> Any:
    if key in {"gradient_checkpointing", "disable_benchmark", "disable_tf32"}:
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)
    if key in {
        "lora_rank",
        "max_train_steps",
        "checkpointing_steps",
        "train_batch_size",
        "resolution",
        "validation_steps",
        "lr_warmup_steps",
        "num_train_epochs",
        "seed",
        "validation_seed",
    }:
        return int(value)
    if key in {"learning_rate"}:
        return float(value)
    if key in {"validation_guidance"}:
        return float(value)
    return value


@dataclass
class FluxLoraDataConfig:
    instance_data_dir: str
    caption_strategy: str = "filename"
    default_caption: str | None = None
    resolution: int = 512
    crop: bool = True
    crop_aspect: str = "square"


@dataclass
class FluxLoraTrainConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    instance_data_dir: str
    model_family: str = "flux"
    model_type: str = "lora"
    lora_rank: int = 4
    learning_rate: float = 8e-5
    lr_scheduler: str = "polynomial"
    lr_warmup_steps: int = 5
    max_train_steps: int = 1000
    train_batch_size: int = 1
    resolution: int = 512
    checkpointing_steps: int = 500
    mixed_precision: str = "bf16"
    optimizer: str = "adamw"
    seed: int = 42
    validation_prompt: str | None = None
    validation_steps: int | None = None
    gradient_checkpointing: bool = True
    data_backend_config: str | None = None
    ignored_keys: list[str] = field(default_factory=list)


def _parse_local_backend(backends: list[dict[str, Any]]) -> FluxLoraDataConfig:
    image_backend = next(
        (
            b
            for b in backends
            if b.get("type") == "local"
            and b.get("dataset_type") != "text_embeds"
            and not b.get("disabled", False)
        ),
        None,
    )
    if image_backend is None:
        raise ValueError(
            "multidatabackend.json must include a local image backend "
            "(text_embeds-only backends are not supported)."
        )
    if any(
        b.get("type") == "local" and b.get("dataset_type") == "text_embeds"
        for b in backends
    ):
        logger.info(
            "text_embeds cache backend is ignored; prompts are encoded on-the-fly."
        )
    return FluxLoraDataConfig(
        instance_data_dir=image_backend["instance_data_dir"],
        caption_strategy=image_backend.get("caption_strategy", "filename"),
        default_caption=image_backend.get("caption"),
        resolution=int(image_backend.get("resolution", 512)),
        crop=bool(image_backend.get("crop", True)),
        crop_aspect=image_backend.get("crop_aspect", "square"),
    )


def load_train_config(
    config_path: str | Path,
    data_config_path: str | Path,
) -> tuple[FluxLoraTrainConfig, FluxLoraDataConfig]:
    with open(config_path) as f:
        raw = json.load(f)
    with open(data_config_path) as f:
        backends = json.load(f)

    normalized: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in raw.items():
        norm_key = _normalize_key(key)
        if norm_key in _UNSUPPORTED_KEYS:
            ignored.append(norm_key)
            continue
        normalized[norm_key] = _coerce_value(norm_key, value)

    if ignored:
        logger.info("Ignoring unsupported SimpleTuner config keys: {}", sorted(ignored))

    data_cfg = _parse_local_backend(backends)
    instance_data_dir = (
        normalized.get("instance_data_dir") or data_cfg.instance_data_dir
    )
    resolution = int(normalized.get("resolution", data_cfg.resolution))

    train_cfg = FluxLoraTrainConfig(
        pretrained_model_name_or_path=normalized["pretrained_model_name_or_path"],
        output_dir=normalized["output_dir"],
        instance_data_dir=instance_data_dir,
        model_family=normalized.get("model_family", "flux"),
        model_type=normalized.get("model_type", "lora"),
        lora_rank=int(normalized.get("lora_rank", 4)),
        learning_rate=float(normalized.get("learning_rate", 8e-5)),
        lr_scheduler=normalized.get("lr_scheduler", "polynomial"),
        lr_warmup_steps=int(normalized.get("lr_warmup_steps", 5)),
        max_train_steps=int(normalized.get("max_train_steps", 1000)),
        train_batch_size=int(normalized.get("train_batch_size", 1)),
        resolution=resolution,
        checkpointing_steps=int(normalized.get("checkpointing_steps", 500)),
        mixed_precision=normalized.get("mixed_precision", "bf16"),
        optimizer=normalized.get("optimizer", "adamw"),
        seed=int(normalized.get("seed", 42)),
        validation_prompt=normalized.get("validation_prompt"),
        validation_steps=normalized.get("validation_steps"),
        gradient_checkpointing=bool(normalized.get("gradient_checkpointing", True)),
        data_backend_config=normalized.get("data_backend_config"),
        ignored_keys=ignored,
    )
    data_cfg.resolution = resolution
    return train_cfg, data_cfg


def stamp_output_dir(output_dir: str) -> str:
    from datetime import datetime

    path = Path(output_dir)
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = path.stem
    name_list = folder_name.split("_")
    if len(name_list[-1]) == 14 and name_list[-1].isdigit():
        folder_name = "_".join(name_list[:-1])
    stamped = path.parent / f"{folder_name}_{time_str}"
    return str(stamped)
