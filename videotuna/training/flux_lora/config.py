"""Load and normalize `configs/domain/` SimpleTuner-style JSON configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from videotuna.utils.logging_config import bound_logger

logger = bound_logger(phase="t2i", flow="flux_lora")

OptimizerChoice = Literal["adamw", "adamw_bf16"]


def _normalize_key(key: str) -> str:
    return key[2:] if key.startswith("--") else key


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


class FluxTextEmbedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: str
    write_batch_size: int | None = None
    disabled: bool = False


class FluxLoraDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_data_dir: str
    caption_strategy: str = "filename"
    default_caption: str | None = None
    resolution: int = 512
    crop: bool = True
    crop_aspect: str = "square"
    resolution_type: str = "pixel_area"
    aspect_bucket_rounding: int = 2
    minimum_image_size: int = 0
    maximum_image_size: int | None = None
    caption_dropout_probability: float = 0.0
    text_embeds: FluxTextEmbedConfig | None = None


class FluxLoraTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretrained_model_name_or_path: str
    output_dir: str
    instance_data_dir: str
    model_family: str = "flux"
    model_type: str = "lora"
    lora_type: str = "standard"
    lora_rank: int = 4
    learning_rate: float = 8e-5
    lr_scheduler: str = "polynomial"
    lr_warmup_steps: int = 5
    max_train_steps: int = 1000
    num_train_epochs: int = -1
    train_batch_size: int = 1
    num_workers: int = 0
    resolution: int = 512
    resolution_type: str = "pixel_area"
    aspect_bucket_rounding: int = 2
    minimum_image_size: int = 0
    checkpointing_steps: int = 500
    checkpoints_total_limit: int | None = None
    resume_from_checkpoint: str | None = None
    mixed_precision: str = "bf16"
    optimizer: OptimizerChoice = "adamw"
    seed: int = 42
    disable_tf32: bool = False
    disable_benchmark: bool = False
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    caption_dropout_probability: float = 0.0
    write_batch_size: int = 128
    validation_prompt: str | None = None
    validation_steps: int | None = None
    validation_resolution: str = "512x512"
    validation_guidance: float = 3.0
    validation_guidance_rescale: float = 0.0
    validation_num_inference_steps: int = 10
    validation_seed: int = 42
    data_backend_config: str | None = None

    @field_validator(
        "disable_tf32",
        "disable_benchmark",
        "gradient_checkpointing",
        mode="before",
    )
    @classmethod
    def _coerce_bool_fields(cls, value: object) -> bool:
        return _coerce_bool(value)

    @model_validator(mode="after")
    def validate_flux_constraints(self) -> Self:
        if self.model_family != "flux":
            raise ValueError(f"model_family must be 'flux', got {self.model_family!r}")
        if self.model_type != "lora":
            raise ValueError(f"model_type must be 'lora', got {self.model_type!r}")
        if self.lora_type != "standard":
            raise ValueError(f"lora_type must be 'standard', got {self.lora_type!r}")
        if self.num_train_epochs != -1:
            raise ValueError(
                "num_train_epochs must be -1 "
                "(PrivTune Flux trainer is step-based via max_train_steps)"
            )
        if self.resolution_type != "pixel_area":
            raise ValueError(
                f"resolution_type must be 'pixel_area', got {self.resolution_type!r}"
            )
        if self.validation_guidance_rescale != 0.0:
            raise ValueError(
                "validation_guidance_rescale is not supported for Flux (must be 0.0)"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.resume_from_checkpoint is not None:
            if (
                not isinstance(self.resume_from_checkpoint, str)
                or not self.resume_from_checkpoint.strip()
            ):
                raise ValueError(
                    "resume_from_checkpoint must be null, 'latest', or a non-empty path"
                )
        return self


def _parse_text_embeds_backend(
    backends: list[dict[str, Any]],
) -> FluxTextEmbedConfig | None:
    embed_backend = next(
        (
            b
            for b in backends
            if b.get("type") == "local" and b.get("dataset_type") == "text_embeds"
        ),
        None,
    )
    if embed_backend is None:
        return None
    if embed_backend.get("disabled", False):
        return FluxTextEmbedConfig(cache_dir="", disabled=True)
    cache_dir = embed_backend.get("cache_dir")
    if not cache_dir:
        raise ValueError("text_embeds backend requires cache_dir")
    write_batch_size = embed_backend.get("write_batch_size")
    parsed_write_batch_size = (
        int(write_batch_size) if write_batch_size is not None else None
    )
    return FluxTextEmbedConfig(
        cache_dir=str(cache_dir),
        write_batch_size=parsed_write_batch_size,
        disabled=False,
    )


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
    maximum_image_size = image_backend.get("maximum_image_size")
    return FluxLoraDataConfig(
        instance_data_dir=image_backend["instance_data_dir"],
        caption_strategy=image_backend.get("caption_strategy", "filename"),
        default_caption=image_backend.get("caption"),
        resolution=int(image_backend.get("resolution", 512)),
        crop=bool(image_backend.get("crop", True)),
        crop_aspect=image_backend.get("crop_aspect", "square"),
        resolution_type=str(image_backend.get("resolution_type", "pixel_area")),
        aspect_bucket_rounding=int(image_backend.get("aspect_bucket_rounding", 2)),
        minimum_image_size=int(image_backend.get("minimum_image_size", 0)),
        maximum_image_size=int(maximum_image_size) if maximum_image_size else None,
        text_embeds=_parse_text_embeds_backend(backends),
    )


def _normalize_train_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize SimpleTuner-style keys and coerce JSON scalar types."""
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        norm_key = _normalize_key(key)
        if norm_key in {
            "gradient_checkpointing",
            "disable_benchmark",
            "disable_tf32",
        }:
            normalized[norm_key] = _coerce_bool(value)
        elif norm_key in {
            "lora_rank",
            "max_train_steps",
            "checkpointing_steps",
            "checkpoints_total_limit",
            "train_batch_size",
            "write_batch_size",
            "resolution",
            "validation_steps",
            "validation_num_inference_steps",
            "lr_warmup_steps",
            "num_train_epochs",
            "seed",
            "validation_seed",
            "num_workers",
            "aspect_bucket_rounding",
            "minimum_image_size",
            "gradient_accumulation_steps",
        }:
            normalized[norm_key] = int(value)
        elif norm_key in {
            "learning_rate",
            "validation_guidance",
            "validation_guidance_rescale",
            "caption_dropout_probability",
        }:
            normalized[norm_key] = float(value)
        else:
            normalized[norm_key] = value
    return normalized


def _merge_data_config(
    data_cfg: FluxLoraDataConfig,
    train_cfg: FluxLoraTrainConfig,
) -> FluxLoraDataConfig:
    """Apply train-config overrides onto data config for dataset construction."""
    write_batch_size = train_cfg.write_batch_size
    if data_cfg.text_embeds and data_cfg.text_embeds.write_batch_size is not None:
        write_batch_size = data_cfg.text_embeds.write_batch_size
    text_embeds = data_cfg.text_embeds
    if text_embeds is not None and not text_embeds.disabled:
        text_embeds = FluxTextEmbedConfig(
            cache_dir=text_embeds.cache_dir,
            write_batch_size=write_batch_size,
            disabled=False,
        )
    return FluxLoraDataConfig(
        instance_data_dir=train_cfg.instance_data_dir,
        caption_strategy=data_cfg.caption_strategy,
        default_caption=data_cfg.default_caption,
        resolution=train_cfg.resolution,
        crop=data_cfg.crop,
        crop_aspect=data_cfg.crop_aspect,
        resolution_type=train_cfg.resolution_type,
        aspect_bucket_rounding=train_cfg.aspect_bucket_rounding,
        minimum_image_size=train_cfg.minimum_image_size,
        maximum_image_size=data_cfg.maximum_image_size,
        caption_dropout_probability=train_cfg.caption_dropout_probability,
        text_embeds=text_embeds,
    )


def load_train_config(
    config_path: str | Path,
    data_config_path: str | Path,
) -> tuple[FluxLoraTrainConfig, FluxLoraDataConfig]:
    with open(config_path) as f:
        raw = json.load(f)
    with open(data_config_path) as f:
        backends = json.load(f)

    normalized = _normalize_train_payload(raw)
    data_cfg = _parse_local_backend(backends)

    instance_data_dir = (
        normalized.get("instance_data_dir") or data_cfg.instance_data_dir
    )
    normalized["instance_data_dir"] = instance_data_dir
    normalized.setdefault("resolution", data_cfg.resolution)
    normalized.setdefault("minimum_image_size", data_cfg.minimum_image_size)
    normalized.setdefault("aspect_bucket_rounding", data_cfg.aspect_bucket_rounding)
    normalized.setdefault("resolution_type", data_cfg.resolution_type)

    train_cfg = FluxLoraTrainConfig.model_validate(normalized)
    merged_data_cfg = _merge_data_config(data_cfg, train_cfg)
    return train_cfg, merged_data_cfg


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
