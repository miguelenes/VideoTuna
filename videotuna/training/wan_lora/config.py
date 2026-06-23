"""Load and validate domain Wan LoRA YAML configs via Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from videotuna.utils.config_mapping import apply_config_mappings

WAN_VIDEO_FLOW_TARGET = "videotuna.flow.wanvideo.WanVideoModelFlow"
DATA_MODULE_TARGET = "videotuna.data.lightningdata.DataModuleFromConfig"
DATASET_FROM_CSV_TARGET = "videotuna.data.datasets.DatasetFromCSV"


class InstantiateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    use_from_pretrained: bool = False


class DatasetFromCSVParams(BaseModel):
    """Strict schema for DatasetFromCSV instantiation payload.

    Used by Wan I2V config validation to enforce that the dataset
    params match one of the two documented I2V CSV layouts from
    docs/runbooks/domain-adult-finetune.md (Phase 2.5).
    """

    model_config = ConfigDict(extra="forbid")

    csv_path: str
    height: int
    width: int
    num_frames: int
    frame_interval: int = 1
    image_to_video: bool = False
    train: bool = True
    i2v_mode: bool = False

    @model_validator(mode="after")
    def validate_i2v_csv_columns(self) -> Self:
        if not self.i2v_mode:
            return self

        import pandas as pd

        try:
            df = pd.read_csv(self.csv_path)
        except (FileNotFoundError, OSError):
            return self

        columns = set(df.columns)
        has_path = "path" in columns
        has_video = "video_path" in columns
        has_image = "image_path" in columns
        has_caption = "caption" in columns

        if not has_caption:
            return self

        pair_mode = has_video and has_image and not has_path
        first_frame = has_path and not has_video and not has_image

        doc = "docs/runbooks/domain-adult-finetune.md (Phase 2.5)"
        layouts = (
            "  - image_to_video: true  ->  path,caption\n"
            "  - image_to_video: false ->  image_path,video_path,caption"
        )

        if self.image_to_video:
            if not first_frame:
                raise ValueError(
                    f"CSV column mismatch: {self.csv_path!r} has columns "
                    f"{sorted(columns)} but image_to_video=true requires "
                    f"first-frame layout with path,caption columns. "
                    f"See {doc} for the two supported Wan I2V layouts:\n"
                    f"{layouts}"
                )
        else:
            if not pair_mode:
                raise ValueError(
                    f"CSV column mismatch: {self.csv_path!r} has columns "
                    f"{sorted(columns)} but image_to_video=false requires "
                    f"pair-mode layout with image_path,video_path,caption "
                    f"columns. See {doc} for the two supported Wan I2V "
                    f"layouts:\n{layouts}"
                )

        return self


class WanLoraFlowParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["t2v-14B", "i2v-14B"]
    ckpt_path: str
    offload_model: bool = True
    ulysses_size: int = 1
    ring_size: int = 1
    t5_fsdp: bool = False
    t5_cpu: bool = False
    dit_fsdp: bool = False
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: str | None = None
    prompt_extend_target_lang: str = "zh"
    seed: int = 42
    gradient_checkpointing: bool = True
    denoiser_config: InstantiateConfig
    first_stage_config: InstantiateConfig
    cond_stage_config: InstantiateConfig
    lora_config: InstantiateConfig

    @model_validator(mode="after")
    def validate_lora_params(self) -> Self:
        r = self.lora_config.params.get("r")
        alpha = self.lora_config.params.get("lora_alpha")
        if r is not None and int(r) <= 0:
            raise ValueError("lora_config.params.r must be > 0")
        if alpha is not None and float(alpha) <= 0:
            raise ValueError("lora_config.params.lora_alpha must be > 0")
        return self


class WanFlowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_type: Literal["wan"] | None = None
    target: str | None = None
    params: WanLoraFlowParams

    @model_validator(mode="after")
    def validate_flow_discriminator(self) -> Self:
        if self.flow_type is None and self.target is None:
            raise ValueError("flow must have either 'flow_type' or 'target'")
        if self.target is not None and self.target != WAN_VIDEO_FLOW_TARGET:
            raise ValueError(
                f"flow.target must be {WAN_VIDEO_FLOW_TARGET!r}, "
                f"got {self.target!r}"
            )
        return self


class WanTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    accelerator: str = "gpu"
    benchmark: bool = True
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    max_epochs: int
    precision: str = "bf16-mixed"
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: str | None = None


class WanLightningCallbacks(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_logger: InstantiateConfig
    model_checkpoint: InstantiateConfig


class WanLightningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    trainer: WanTrainerConfig
    callbacks: WanLightningCallbacks


class WanTrainSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    ckpt: str
    logdir: str
    seed: int
    debug: bool = False
    first_stage_key: str
    cond_stage_key: str
    mapping: dict[str, str] | None = None
    lr_config: dict[str, Any]
    data: InstantiateConfig
    lightning: WanLightningConfig

    @model_validator(mode="after")
    def validate_data_targets(self) -> Self:
        if self.data.target != DATA_MODULE_TARGET:
            raise ValueError(
                f"train.data.target must be {DATA_MODULE_TARGET!r}, "
                f"got {self.data.target!r}"
            )
        train_dataset = self.data.params.get("train")
        if not isinstance(train_dataset, dict):
            raise ValueError("train.data.params.train must be present")
        if train_dataset.get("target") != DATASET_FROM_CSV_TARGET:
            raise ValueError(
                f"train.data.params.train.target must be {DATASET_FROM_CSV_TARGET!r}, "
                f"got {train_dataset.get('target')!r}"
            )
        return self


class WanInferenceSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["t2v", "i2v"]
    ckpt_path: str
    savedir: str
    seed: int
    height: int
    width: int
    image: str | None = None
    prompt_file: str | None = None
    prompt_dir: str | None = None
    solver: str = "unipc"
    num_inference_steps: int = 20
    time_shift: float = 3.0
    unconditional_guidance_scale: float = 5.0
    frames: int = 81
    n_samples_prompt: int = 1
    bs: int = 1
    savefps: int = 30
    enable_model_cpu_offload: bool = True
    mapping: dict[str, str] | None = None


class WanLoraTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow: WanFlowConfig
    train: WanTrainSection
    inference: WanInferenceSection

    @model_validator(mode="after")
    def validate_i2v_consistency(self) -> Self:
        if self.flow.params.task != "i2v-14B":
            return self
        if self.inference.mode != "i2v":
            raise ValueError(
                "inference.mode must be 'i2v' when flow.params.task is i2v-14B"
            )
        denoiser_params = self.flow.params.denoiser_config.params
        if denoiser_params.get("model_type") != "i2v":
            raise ValueError(
                "flow.params.denoiser_config.params.model_type must be "
                "'i2v' for i2v-14B"
            )
        if denoiser_params.get("subfolder") != "high_noise_model":
            raise ValueError(
                "flow.params.denoiser_config.params.subfolder must be "
                "'high_noise_model' for i2v-14B"
            )
        return self

    @model_validator(mode="after")
    def validate_i2v_dataset_layout(self) -> Self:
        if self.flow.params.task != "i2v-14B":
            return self

        data_params = self.train.data.params
        train_entry = data_params.get("train", {})
        ds_params = train_entry.get("params", {})
        target = train_entry.get("target")

        if target != DATASET_FROM_CSV_TARGET:
            raise ValueError(
                f"train.data.params.train.target must be "
                f"{DATASET_FROM_CSV_TARGET!r} for i2v-14B, "
                f"got {target!r}"
            )

        try:
            parsed = DatasetFromCSVParams.model_validate(ds_params)
        except Exception as exc:
            raise ValueError(
                f"Invalid train.data.params.train.params for i2v-14B: {exc}"
            ) from exc

        if not parsed.i2v_mode:
            raise ValueError(
                "For i2v-14B task, train.data.params.train.params.i2v_mode "
                "must be true so that DatasetFromCSV enforces the correct "
                "Wan I2V CSV layout. See the two documented layouts in "
                "docs/runbooks/domain-adult-finetune.md (Phase 2.5):\n"
                "  - image_to_video: false  ->  image_path,video_path,caption\n"
                "  - image_to_video: true   ->  path,caption\n"
                "Set i2v_mode: true in configs/domain/wan_i2v_lora.yaml."
            )

        return self


def _register_dtype_resolver() -> None:
    if OmegaConf.has_resolver("dtype_resolver"):
        return

    def resolve_dtype(dtype_str: str):
        mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_str)

    OmegaConf.register_new_resolver("dtype_resolver", resolve_dtype)


def _merge_and_resolve(
    config_paths: Sequence[str | Path],
    cli_overrides: list[str] | None = None,
    *,
    apply_inference_mapping: bool = True,
) -> dict[str, Any]:
    configs = [OmegaConf.load(str(path)) for path in config_paths]
    cli = OmegaConf.from_dotlist(cli_overrides or [])
    if configs:
        merged = OmegaConf.merge(*configs, cli)
    else:
        merged = cli

    if not isinstance(merged, DictConfig):
        raise TypeError(f"Expected YAML mapping config, got {type(merged).__name__}")

    apply_config_mappings(merged, section="train")
    if apply_inference_mapping:
        apply_config_mappings(merged, section="inference")

    _register_dtype_resolver()
    OmegaConf.resolve(merged)
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Wan LoRA config must resolve to a mapping")
    return cast(dict[str, Any], resolved)


def load_wan_lora_config(
    config_path: str | Path,
    *,
    extra_configs: Sequence[str | Path] = (),
    cli_overrides: list[str] | None = None,
) -> WanLoraTrainConfig:
    """Load domain Wan YAML through merge, mapping, resolve, and Pydantic validation."""
    paths = [Path(config_path), *[Path(path) for path in extra_configs]]
    resolved = _merge_and_resolve(paths, cli_overrides)
    return WanLoraTrainConfig.model_validate(resolved)


def validated_config_to_dictconfig(config: WanLoraTrainConfig) -> DictConfig:
    """Convert a validated model back to OmegaConf for existing runtime code."""
    return OmegaConf.create(config.model_dump(mode="json"))
