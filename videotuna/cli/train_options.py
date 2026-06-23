"""Typed cyclopts option groups for training entrypoints."""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Annotated, Any

from cyclopts import Parameter

FLUX_T2I_CONFIG = "configs/domain/flux_t2i.json"
FLUX_T2I_DATA_CONFIG = "configs/domain/flux_t2i_data.json"
FLUX_T2I_CLOUD_SMOKE = "configs/domain/flux_t2i_cloud_smoke.json"
WAN_T2V_LORA_CONFIG = "configs/domain/wan_t2v_lora.yaml"
WAN_T2V_LORA_CLOUD_SMOKE = "configs/domain/wan_t2v_lora_cloud_smoke.yaml"
WAN_I2V_LORA_CONFIG = "configs/domain/wan_i2v_lora.yaml"


@Parameter(name="*")
@dataclass
class FluxTrainOptions:
    """Flux T2I LoRA training flags."""

    config: Annotated[
        str | None,
        Parameter(
            name="config",
            alias=["--config-path"],
            help=f"Training config JSON (default: {FLUX_T2I_CONFIG}).",
        ),
    ] = None
    data_config_path: Annotated[
        str | None,
        Parameter(
            name="data-config-path",
            help=f"Multidatabackend JSON (default: {FLUX_T2I_DATA_CONFIG}).",
        ),
    ] = None


@Parameter(name="*")
@dataclass
class WanTrainOptions:
    """Wan 2.1 LoRA training flags."""

    config: Annotated[
        str | None,
        Parameter(
            name="config",
            alias=["--base"],
            help="Primary YAML config (domain preset supplies the default path).",
        ),
    ] = None
    ckpt: Annotated[
        str | None,
        Parameter(name="ckpt", help="Pretrained base model checkpoint path."),
    ] = None
    logdir: Annotated[
        str | None,
        Parameter(name="logdir", help="Training log root directory."),
    ] = None
    name: Annotated[
        str | None,
        Parameter(
            name="name",
            help="Experiment name (timestamp appended when omitted).",
        ),
    ] = None
    devices: Annotated[
        str | None,
        Parameter(name="devices", help="Lightning trainer devices (e.g. 0,)."),
    ] = None


@dataclass(frozen=True)
class FluxTrainPreset:
    """Baked-in defaults for a Flux training Poetry entrypoint."""

    cli_name: str
    config_path: str
    data_config_path: str


@dataclass(frozen=True)
class WanTrainPreset:
    """Baked-in defaults for a Wan training Poetry entrypoint."""

    cli_name: str
    config: str
    ckpt: str
    expname: str
    logdir: str = "results/train"
    devices: str = "0,"


def _non_null_values(options: Any) -> dict[str, Any]:
    return {field.name: getattr(options, field.name) for field in fields(options)}


def flatten_extra_cli_to_argv(extra_cli: dict[str, Any]) -> list[str]:
    """Convert cyclopts **kwargs captures back to subprocess argv tokens."""
    argv: list[str] = []
    for key, value in extra_cli.items():
        argv.extend([f"--{key}", str(value)])
    return argv


def build_flux_train_argv(
    preset: FluxTrainPreset,
    options: FluxTrainOptions | None = None,
    **extra_cli: str,
) -> list[str]:
    """Build argv for ``accelerate launch scripts/train_flux_lora.py``."""
    opts = options or FluxTrainOptions()
    config_path = opts.config or preset.config_path
    data_config_path = opts.data_config_path or preset.data_config_path
    argv = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines=1",
        "scripts/train_flux_lora.py",
        "--config_path",
        config_path,
        "--data_config_path",
        data_config_path,
    ]
    argv.extend(flatten_extra_cli_to_argv(extra_cli))
    return argv


def build_wan_train_argv(
    preset: WanTrainPreset,
    options: WanTrainOptions | None = None,
    *,
    timestamp: str | None = None,
    **extra_cli: str,
) -> list[str]:
    """Build argv for ``python scripts/train_new.py``."""
    opts = options or WanTrainOptions()
    config = opts.config or preset.config
    ckpt = opts.ckpt or preset.ckpt
    logdir = opts.logdir or preset.logdir
    devices = opts.devices if opts.devices is not None else preset.devices
    if opts.name:
        name = opts.name
    else:
        stamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{preset.expname}_{stamp}"

    argv = [
        sys.executable,
        "scripts/train_new.py",
        "--ckpt",
        ckpt,
        "--base",
        config,
        "--logdir",
        logdir,
        "--name",
        name,
        "--devices",
        devices,
    ]
    argv.extend(flatten_extra_cli_to_argv(extra_cli))
    return argv
