"""cyclopts App for PrivTune training entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable

from cyclopts import App

from videotuna.cli.train_options import (
    FLUX_T2I_CONFIG,
    FLUX_T2I_DATA_CONFIG,
    WAN_I2V_LORA_CONFIG,
    WAN_T2V_LORA_CONFIG,
    FluxTrainOptions,
    FluxTrainPreset,
    WanTrainOptions,
    WanTrainPreset,
    build_flux_train_argv,
    build_wan_train_argv,
)
from videotuna.utils.cli_console import install_pretty_tracebacks

PRESET_TRAIN_T2I = FluxTrainPreset(
    cli_name="train-domain-t2i",
    config_path=FLUX_T2I_CONFIG,
    data_config_path=FLUX_T2I_DATA_CONFIG,
)
PRESET_TRAIN_T2V = WanTrainPreset(
    cli_name="train-domain-t2v",
    config=WAN_T2V_LORA_CONFIG,
    ckpt="checkpoints/wan/Wan2.1-T2V-14B",
    expname="train_wan_domain_t2v_lora",
)
PRESET_TRAIN_I2V = WanTrainPreset(
    cli_name="train-domain-i2v",
    config=WAN_I2V_LORA_CONFIG,
    ckpt="checkpoints/wan/Wan2.1-I2V-14B-480P",
    expname="train_wan_domain_i2v_lora",
)


def _run_flux_training(
    flux: FluxTrainOptions | None,
    *,
    preset: FluxTrainPreset,
    **extra_cli: str,
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    argv = build_flux_train_argv(preset, flux, **extra_cli)
    result = subprocess.run(argv, check=False)
    raise SystemExit(result.returncode)


def _run_wan_training(
    wan: WanTrainOptions | None,
    *,
    preset: WanTrainPreset,
    **extra_cli: str,
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    argv = build_wan_train_argv(preset, wan, **extra_cli)
    result = subprocess.run(argv, check=False)
    raise SystemExit(result.returncode)


def _register_flux_command(
    app: App,
    preset: FluxTrainPreset,
    *,
    name: str | None = None,
    is_default: bool = False,
) -> None:
    command_name = name or preset.cli_name

    def handler(
        flux: FluxTrainOptions | None = None,
        **extra_cli: str,
    ) -> None:
        _run_flux_training(flux, preset=preset, **extra_cli)

    handler.__doc__ = (
        f"Train Flux T2I domain LoRA with config {preset.config_path} "
        f"and data config {preset.data_config_path}."
    )
    if is_default:
        app.default(handler)
    else:
        app.command(name=command_name)(handler)


def _register_wan_command(
    app: App,
    preset: WanTrainPreset,
    *,
    name: str | None = None,
    is_default: bool = False,
) -> None:
    command_name = name or preset.cli_name

    def handler(
        wan: WanTrainOptions | None = None,
        **extra_cli: str,
    ) -> None:
        _run_wan_training(wan, preset=preset, **extra_cli)

    handler.__doc__ = (
        f"Train Wan domain LoRA with YAML config {preset.config} "
        f"and base checkpoint {preset.ckpt}."
    )
    if is_default:
        app.default(handler)
    else:
        app.command(name=command_name)(handler)


def _make_flux_app(preset: FluxTrainPreset) -> App:
    app = App(name=preset.cli_name, help="PrivTune Flux T2I domain LoRA training.")
    _register_flux_command(app, preset, is_default=True)
    return app


def _make_wan_app(preset: WanTrainPreset) -> App:
    app = App(name=preset.cli_name, help="PrivTune Wan domain LoRA training.")
    _register_wan_command(app, preset, is_default=True)
    return app


install_pretty_tracebacks()

app = App(name="privtune-train", help="PrivTune domain LoRA training.")
_register_flux_command(app, PRESET_TRAIN_T2I, name="train-domain-t2i")
_register_wan_command(app, PRESET_TRAIN_T2V, name="train-domain-t2v")
_register_wan_command(app, PRESET_TRAIN_I2V, name="train-domain-i2v")


def _entry_for_flux_preset(preset: FluxTrainPreset) -> Callable[[], None]:
    cli_app = _make_flux_app(preset)

    def entry() -> None:
        raise SystemExit(cli_app(sys.argv[1:]))

    entry.__name__ = preset.cli_name.replace("-", "_")
    entry.__doc__ = f"Entry point for {preset.cli_name}."
    return entry


def _entry_for_wan_preset(preset: WanTrainPreset) -> Callable[[], None]:
    cli_app = _make_wan_app(preset)

    def entry() -> None:
        raise SystemExit(cli_app(sys.argv[1:]))

    entry.__name__ = preset.cli_name.replace("-", "_")
    entry.__doc__ = f"Entry point for {preset.cli_name}."
    return entry


def main() -> None:
    """Dispatch subcommands on the shared training App."""
    raise SystemExit(app(sys.argv[1:]))


train_domain_t2i_entry = _entry_for_flux_preset(PRESET_TRAIN_T2I)
train_domain_t2v_entry = _entry_for_wan_preset(PRESET_TRAIN_T2V)
train_domain_i2v_entry = _entry_for_wan_preset(PRESET_TRAIN_I2V)
train_flux_lora_entry = train_domain_t2i_entry
train_wan2_1_t2v_lora_entry = train_domain_t2v_entry
train_wan2_1_i2v_lora_entry = train_domain_i2v_entry
