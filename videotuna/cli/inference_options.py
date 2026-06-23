"""Typed cyclopts option groups for inference entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from videotuna.utils.inference_profile import MemoryPreset

DtypeChoice = Literal["bf16", "fp16", "fp32"]
DeviceMapChoice = Literal["auto"]


@Parameter(name="*")
class StandardInferenceOptions(BaseModel):
    """Memory, device, and performance flags shared by all inference commands."""

    model_config = ConfigDict(extra="forbid")

    cpu_smoke: Annotated[bool | None, Parameter(name="cpu-smoke")] = None
    device: Annotated[str | None, Parameter(name="device", alias="--gpu-id")] = None
    min_vram_gb: Annotated[float | None, Parameter(name="min-vram-gb")] = None
    memory_preset: Annotated[MemoryPreset | None, Parameter(name="memory-preset")] = (
        None
    )
    enable_vae_tiling: Annotated[bool | None, Parameter(name="enable_vae_tiling")] = (
        None
    )
    enable_vae_slicing: Annotated[bool | None, Parameter(name="enable_vae_slicing")] = (
        None
    )
    enable_model_cpu_offload: Annotated[
        bool | None, Parameter(name="enable_model_cpu_offload")
    ] = None
    enable_sequential_cpu_offload: Annotated[
        bool | None, Parameter(name="enable_sequential_cpu_offload")
    ] = None
    dtype: DtypeChoice | None = None
    device_map: Annotated[DeviceMapChoice | None, Parameter(name="device-map")] = None
    max_memory_per_gpu: Annotated[
        str | None,
        Parameter(
            name="max-memory-per-gpu",
            help="Per-GPU memory limit for device_map=auto (e.g. '22GiB').",
        ),
    ] = None
    ulysses_degree: int | None = None
    ring_degree: int | None = None
    compile: Annotated[bool | None, Parameter(name="compile")] = None
    fuse_qkv: bool | None = None
    enable_attention_cache: bool | None = None
    transformer_quant: Annotated[
        str | None,
        Parameter(
            name="transformer-quant",
            help=(
                "Diffusers transformer weight-only quant: "
                "none, int8_wo, int4_wo, fp8_wo (CUDA; fp8_wo needs Ada/Hopper+)."
            ),
        ),
    ] = None
    quant_backend: Annotated[str | None, Parameter(name="quant-backend")] = None


@Parameter(name="*")
class InferenceRunOptions(BaseModel):
    """Model, prompt, and sampling flags for inference."""

    model_config = ConfigDict(extra="forbid")

    mode: str | None = None
    ckpt_path: str | None = None
    lorackpt: str | None = None
    trained_ckpt: str | None = None
    config: str | None = None
    prompt_file: str | None = None
    prompt_dir: str | None = None
    savedir: str | None = None
    standard_vbench: bool | None = None
    seed: int | None = None
    height: int | None = None
    width: int | None = None
    frames: int | None = None
    fps: int | None = None
    n_samples_prompt: int | None = None
    bs: int | None = None
    ddim_steps: int | None = None
    ddim_eta: float | None = None
    uncond_prompt: str | None = None
    unconditional_guidance_scale: float | None = None
    unconditional_guidance_scale_temporal: float | None = None
    multiple_cond_cfg: bool | None = None
    cfg_img: float | None = None
    timestep_spacing: str | None = None
    guidance_rescale: float | None = None
    loop: bool | None = None
    gfi: bool | None = None
    savefps: str | None = None
    time_shift: float | None = None
    num_inference_steps: int | None = None
    i2v_resolution: str | None = None
    lora_rank: int | None = None


@dataclass(frozen=True)
class InferencePreset:
    """Baked-in defaults for a Poetry inference entrypoint."""

    cli_name: str
    config: str
    enable_model_cpu_offload: bool = False
    require_checkpoint: bool = False
    require_prompt_dir: bool = False


class InferenceRunConfig(BaseModel):
    """Validated inference CLI options consumed directly by ``run_inference``."""

    model_config = ConfigDict(extra="forbid")

    config: str
    mode: str | None = None
    ckpt_path: str | None = None
    lorackpt: str | None = None
    trained_ckpt: str | None = None
    prompt_file: str | None = None
    prompt_dir: str | None = None
    savedir: str | None = None
    standard_vbench: bool | None = None
    seed: int | None = None
    height: int | None = None
    width: int | None = None
    frames: int | None = None
    fps: int | None = None
    n_samples_prompt: int | None = None
    bs: int | None = None
    ddim_steps: int | None = None
    ddim_eta: float | None = None
    uncond_prompt: str | None = None
    unconditional_guidance_scale: float | None = None
    unconditional_guidance_scale_temporal: float | None = None
    multiple_cond_cfg: bool | None = None
    cfg_img: float | None = None
    timestep_spacing: str | None = None
    guidance_rescale: float | None = None
    loop: bool | None = None
    gfi: bool | None = None
    savefps: str | None = None
    time_shift: float | None = None
    num_inference_steps: int | None = None
    i2v_resolution: str | None = None
    lora_rank: int | None = None
    cpu_smoke: bool = False
    device: str | None = None
    min_vram_gb: float | None = None
    memory_preset: MemoryPreset | None = None
    enable_vae_tiling: bool = False
    enable_vae_slicing: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    dtype: DtypeChoice | None = None
    device_map: DeviceMapChoice | None = None
    max_memory_per_gpu: str | None = None
    ulysses_degree: int | None = None
    ring_degree: int | None = None
    compile: bool = False
    fuse_qkv: bool = False
    enable_attention_cache: bool = False
    transformer_quant: str | None = None
    quant_backend: str | None = None


def inference_options_to_config(
    *,
    run: InferenceRunOptions | None = None,
    standard: StandardInferenceOptions | None = None,
    preset: InferencePreset | None = None,
) -> InferenceRunConfig:
    """Flatten typed option groups into a validated inference config."""
    merged: dict[str, Any] = {}

    if preset is not None:
        merged["config"] = preset.config
        if preset.enable_model_cpu_offload:
            merged["enable_model_cpu_offload"] = True

    for key, value in (
        (run or InferenceRunOptions()).model_dump(exclude_none=True).items()
    ):
        merged[key] = value

    for key, value in (
        (standard or StandardInferenceOptions()).model_dump(exclude_none=True).items()
    ):
        merged[key] = value

    if "config" not in merged:
        raise ValueError("Inference requires a YAML config path (--config or preset).")

    bool_defaults = (
        "enable_vae_tiling",
        "enable_vae_slicing",
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "compile",
        "fuse_qkv",
        "enable_attention_cache",
    )
    for key in bool_defaults:
        merged.setdefault(key, False)
    merged.setdefault("cpu_smoke", False)

    return InferenceRunConfig.model_validate(merged)


def validate_preset_requirements(
    run: InferenceRunOptions,
    preset: InferencePreset,
) -> None:
    """Enforce checkpoint / prompt requirements for validation entrypoints."""
    import sys

    if preset.require_checkpoint and not (run.trained_ckpt or run.lorackpt):
        print(
            f"Error: {preset.cli_name} requires --trained_ckpt <denoiser.ckpt> "
            "or --lorackpt <checkpoint-dir>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if preset.require_prompt_dir and not run.prompt_dir:
        print(
            f"Error: {preset.cli_name} requires --prompt_dir <image+prompt pairs>",
            file=sys.stderr,
        )
        raise SystemExit(2)
