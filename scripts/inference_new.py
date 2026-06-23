from pathlib import Path
from typing import cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from videotuna.base.generation_base import GenerationBase
from videotuna.cli.inference_options import InferenceRunConfig
from videotuna.settings import get_settings, inference_settings_session
from videotuna.utils.args_utils import prepare_inference_config
from videotuna.utils.attention import (
    get_attn_backend_requested,
    get_resolved_attn_backend,
    get_torch_compile_mode,
)
from videotuna.utils.common_utils import (
    instantiate_from_config,
    monitor_resources,
    save_metrics,
)
from videotuna.utils.device_utils import (
    checkpoint_available,
    describe_compute_environment,
    log_startup_device_summary,
    require_accelerator_for_flow,
    require_min_vram,
    resolve_cpu_mode,
    resolve_inference_device,
    snapshot_nvidia_smi,
)
from videotuna.utils.diffusers_optimizations import apply_flow_memory_config
from videotuna.utils.diffusers_quantization import (
    maybe_adjust_offload_for_quant,
    validate_transformer_quant,
)
from videotuna.utils.inference_cli import apply_cpu_smoke_limits
from videotuna.utils.inference_profile import resolve_inference_profile


def run_inference(run_config: InferenceRunConfig, gpu_num=1, rank=0, **kwargs):
    """

    Inference t2v/i2v models

    """

    with inference_settings_session(
        cpu_smoke=run_config.cpu_smoke,
        compile_flag=run_config.compile,
    ):
        try:
            _run_inference_impl(run_config, gpu_num=gpu_num, rank=rank, **kwargs)

        except RuntimeError as exc:
            smi = snapshot_nvidia_smi()

            if smi:
                logger.error("nvidia-smi snapshot:\n{}", smi)

            raise exc


def _prepare_inference_quant(
    run_config: InferenceRunConfig,
    inference_config,
) -> None:
    """Validate transformer quant settings before model load."""

    has_lora = bool(
        getattr(inference_config, "trained_ckpt", None)
        or getattr(inference_config, "lorackpt", None)
    )

    profile = resolve_inference_profile(run_config, apply_preset=False)

    transformer_quant = validate_transformer_quant(
        transformer_quant=getattr(inference_config, "transformer_quant", None),
        quant_backend=getattr(inference_config, "quant_backend", None),
        offload_mode=profile.offload_mode,
        compile_enabled=run_config.compile,
        has_lora=has_lora,
    )

    if transformer_quant != "none":
        maybe_adjust_offload_for_quant(inference_config, transformer_quant)


def _run_inference_impl(run_config: InferenceRunConfig, gpu_num=1, rank=0, **kwargs):
    assert Path(
        run_config.config
    ).exists(), f"Error: config file {run_config.config} NOT Found!"

    config = OmegaConf.load(run_config.config)

    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected YAML mapping config, got {type(config).__name__}")

    config = prepare_inference_config(run_config, config)

    inference_config = config.pop(
        "inference", OmegaConf.create(flags={"allow_objects": True})
    )

    seed_everything(inference_config.seed)

    flow_config = config.pop("flow", OmegaConf.create(flags={"allow_objects": True}))

    flow_target = flow_config.get("target", "")

    flow_params = flow_config.get("params", OmegaConf.create())

    cpu_mode = resolve_cpu_mode(cli_smoke=run_config.cpu_smoke)

    if cpu_mode == "smoke":
        apply_cpu_smoke_limits(inference_config, flow_config)

    device_prefer = getattr(inference_config, "device", None) or run_config.device

    if device_prefer is None and cpu_mode in ("smoke", "force"):
        device_prefer = "cpu"

    device = resolve_inference_device(device_prefer)

    inference_config.device = str(device)

    logger.info("Compute environment: {}", describe_compute_environment())

    _prepare_inference_quant(run_config, inference_config)

    require_accelerator_for_flow(
        flow_target,
        cpu_mode=cpu_mode,
        min_vram_gb=getattr(inference_config, "min_vram_gb", None),
        model_family=OmegaConf.select(flow_params, "model_family"),
        model_variant=OmegaConf.select(flow_params, "model_variant"),
        height=getattr(inference_config, "height", None),
        width=getattr(inference_config, "width", None),
        frames=getattr(inference_config, "frames", None),
    )

    min_vram = getattr(inference_config, "min_vram_gb", None)

    if min_vram is not None:
        require_min_vram(
            float(min_vram),
            device=device,
            context=f"Flow: {flow_target}",
        )

    profile = resolve_inference_profile(run_config, apply_preset=False)

    log_startup_device_summary(
        device,
        profile.dtype,
        get_resolved_attn_backend(),
        profile.offload_mode,
        attn_backend_requested=get_attn_backend_requested(),
        memory_preset=profile.memory_preset,
        compile_enabled=get_settings().torch_compile,
        compile_mode=get_torch_compile_mode(),
    )

    ckpt_path = getattr(inference_config, "ckpt_path", None)

    if ckpt_path and not checkpoint_available(ckpt_path, flow_target=flow_target):
        raise FileNotFoundError(
            f"Checkpoint path not found: {ckpt_path}\n"
            "Download model weights into checkpoints/ or pass a Hugging Face model id "
            "(org/model). See docs/checkpoints.md for setup."
        )

    # 1. create flow

    flow = cast(GenerationBase, instantiate_from_config(flow_config, resolve=True))

    flow.from_pretrained(
        inference_config.ckpt_path,
        inference_config.trained_ckpt,
        inference_config.lorackpt,
        device=str(device),
    )

    apply_flow_memory_config(flow, inference_config)

    flow.enable_vram_management()

    flow.eval()

    # 2. flow inference

    num_frames = int(getattr(inference_config, "frames", 1) or 1)

    device_index = (
        device.index if device.type == "cuda" and device.index is not None else 0
    )

    decorated_inference = monitor_resources(
        frames=num_frames,
        return_metrics=True,
        inference_config=inference_config,
        device_index=device_index,
    )(flow.inference)

    metrics = decorated_inference(inference_config)

    if metrics and inference_config.savedir:
        if get_settings().metrics_owner == "script":
            save_metrics(
                metrics=metrics,
                savedir=inference_config.savedir,
                config=inference_config,
            )


if __name__ == "__main__":
    from videotuna.cli.inference_app import generic_inference_entry

    generic_inference_entry()
