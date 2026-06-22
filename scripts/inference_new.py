from typing import cast
import os
import sys
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

sys.path.insert(0, os.getcwd())
sys.path.insert(1, f"{os.getcwd()}/src")

from videotuna.base.generation_base import GenerationBase
from videotuna.utils.args_utils import prepare_inference_args
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
from videotuna.utils.fp8_utils import validate_fp8_inference
from videotuna.utils.inference_cli import (
    add_standard_inference_flags,
    apply_compile_env,
    apply_cpu_smoke_limits,
    resolve_offload_mode,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="inference mode: t2v/i2v",
    )
    #
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument(
        "--lorackpt",
        type=str,
        default=None,
        help="[Optional] checkpoint path for lora model. ",
    )
    parser.add_argument(
        "--trained_ckpt", type=str, default=None, help="denoiser full checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="model config (yaml) path"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="a text file containing many prompts for text-to-video",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help=(
            "a input dir containing images and prompts for "
            "image-to-video/interpolation"
        ),
    )
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument(
        "--standard_vbench",
        action="store_true",
        default=None,
        help="inference standard vbench prompts",
    )
    #
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    #
    parser.add_argument(
        "--height", type=int, default=None, help="video height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="video width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="video frame number, in pixel space"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "video motion speed. 512 or 1024 model: large value -> slow motion; "
            "256 model: large value -> large motion;"
        ),
    )
    parser.add_argument(
        "--n_samples_prompt",
        type=int,
        default=None,
        help="num of samples per prompt",
    )
    #
    parser.add_argument("--bs", type=int, default=None, help="batch size for inference")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument(
        "--uncond_prompt",
        type=str,
        default=None,
        help="unconditional prompts, or negative prompts",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=None,
        help="prompt classifier-free guidance",
    )
    parser.add_argument(
        "--unconditional_guidance_scale_temporal",
        type=float,
        default=None,
        help="temporal consistency guidance",
    )
    # dc args
    parser.add_argument(
        "--multiple_cond_cfg",
        action="store_true",
        default=None,
        help="i2v: use multi-condition cfg or not",
    )
    parser.add_argument(
        "--cfg_img",
        type=float,
        default=None,
        help="guidance scale for image conditioning",
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default=None,
        help=(
            "The way the timesteps should be scaled. Refer to Table 2 of "
            "[Common Diffusion Noise Schedules and Sample Steps are Flawed]"
            "(https://huggingface.co/papers/2305.08891) for more information."
        ),
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=None,
        help=(
            "guidance rescale in [Common Diffusion Noise Schedules and "
            "Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=None,
        help="generate looping videos or not",
    )
    parser.add_argument(
        "--gfi",
        action="store_true",
        default=None,
        help="generate generative frame interpolation (gfi) or not",
    )
    parser.add_argument(
        "--savefps", type=str, default=None, help="video fps to generate"
    )
    parser.add_argument(
        "--time_shift",
        type=float,
        default=None,
        help="time shift",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="sampling steps",
    )
    parser.add_argument(
        "--dit_weight",
        type=str,
        default=None,
        help="hunyuan dit weight",
    )
    parser.add_argument(
        "--i2v_resolution",
        type=str,
        default=None,
        help="target resolution",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank for CogVideoX adapter scaling (default: 128).",
    )
    add_standard_inference_flags(parser)
    return parser


def run_inference(args, gpu_num=1, rank=0, **kwargs):
    """
    Inference t2v/i2v models
    """
    try:
        _run_inference_impl(args, gpu_num=gpu_num, rank=rank, **kwargs)
    except RuntimeError as exc:
        smi = snapshot_nvidia_smi()
        if smi:
            logger.error("nvidia-smi snapshot:\n{}", smi)
        raise exc


def _run_inference_impl(args, gpu_num=1, rank=0, **kwargs):
    # load and replace inference args with user agrgument
    assert Path(args.config).exists(), f"Error: config file {args.config} NOT Found!"
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected YAML mapping config, got {type(config).__name__}")
    config = prepare_inference_args(args, config)

    inference_config = config.pop(
        "inference", OmegaConf.create(flags={"allow_objects": True})
    )
    seed_everything(inference_config.seed)

    flow_config = config.pop("flow", OmegaConf.create(flags={"allow_objects": True}))
    flow_target = flow_config.get("target", "")
    flow_params = flow_config.get("params", OmegaConf.create())

    cpu_mode = resolve_cpu_mode(cli_smoke=bool(getattr(args, "cpu_smoke", False)))
    if cpu_mode == "smoke":
        apply_cpu_smoke_limits(inference_config, flow_config)

    device_prefer = getattr(inference_config, "device", None) or getattr(args, "device", None)
    if device_prefer is None and cpu_mode in ("smoke", "force"):
        device_prefer = "cpu"
    device = resolve_inference_device(device_prefer)
    inference_config.device = str(device)

    logger.info("Compute environment: {}", describe_compute_environment())

    apply_compile_env(bool(getattr(args, "compile", False)))
    if getattr(args, "enable_fp8", False):
        dit_weight = getattr(inference_config, "dit_weight", None) or getattr(
            inference_config, "trained_ckpt", None
        )
        validate_fp8_inference(str(dit_weight) if dit_weight else "")

    require_accelerator_for_flow(
        flow_target,
        cpu_mode=cpu_mode,
        min_vram_gb=getattr(inference_config, "min_vram_gb", None),
        model_family=OmegaConf.select(flow_params, "model_family"),
        model_variant=OmegaConf.select(flow_params, "model_variant"),
        height=getattr(inference_config, "height", None),
        width=getattr(inference_config, "width", None),
    )

    min_vram = getattr(inference_config, "min_vram_gb", None)
    if min_vram is not None:
        require_min_vram(
            float(min_vram),
            device=device,
            context=f"Flow: {flow_target}",
        )

    log_startup_device_summary(
        device,
        getattr(inference_config, "dtype", None),
        get_resolved_attn_backend(),
        resolve_offload_mode(inference_config),
        attn_backend_requested=get_attn_backend_requested(),
        memory_preset=getattr(inference_config, "memory_preset", None),
        compile_enabled=os.environ.get("VIDEOTUNA_TORCH_COMPILE", "0") == "1",
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
    device_index = device.index if device.type == "cuda" and device.index is not None else 0
    decorated_inference = monitor_resources(
        frames=num_frames,
        return_metrics=True,
        inference_config=inference_config,
        device_index=device_index,
    )(flow.inference)
    metrics = decorated_inference(inference_config)
    if metrics and inference_config.savedir:
        if os.environ.get("VIDEOTUNA_METRICS_OWNER", "script") == "script":
            save_metrics(
                metrics=metrics,
                savedir=inference_config.savedir,
                config=inference_config,
            )


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_inference(args)
