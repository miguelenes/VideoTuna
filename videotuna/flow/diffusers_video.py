"""Unified Diffusers pipeline flow for video and image generation."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    Flux2Pipeline,
    FluxPipeline,
    HunyuanVideo15ImageToVideoPipeline,
    HunyuanVideo15Pipeline,
    LTXPipeline,
    MochiPipeline,
    WanImageToVideoPipeline,
    WanPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from loguru import logger
from omegaconf import DictConfig

from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import monitor_resources
from videotuna.utils.diffusers_optimizations import (
    apply_diffusers_optimizations,
    transformer_cache_context,
)

WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

COGVIDEOX_VARIANTS = {
    "2b": "THUDM/CogVideoX-2b",
    "5b": "THUDM/CogVideoX-5b",
    "1.5": "THUDM/CogVideoX1.5-5B",
}

FLUX_VARIANTS = {
    "2-dev": "black-forest-labs/FLUX.2-dev",
    "2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "1-dev": "black-forest-labs/FLUX.1-dev",
    "1-schnell": "black-forest-labs/FLUX.1-schnell",
    # Legacy aliases
    "dev": "black-forest-labs/FLUX.1-dev",
    "schnell": "black-forest-labs/FLUX.1-schnell",
}

WAN_T2V_VARIANTS = {
    "2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "2.2": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
}

WAN_I2V_VARIANTS = {
    "2.1": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    "2.2": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
}

HUNYUAN_VARIANTS = {
    "720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
}

HUNYUAN_I2V_VARIANTS = {
    "720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
    "480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
}

MODEL_REGISTRY: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("cogvideox", "t2v"): {
        "pipeline_cls": CogVideoXPipeline,
        "default_id": "THUDM/CogVideoX1.5-5B",
        "variants": COGVIDEOX_VARIANTS,
        "scheduler": "dpm",
        "export_fps": 16,
    },
    ("cogvideox", "i2v"): {
        "pipeline_cls": CogVideoXImageToVideoPipeline,
        "default_id": "THUDM/CogVideoX1.5-5B-I2V",
        "variants": {
            **COGVIDEOX_VARIANTS,
            "5b-i2v": "THUDM/CogVideoX-5b-I2V",
            "1.5-i2v": "THUDM/CogVideoX1.5-5B-I2V",
        },
        "scheduler": "dpm",
        "export_fps": 16,
    },
    ("cogvideox", "v2v"): {
        "pipeline_cls": CogVideoXVideoToVideoPipeline,
        "default_id": "THUDM/CogVideoX1.5-5B",
        "variants": COGVIDEOX_VARIANTS,
        "scheduler": "dpm",
        "export_fps": 16,
    },
    ("flux", "t2i"): {
        "pipeline_cls": Flux2Pipeline,
        "legacy_pipeline_cls": FluxPipeline,
        "default_id": "black-forest-labs/FLUX.2-dev",
        "variants": FLUX_VARIANTS,
        "flux1_variants": {"dev", "schnell", "1-dev", "1-schnell"},
    },
    ("mochi", "t2v"): {
        "pipeline_cls": MochiPipeline,
        "default_id": "genmo/mochi-1-preview",
        "variant": "bf16",
        "export_fps": 30,
    },
    ("wan", "t2v"): {
        "pipeline_cls": WanPipeline,
        "default_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "variants": WAN_T2V_VARIANTS,
        "export_fps": 16,
        "negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
    },
    ("wan", "i2v"): {
        "pipeline_cls": WanImageToVideoPipeline,
        "default_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "variants": WAN_I2V_VARIANTS,
        "export_fps": 16,
        "negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
    },
    ("hunyuan", "t2v"): {
        "pipeline_cls": HunyuanVideo15Pipeline,
        "default_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "variants": HUNYUAN_VARIANTS,
        "export_fps": 24,
    },
    ("hunyuan", "i2v"): {
        "pipeline_cls": HunyuanVideo15ImageToVideoPipeline,
        "default_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        "variants": HUNYUAN_I2V_VARIANTS,
        "export_fps": 24,
    },
    ("ltx", "t2v"): {
        "pipeline_cls": LTXPipeline,
        "default_id": "Lightricks/LTX-Video",
        "export_fps": 24,
    },
}


def resolve_model_id(
    model_family: str,
    mode: str,
    pretrained_model_name_or_path: Optional[str],
    model_variant: Optional[str] = None,
) -> str:
    key = (model_family.lower(), mode.lower())
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported diffusers model: {model_family}/{mode}")
    entry = MODEL_REGISTRY[key]
    if pretrained_model_name_or_path:
        return pretrained_model_name_or_path
    variants = entry.get("variants")
    if variants and model_variant:
        return variants.get(model_variant, entry["default_id"])
    return entry["default_id"]


def resolve_torch_dtype(dtype_flag: Optional[str]) -> torch.dtype:
    if dtype_flag in ("fp16", "float16"):
        return torch.float16
    return torch.bfloat16


def _resolve_flux_pipeline_cls(entry: Dict[str, Any], model_variant: Optional[str]) -> Any:
    flux1_variants = entry.get("flux1_variants", set())
    if model_variant in flux1_variants:
        return entry.get("legacy_pipeline_cls", entry["pipeline_cls"])
    model_id = resolve_model_id("flux", "t2i", None, model_variant)
    if "FLUX.1" in model_id or "flux.1" in model_id.lower():
        return entry.get("legacy_pipeline_cls", entry["pipeline_cls"])
    return entry["pipeline_cls"]


def _hunyuan_attention_context(model_family: str):
    if model_family != "hunyuan":
        return nullcontext()
    try:
        from diffusers import attention_backend
    except ImportError:
        return nullcontext()
    backend = os.environ.get("VIDEOTUNA_ATTN_BACKEND", "auto")
    if backend == "flash":
        return attention_backend("flash_hub")
    return nullcontext()


class DiffusersVideoFlow(GenerationBase):
    """Diffusers-native inference for CogVideoX, Flux, Mochi, Wan, Hunyuan, and LTX."""

    def __init__(
        self,
        model_family: str,
        mode: str,
        pretrained_model_name_or_path: Optional[str] = None,
        pipeline_only: bool = True,
        model_variant: Optional[str] = None,
        lora_rank: int = 128,
        lora_weight_name: str = "pytorch_lora_weights.safetensors",
        fuse_qkv: bool = False,
        enable_attention_cache: bool = False,
        **kwargs,
    ):
        super().__init__(pipeline_only=True)
        self.model_family = model_family.lower()
        self.mode = mode.lower()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_variant = model_variant
        self.lora_rank = lora_rank
        self.lora_weight_name = lora_weight_name
        self.fuse_qkv = fuse_qkv
        self.enable_attention_cache = enable_attention_cache
        self._model_id: Optional[str] = None
        self._lora_path: Optional[str] = None
        self._dtype = torch.bfloat16

    def from_pretrained(
        self,
        ckpt_path: Optional[str] = None,
        denoiser_ckpt_path: Optional[str] = None,
        lora_ckpt_path: Optional[str] = None,
        ignore_missing_ckpts: bool = False,
    ):
        self._model_id = resolve_model_id(
            self.model_family,
            self.mode,
            ckpt_path or self.pretrained_model_name_or_path,
            self.model_variant,
        )
        self._lora_path = lora_ckpt_path
        logger.info(
            "DiffusersVideoFlow: model_id={} family={} mode={}",
            self._model_id,
            self.model_family,
            self.mode,
        )

    def enable_vram_management(self):
        """No-op; optimizations are applied in inference() from CLI flags."""

    def eval(self):
        if self.pipeline is not None:
            self.pipeline.set_progress_bar_config(disable=False)

    def _load_pipeline(self, dtype: torch.dtype) -> None:
        key = (self.model_family, self.mode)
        entry = MODEL_REGISTRY[key]
        if self.model_family == "flux":
            pipeline_cls = _resolve_flux_pipeline_cls(entry, self.model_variant)
        else:
            pipeline_cls = entry["pipeline_cls"]
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if self.model_family == "mochi":
            load_kwargs["variant"] = entry.get("variant", "bf16")
        self.pipeline = pipeline_cls.from_pretrained(self._model_id, **load_kwargs)
        self._configure_scheduler(entry)
        self._load_lora_weights()

    def _configure_scheduler(self, entry: Dict[str, Any]) -> None:
        if self.model_family != "cogvideox":
            return
        scheduler_kind = entry.get("scheduler", "dpm")
        model_id_lower = (self._model_id or "").lower()
        if "2b" in model_id_lower:
            scheduler_kind = "ddim"
        if scheduler_kind == "ddim":
            self.pipeline.scheduler = CogVideoXDDIMScheduler.from_config(
                self.pipeline.scheduler.config, timestep_spacing="trailing"
            )
        else:
            self.pipeline.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipeline.scheduler.config, timestep_spacing="trailing"
            )

    def _load_lora_weights(self) -> None:
        if not self._lora_path:
            return
        if self.model_family == "cogvideox":
            self.pipeline.load_lora_weights(
                self._lora_path,
                weight_name=self.lora_weight_name,
                adapter_name="videotuna-lora",
            )
            if hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters(
                    ["videotuna-lora"], [self.lora_rank / max(self.lora_rank, 1)]
                )
            elif hasattr(self.pipeline, "fuse_lora"):
                self.pipeline.fuse_lora(lora_scale=1.0 / self.lora_rank)
        elif self.model_family == "flux":
            self.pipeline.load_lora_weights(self._lora_path)
            logger.info("Loaded Flux LoRA weights from {}", self._lora_path)

    def _resolve_inputs(
        self, args: DictConfig
    ) -> Tuple[List[str], List[Optional[str]]]:
        if self.mode == "t2v" or self.mode == "t2i":
            prompts = self.load_inference_inputs(args.prompt_file, "t2v")
            return prompts, [None] * len(prompts)
        if self.mode == "i2v":
            prompts, images = self.load_inference_inputs(args.prompt_dir, "i2v")
            return prompts, images
        if self.mode == "v2v":
            prompt_dir = args.prompt_dir
            if not prompt_dir:
                raise ValueError("v2v mode requires --prompt_dir")
            prompts, _ = self.load_prompts_images(prompt_dir)
            videos = sorted(self.get_target_filelist(prompt_dir, ext="mp4"))
            if len(prompts) != len(videos):
                raise ValueError(
                    f"v2v: {len(prompts)} prompts but {len(videos)} videos "
                    f"in {prompt_dir}"
                )
            return prompts, videos
        raise ValueError(f"Unsupported mode: {self.mode}")

    @torch.inference_mode()
    def inference(self, args: DictConfig) -> Dict[str, Any]:
        os.makedirs(args.savedir, exist_ok=True)
        if getattr(args, "lora_rank", None):
            self.lora_rank = int(args.lora_rank)
        if getattr(args, "lorackpt", None):
            self._lora_path = args.lorackpt
        self._dtype = resolve_torch_dtype(getattr(args, "dtype", None))
        if self.pipeline is None:
            self._load_pipeline(self._dtype)

        if not hasattr(args, "fuse_qkv"):
            args.fuse_qkv = self.fuse_qkv
        if not hasattr(args, "enable_attention_cache"):
            args.enable_attention_cache = self.enable_attention_cache

        apply_diffusers_optimizations(
            self.pipeline,
            args,
            model_family=self.model_family,
            disable_progress_bar=False,
        )

        prompts, media_paths = self._resolve_inputs(args)
        num_steps = int(
            getattr(args, "num_inference_steps", None)
            or getattr(args, "ddim_steps", 50)
            or 50
        )
        guidance = float(
            getattr(args, "unconditional_guidance_scale", None)
            or getattr(args, "guidance_scale", 6.0)
            or 6.0
        )
        seed = int(getattr(args, "seed", 42) or 42)
        frames = int(getattr(args, "frames", 49) or 49)
        height = getattr(args, "height", None)
        width = getattr(args, "width", None)
        n_samples = int(getattr(args, "n_samples_prompt", 1) or 1)

        per_sample: List[Dict[str, Any]] = []
        gpu_metrics: List[float] = []
        time_metrics: List[float] = []

        for idx, (prompt, media_path) in enumerate(zip(prompts, media_paths)):
            for sample_idx in range(n_samples):
                sample_seed = seed + idx * n_samples + sample_idx
                result = self._generate_sample(
                    prompt=prompt,
                    media_path=media_path,
                    num_steps=num_steps,
                    guidance=guidance,
                    seed=sample_seed,
                    frames=frames,
                    height=height,
                    width=width,
                    args=args,
                )
                per_sample.append(result)
                gpu_metrics.append(result.get("peak_vram_gb", -1.0))
                time_metrics.append(result.get("wall_time_s", -1.0))
                self._save_output(
                    result["result"],
                    args,
                    prompt,
                    idx,
                    sample_idx,
                )

        self.save_metrics(
            gpu=gpu_metrics,
            time=time_metrics,
            config=args,
            savedir=args.savedir,
            frames=frames if self.mode != "t2i" else 1,
        )
        return {"per_sample": per_sample, "gpu": gpu_metrics, "time": time_metrics}

    @monitor_resources(return_metrics=True)
    def _generate_sample(
        self,
        prompt: str,
        media_path: Optional[str],
        num_steps: int,
        guidance: float,
        seed: int,
        frames: int,
        height: Optional[int],
        width: Optional[int],
        args: DictConfig,
    ) -> Any:
        generator = torch.Generator().manual_seed(seed)
        pipe_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "generator": generator,
        }

        entry = MODEL_REGISTRY[(self.model_family, self.mode)]

        with transformer_cache_context(self.pipeline):
            with _hunyuan_attention_context(self.model_family):
                if self.model_family == "cogvideox":
                    pipe_kwargs.update(
                        num_frames=frames,
                        guidance_scale=guidance,
                        use_dynamic_cfg=True,
                    )
                    if height is not None:
                        pipe_kwargs["height"] = height
                    if width is not None:
                        pipe_kwargs["width"] = width
                    if self.mode == "i2v":
                        pipe_kwargs["image"] = load_image(media_path)
                    elif self.mode == "v2v":
                        pipe_kwargs["video"] = load_video(media_path)
                    output = self.pipeline(**pipe_kwargs).frames[0]
                elif self.model_family == "flux":
                    pipe_kwargs.update(
                        guidance_scale=guidance,
                        height=height or 768,
                        width=width or 1360,
                    )
                    if isinstance(self.pipeline, FluxPipeline):
                        pipe_kwargs["max_sequence_length"] = 256
                    else:
                        pipe_kwargs["max_sequence_length"] = 512
                    output = self.pipeline(**pipe_kwargs).images[0]
                elif self.model_family == "mochi":
                    pipe_kwargs.update(
                        num_frames=frames,
                        guidance_scale=guidance,
                    )
                    if height is not None:
                        pipe_kwargs["height"] = height
                    if width is not None:
                        pipe_kwargs["width"] = width
                    neg = getattr(args, "uncond_prompt", None)
                    if neg:
                        pipe_kwargs["negative_prompt"] = neg
                    autocast_ctx = (
                        torch.autocast("cuda", self._dtype, cache_enabled=False)
                        if torch.cuda.is_available()
                        else torch.autocast("cpu", enabled=False)
                    )
                    with autocast_ctx:
                        output = self.pipeline(**pipe_kwargs).frames[0]
                elif self.model_family == "wan":
                    pipe_kwargs.update(
                        num_frames=frames,
                        guidance_scale=guidance,
                    )
                    if height is not None:
                        pipe_kwargs["height"] = height
                    if width is not None:
                        pipe_kwargs["width"] = width
                    neg = getattr(args, "uncond_prompt", None) or entry.get(
                        "negative_prompt"
                    )
                    if neg:
                        pipe_kwargs["negative_prompt"] = neg
                    if self.mode == "i2v":
                        pipe_kwargs["image"] = load_image(media_path)
                    output = self.pipeline(**pipe_kwargs).frames[0]
                elif self.model_family == "hunyuan":
                    pipe_kwargs.update(num_frames=frames)
                    if height is not None:
                        pipe_kwargs["height"] = height
                    if width is not None:
                        pipe_kwargs["width"] = width
                    neg = getattr(args, "uncond_prompt", None)
                    if neg:
                        pipe_kwargs["negative_prompt"] = neg
                    if self.mode == "i2v":
                        pipe_kwargs["image"] = load_image(media_path)
                    output = self.pipeline(**pipe_kwargs).frames[0]
                elif self.model_family == "ltx":
                    pipe_kwargs.update(
                        num_frames=frames,
                        guidance_scale=guidance,
                    )
                    if height is not None:
                        pipe_kwargs["height"] = height
                    if width is not None:
                        pipe_kwargs["width"] = width
                    neg = getattr(args, "uncond_prompt", None)
                    if neg:
                        pipe_kwargs["negative_prompt"] = neg
                    output = self.pipeline(**pipe_kwargs).frames[0]
                else:
                    raise ValueError(f"Unknown model family: {self.model_family}")

        return output

    def _save_output(
        self,
        output: Any,
        args: DictConfig,
        prompt: str,
        idx: int,
        sample_idx: int,
    ) -> None:
        entry = MODEL_REGISTRY[(self.model_family, self.mode)]
        safe_prompt = prompt[:80].replace("/", "_").replace(" ", "_")
        if self.mode == "t2i":
            filename = f"{idx:03d}_{sample_idx:02d}_{safe_prompt}.jpg"
            out_path = os.path.join(args.savedir, filename)
            output.save(out_path)
            return

        fps = int(getattr(args, "savefps", None) or entry.get("export_fps", 8))
        filename = f"{idx:03d}_{sample_idx:02d}_{safe_prompt}.mp4"
        out_path = os.path.join(args.savedir, filename)
        export_to_video(output, out_path, fps=fps)
