"""Unified Diffusers pipeline flow for Flux T2I and Wan 2.2 T2V / I2V."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import FluxPipeline, WanImageToVideoPipeline, WanPipeline
from omegaconf import DictConfig

from videotuna.base.generation_base import GenerationBase
from videotuna.base.inference_manifest import InferenceSample
from videotuna.settings import get_settings
from videotuna.utils.common_utils import monitor_resources
from videotuna.utils.device_utils import resolve_inference_device
from videotuna.utils.diffusers_optimizations import (
    apply_diffusers_optimizations,
    transformer_cache_context,
)
from videotuna.utils.diffusers_quantization import (
    build_pipeline_quantization_config,
    normalize_quant_backend,
    normalize_transformer_quant,
    resolve_quant_components,
)
from videotuna.utils.logging_config import bound_logger, resolve_device_label
from videotuna.utils.wan_lora_bridge import (
    apply_native_wan_lora_to_pipeline,
    is_native_wan_lora_ckpt,
)

WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

FLUX_VARIANTS = {
    "1-dev": "black-forest-labs/FLUX.1-dev",
    "1-schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
    "schnell": "black-forest-labs/FLUX.1-schnell",
}

WAN_T2V_VARIANTS = {
    "2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "2.2": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
}

WAN_I2V_VARIANTS = {
    "2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "2.2": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
}

MODEL_REGISTRY: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("flux", "t2i"): {
        "pipeline_cls": FluxPipeline,
        "default_id": "black-forest-labs/FLUX.1-dev",
        "variants": FLUX_VARIANTS,
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


class DiffusersVideoFlow(GenerationBase):
    """Diffusers-native inference for Flux T2I and Wan 2.2 T2V / I2V."""

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
        self._inference_device: Optional[str] = None
        self._transformer_quant = "none"
        self._quant_backend = "torchao"
        self._log = bound_logger(phase=self.mode, flow="diffusers_video")

    def from_pretrained(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
        lora_ckpt_path: Optional[Union[str, Path]] = None,
        ignore_missing_ckpts: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        ckpt_str = str(ckpt_path) if ckpt_path is not None else None
        self._model_id = resolve_model_id(
            self.model_family,
            self.mode,
            ckpt_str or self.pretrained_model_name_or_path,
            self.model_variant,
        )
        if lora_ckpt_path is not None:
            self._lora_path = str(lora_ckpt_path)
        elif denoiser_ckpt_path is not None and self.model_family == "wan":
            self._lora_path = str(denoiser_ckpt_path)
        self._inference_device = device
        self._log.info(
            "DiffusersVideoFlow: model_id={} family={} mode={} lora={}",
            self._model_id,
            self.model_family,
            self.mode,
            self._lora_path,
        )

    def enable_vram_management(self):
        """No-op; optimizations are applied in inference() from CLI flags."""

    def eval(self) -> DiffusersVideoFlow:
        if self.pipeline is not None:
            self.pipeline.set_progress_bar_config(disable=False)
        return self

    def _require_pipeline(self) -> Any:
        assert self.pipeline is not None, "Pipeline is not loaded"
        return self.pipeline

    def _load_pipeline(self, dtype: torch.dtype) -> None:
        key = (self.model_family, self.mode)
        entry = MODEL_REGISTRY[key]
        pipeline_cls = entry["pipeline_cls"]
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        quant = normalize_transformer_quant(self._transformer_quant)
        if quant != "none":
            components = resolve_quant_components(
                self.model_family,
                self.model_variant,
                self.mode,
            )
            quant_config = build_pipeline_quantization_config(
                transformer_quant=quant,
                quant_backend=normalize_quant_backend(self._quant_backend),
                components=components,
            )
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
        self.pipeline = pipeline_cls.from_pretrained(self._model_id, **load_kwargs)
        self._load_lora_weights()

    def _load_lora_weights(self) -> None:
        if not self._lora_path:
            return
        pipeline = self._require_pipeline()
        if self.model_family == "flux":
            pipeline.load_lora_weights(self._lora_path)
            self._log.info("Loaded Flux LoRA weights from {}", self._lora_path)
            return
        if self.model_family == "wan":
            if is_native_wan_lora_ckpt(self._lora_path):
                if self.mode == "i2v":
                    from videotuna.utils.wan_lora_bridge import (
                        apply_native_wan_lora_to_i2v_pipeline,
                    )

                    reports = apply_native_wan_lora_to_i2v_pipeline(
                        pipeline, self._lora_path
                    )
                else:
                    reports = apply_native_wan_lora_to_pipeline(
                        pipeline, self._lora_path
                    )
                self._log.info(
                    "Applied native Wan 2.1 LoRA bridge from {} ({})",
                    self._lora_path,
                    [r.as_dict() for r in reports],
                )
                return
            pipeline.load_lora_weights(self._lora_path)
            self._log.info("Loaded Wan Diffusers LoRA from {}", self._lora_path)

    @torch.inference_mode()
    def inference(self, args: DictConfig) -> Dict[str, Any]:
        os.makedirs(args.savedir, exist_ok=True)
        if getattr(args, "lora_rank", None):
            self.lora_rank = int(args.lora_rank)
        if getattr(args, "lorackpt", None):
            self._lora_path = args.lorackpt
        if getattr(args, "trained_ckpt", None) and self.model_family == "wan":
            self._lora_path = args.trained_ckpt
        self._transformer_quant = normalize_transformer_quant(
            getattr(args, "transformer_quant", None)
        )
        self._quant_backend = normalize_quant_backend(
            getattr(args, "quant_backend", None)
        )
        self._dtype = resolve_torch_dtype(getattr(args, "dtype", None))
        inference_device = resolve_inference_device(
            getattr(args, "device", None) or self._inference_device
        )
        self._log = self._log.bind(device=resolve_device_label(inference_device))
        if self.pipeline is None:
            self._load_pipeline(self._dtype)
        pipeline = self._require_pipeline()

        if not hasattr(args, "fuse_qkv"):
            args.fuse_qkv = self.fuse_qkv
        if not hasattr(args, "enable_attention_cache"):
            args.enable_attention_cache = self.enable_attention_cache

        apply_diffusers_optimizations(
            pipeline,
            args,
            model_family=self.model_family,
            disable_progress_bar=False,
            device=inference_device,
        )

        seed = int(getattr(args, "seed", 42) or 42)
        n_samples = int(getattr(args, "n_samples_prompt", 1) or 1)
        fps = int(getattr(args, "savefps", None) or 8)
        frames = int(getattr(args, "frames", 49) or 49)

        prompt_source = getattr(args, "prompt_file", None)
        if self.mode == "i2v":
            prompt_source = getattr(args, "prompt_dir", None)
        samples = self.build_manifest_inputs(
            prompt_source,
            self.mode,
            args,
            n_samples=n_samples,
            seed=seed,
            per_sample_seed=True,
        )
        self.assign_output_paths(samples, args.savedir)

        manifest = self.create_manifest(
            model_id=self._model_id,
            lora_path=self._lora_path,
            model_family=self.model_family,
            mode=self.mode,
            config=args,
        )

        per_sample: List[Dict[str, Any]] = []
        gpu_metrics: List[float] = []
        time_metrics: List[float] = []

        for sample in samples:
            result = self._generate_sample(sample, args)
            per_sample.append(result)
            gpu_metrics.append(result.get("peak_vram_gb", -1.0))
            time_metrics.append(result.get("wall_time_s", -1.0))

            sample.peak_vram_gb = result.get("peak_vram_gb")
            sample.wall_time_s = result.get("wall_time_s")
            sample.seconds_per_frame = result.get("seconds_per_frame")
            sample.metrics = {k: v for k, v in result.items() if k not in {"result"}}
            manifest.add_sample(sample)
            self.save_output(sample, result["result"], fps=fps)

        metrics_file: Optional[str] = None
        if get_settings().metrics_owner == "flow":
            self.save_metrics(
                gpu=gpu_metrics,
                time=time_metrics,
                config=args,
                savedir=args.savedir,
                frames=frames if self.mode != "t2i" else 1,
            )
            metrics_file = os.path.join(args.savedir, "metrics.json")

        self.save_manifest(manifest, args.savedir, metrics_file=metrics_file)
        return {"per_sample": per_sample, "gpu": gpu_metrics, "time": time_metrics}

    @monitor_resources(return_metrics=True)
    def _generate_sample(
        self,
        sample: InferenceSample,
        args: DictConfig,
    ) -> Any:
        from PIL import Image

        generator = torch.Generator().manual_seed(sample.seed)
        pipe_kwargs: Dict[str, Any] = {
            "prompt": sample.prompt,
            "num_inference_steps": sample.num_inference_steps or 50,
            "generator": generator,
        }

        entry = MODEL_REGISTRY[(self.model_family, self.mode)]
        pipeline = self._require_pipeline()

        with transformer_cache_context(pipeline):
            if self.model_family == "flux":
                pipe_kwargs.update(
                    guidance_scale=sample.guidance_scale or 3.5,
                    height=sample.height or 768,
                    width=sample.width or 1360,
                    max_sequence_length=256,
                )
                output = pipeline(**pipe_kwargs).images[0]
            elif self.model_family == "wan":
                pipe_kwargs.update(
                    num_frames=sample.frames or 49,
                    guidance_scale=sample.guidance_scale or 6.0,
                )
                if sample.height is not None:
                    pipe_kwargs["height"] = sample.height
                if sample.width is not None:
                    pipe_kwargs["width"] = sample.width
                neg = getattr(args, "uncond_prompt", None) or entry.get(
                    "negative_prompt"
                )
                if neg:
                    pipe_kwargs["negative_prompt"] = neg
                if self.mode == "i2v":
                    if not sample.image_path:
                        raise ValueError("I2V generation requires an image path")
                    pipe_kwargs["image"] = Image.open(sample.image_path).convert("RGB")
                output = pipeline(**pipe_kwargs).frames[0]
            else:
                raise ValueError(f"Unknown model family: {self.model_family}")

        return output
