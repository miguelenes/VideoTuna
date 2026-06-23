import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision
from einops import rearrange
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from videotuna.base.inference_manifest import InferenceManifest, InferenceSample


class InferenceBase:
    """
    Base class for inference models.

    Provides a manifest-driven batch pipeline shared by all inference flows:
    prompt loading, prompt-directory discovery, deterministic filename generation,
    output writing, and machine-readable manifest export.
    """

    def __init__(self):
        pass

    @staticmethod
    def save_video(vid_tensor: torch.Tensor, savepath: str, fps: int = 10) -> None:
        """
        Save a video tensor to the specified path.

        :param vid_tensor: The video tensor to be saved, shape ``[c, t, h, w]``.
        :param savepath: The path where the video will be saved.
        :param fps: Frames per second for the saved video. Default is 10.
        """
        assert vid_tensor.dim() == 4, "Invalid video tensor shape."
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = rearrange(video, "c t h w -> t c h w")
        video = (video + 1.0) / 2.0
        video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

        torchvision.io.write_video(
            savepath, video, fps=fps, video_codec="h264", options={"crf": "10"}
        )

    def save_metrics(
        self,
        gpu: List[float],
        time: List[float],
        config: DictConfig,
        savedir: str,
        frames: int = 1,
    ) -> None:
        """Write aggregated metrics.json beside the generated outputs."""
        from videotuna.utils.common_utils import save_metrics as write_metrics

        write_metrics(
            savedir=savedir,
            config=config,
            gpu=gpu,
            time=time,
            frames=frames,
        )

    @staticmethod
    def _load_prompts_from_txt(prompt_file: Union[str, Path]) -> List[str]:
        """Load and return a list of prompts from a text file, stripping whitespace."""
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip() != ""]

    @staticmethod
    def _load_prompts(prompts: Optional[Union[str, Path]]) -> List[str]:
        """Load prompts from a ``.txt`` file or treat the input as a single prompt."""
        if prompts is None:
            return []
        if os.path.isfile(prompts) and str(prompts).endswith(".txt"):
            return InferenceBase._load_prompts_from_txt(prompts)
        logger.info("Process the input path as a prompt")
        return [str(prompts)]

    @staticmethod
    def _get_target_filelist(data_dir: str, ext: str) -> List[str]:
        """Return sorted file paths matching the comma-separated extensions."""
        file_list = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(tuple(ext.split(",")))
        ]
        file_list.sort()
        if len(file_list) == 0:
            raise ValueError(f"No file with extensions {ext} found in {data_dir}.")
        return file_list

    @staticmethod
    def _load_prompts_images(prompt_dir: str) -> Tuple[List[str], List[str]]:
        """Load one prompt file and all matching images from a directory."""
        prompt_files = InferenceBase._get_target_filelist(prompt_dir, ext="txt")
        if len(prompt_files) > 1:
            logger.warning(
                "Warning: multiple prompt files exist. The one "
                f"{os.path.split(prompt_files[0])[1]} is used."
            )
            prompt_file = prompt_files[0]
        elif len(prompt_files) == 1:
            prompt_file = prompt_files[0]
        else:
            raise ValueError(f"Error: found NO prompt file in {prompt_dir}")

        prompt_list = InferenceBase._load_prompts_from_txt(prompt_file)
        image_path_list = InferenceBase._get_target_filelist(
            prompt_dir, ext="png,jpg,webp,jpeg"
        )
        return prompt_list, image_path_list

    @staticmethod
    def _sanitize_for_filename(text: str, max_length: int = 80) -> str:
        """Create a filesystem-safe basename from a prompt."""
        text = text[:max_length]
        text = re.sub(r"[^\w\s-]", "_", text)
        text = re.sub(r"[\s_]+", "_", text).strip("_")
        return text or "sample"

    @staticmethod
    def _output_extension(mode: str) -> str:
        """Return the file extension for a given mode."""
        if mode == "t2i":
            return "jpg"
        return "mp4"

    @staticmethod
    def _generate_filename(
        prompt: str,
        seed: int,
        image_path: Optional[str],
        mode: str,
        index: int,
        sample_index: int,
    ) -> str:
        """Generate a deterministic, collision-resistant filename.

        The filename includes a truncated hash of the full provenance tuple so that
        duplicate prompts/conditions are disambiguated without relying on the
        sanitized prompt text alone.
        """
        hash_input = f"{prompt}:{seed}:{image_path or ''}:{index}:{sample_index}"
        digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
        safe_prompt = InferenceBase._sanitize_for_filename(prompt)
        ext = InferenceBase._output_extension(mode)
        return f"{index:04d}_{sample_index:03d}_{safe_prompt}_{digest}.{ext}"

    def build_manifest_inputs(
        self,
        prompt_source: Optional[Union[str, Path]],
        mode: str,
        args: Any,
        *,
        n_samples: int = 1,
        seed: int = 42,
        per_sample_seed: bool = True,
    ) -> List[InferenceSample]:
        """Resolve prompts and conditions into a flat list of ``InferenceSample``.

        T2V/T2I receive a prompt file or a single prompt. I2V receives a prompt
        directory and pairs each prompt with a discovered image. All modes share
        the same output schema, so callers can iterate over samples without ad hoc
        branching.
        """
        if prompt_source is None:
            raise ValueError("Please provide a valid prompts or prompts path.")

        mode = mode.lower().strip()
        if mode == "t2i":
            prompts = self._load_prompts(prompt_source)
            images: List[Optional[str]] = [None] * len(prompts)
        elif mode == "t2v":
            prompts = self._load_prompts(prompt_source)
            images = [None] * len(prompts)
        elif mode == "i2v":
            prompts, images = self._load_prompts_images(prompt_source)
            if len(prompts) != len(images):
                raise ValueError(
                    f"I2V prompt/image count mismatch: {len(prompts)} prompts, "
                    f"{len(images)} images in {prompt_source}"
                )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported: t2i, t2v, i2v.")

        num_steps = int(
            getattr(args, "num_inference_steps", None)
            or getattr(args, "ddim_steps", None)
            or 50
        )
        guidance = float(
            getattr(args, "unconditional_guidance_scale", None)
            or getattr(args, "guidance_scale", None)
            or 6.0
        )
        height = getattr(args, "height", None)
        width = getattr(args, "width", None)
        frames = getattr(args, "frames", None)
        fps = getattr(args, "savefps", None) or getattr(args, "fps", None)

        samples: List[InferenceSample] = []
        for idx, (prompt, image_path) in enumerate(zip(prompts, images)):
            for sample_idx in range(max(n_samples, 1)):
                sample_seed = seed
                if per_sample_seed:
                    sample_seed = seed + idx * max(n_samples, 1) + sample_idx

                sample_id = f"{mode}-{idx:04d}-{sample_idx:03d}-{sample_seed}"
                samples.append(
                    InferenceSample(
                        sample_id=sample_id,
                        prompt=prompt,
                        image_path=image_path,
                        mode=mode,
                        index=idx,
                        sample_index=sample_idx,
                        seed=sample_seed,
                        height=int(height) if height is not None else None,
                        width=int(width) if width is not None else None,
                        frames=int(frames) if frames is not None else None,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        fps=int(fps) if fps is not None else None,
                    )
                )
        return samples

    def assign_output_paths(
        self,
        samples: List[InferenceSample],
        savedir: str,
    ) -> None:
        """Assign collision-free output paths to every sample in the batch."""
        Path(savedir).mkdir(parents=True, exist_ok=True)
        seen: set = set()
        for sample in samples:
            filename = self._generate_filename(
                sample.prompt,
                sample.seed,
                sample.image_path,
                sample.mode,
                sample.index,
                sample.sample_index,
            )
            output_path = os.path.join(savedir, filename)

            # Hash collisions are vanishingly unlikely, but guard against them
            # and duplicate sample records by appending a counter.
            counter = 0
            original_output_path = output_path
            while output_path in seen or os.path.exists(output_path):
                counter += 1
                base, ext = os.path.splitext(original_output_path)
                output_path = f"{base}_{counter:03d}{ext}"

            seen.add(output_path)
            sample.output_path = output_path

    def create_manifest(
        self,
        model_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        model_family: Optional[str] = None,
        mode: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> InferenceManifest:
        """Create an ``InferenceManifest`` for the current run."""
        config_dict: Optional[Dict[str, Any]] = None
        if config is not None:
            if isinstance(config, DictConfig):
                config_dict = OmegaConf.to_container(config, resolve=True)
            elif hasattr(config, "items"):
                config_dict = dict(config)
            elif hasattr(config, "__dict__"):
                config_dict = vars(config)
        return InferenceManifest(
            model_id=model_id,
            lora_path=lora_path,
            model_family=model_family,
            mode=mode,
            config=config_dict,
        )

    def save_output(
        self, sample: InferenceSample, output: Any, fps: Optional[int] = None
    ) -> None:
        """Save one generated sample to its assigned path.

        Dispatches to the correct writer based on the sample mode and output type:
        PIL images for T2I, Diffusers frame lists for Diffusers video, or torch
        tensors for native Wan video.
        """
        if sample.output_path is None:
            raise ValueError(f"Sample {sample.sample_id} has no output_path")

        Path(os.path.dirname(sample.output_path)).mkdir(parents=True, exist_ok=True)

        if sample.mode == "t2i":
            # Diffusers Flux returns a PIL Image
            if hasattr(output, "save") and callable(output.save):
                output.save(sample.output_path)
            else:
                raise TypeError(
                    f"T2I output for {sample.sample_id} is not a PIL-like image"
                )
            return

        # Video output: either a native tensor or a Diffusers frame list.
        if isinstance(output, torch.Tensor):
            self.save_video(output, sample.output_path, fps=fps or sample.fps or 10)
            return

        # Diffusers video output is a list of PIL frames.
        from diffusers.utils import export_to_video

        export_to_video(output, sample.output_path, fps=fps or sample.fps or 8)

    def save_manifest(
        self,
        manifest: InferenceManifest,
        savedir: str,
        *,
        metrics_file: Optional[str] = None,
    ) -> str:
        """Write the manifest to ``savedir/manifest.json``."""
        if metrics_file is not None:
            manifest.metrics_file = metrics_file
        return manifest.write(savedir)
