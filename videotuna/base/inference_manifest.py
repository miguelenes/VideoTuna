"""Manifest schema for manifest-driven batch inference.

The manifest captures the full provenance of every generated sample so that
prompt-to-output mappings are preserved in a machine-readable format alongside
legacy metrics.json files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from videotuna.utils.common_utils import _strip_non_serializable_metrics


MANIFEST_VERSION = "1.0"


@dataclass
class InferenceSample:
    """Provenance for a single generated sample.

    The schema is intentionally identical for T2I, T2V, and I2V. Fields that
    do not apply to a mode (e.g. ``image_path`` for T2V/T2I) are omitted or
    ``None``.
    """

    sample_id: str
    prompt: str
    output_path: Optional[str] = None
    seed: int = 0
    mode: str = "t2v"
    index: int = 0
    sample_index: int = 0
    height: Optional[int] = None
    width: Optional[int] = None
    frames: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    image_path: Optional[str] = None
    fps: Optional[int] = None
    peak_vram_gb: Optional[float] = None
    wall_time_s: Optional[float] = None
    seconds_per_frame: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "index": self.index,
            "sample_index": self.sample_index,
            "mode": self.mode,
            "prompt": self.prompt,
            "output_path": self.output_path,
            "seed": self.seed,
        }
        if self.image_path is not None:
            result["image_path"] = self.image_path
        if self.height is not None:
            result["height"] = self.height
        if self.width is not None:
            result["width"] = self.width
        if self.frames is not None:
            result["frames"] = self.frames
        if self.num_inference_steps is not None:
            result["num_inference_steps"] = self.num_inference_steps
        if self.guidance_scale is not None:
            result["guidance_scale"] = self.guidance_scale
        if self.fps is not None:
            result["fps"] = self.fps
        if self.peak_vram_gb is not None:
            result["peak_vram_gb"] = self.peak_vram_gb
        if self.wall_time_s is not None:
            result["wall_time_s"] = self.wall_time_s
        if self.seconds_per_frame is not None:
            result["seconds_per_frame"] = self.seconds_per_frame
        if self.metrics:
            result["metrics"] = _strip_non_serializable_metrics(self.metrics)
        return result


@dataclass
class InferenceManifest:
    """Full manifest for a batch inference run.

    ``manifest.json`` is written next to the generated outputs and is intended
    to be the durable, machine-readable record of what was produced and how.
    """

    samples: List[InferenceSample] = field(default_factory=list)
    version: str = MANIFEST_VERSION
    model_id: Optional[str] = None
    lora_path: Optional[str] = None
    model_family: Optional[str] = None
    mode: Optional[str] = None
    metrics_file: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_sample(self, sample: InferenceSample) -> None:
        self.samples.append(sample)

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "version": self.version,
            "model_family": self.model_family,
            "mode": self.mode,
            "model_id": self.model_id,
            "lora_path": self.lora_path,
            "samples": [s.as_dict() for s in self.samples],
        }
        if self.metrics_file is not None:
            result["metrics_file"] = self.metrics_file
        if self.config is not None:
            result["config"] = self.config
        if self.extra:
            result["extra"] = _strip_non_serializable_metrics(self.extra)
        return result

    def write(self, savedir: str, filename: str = "manifest.json") -> str:
        Path(savedir).mkdir(parents=True, exist_ok=True)
        path = Path(savedir) / filename
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    def output_paths(self) -> List[str]:
        return [s.output_path for s in self.samples]
