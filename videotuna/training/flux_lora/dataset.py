"""Local image + caption dataset for Flux LoRA training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from videotuna.training.flux_lora.config import FluxLoraDataConfig

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FluxLoraSample(TypedDict):
    pixel_values: torch.Tensor
    caption: str


def _load_caption(
    image_path: Path, caption_strategy: str, default_caption: str | None
) -> str:
    if caption_strategy == "filename":
        txt_path = image_path.with_suffix(".txt")
        if txt_path.is_file():
            return txt_path.read_text(encoding="utf-8").strip()
        if default_caption:
            return default_caption
        raise ValueError(
            f"Missing caption file for {image_path} (caption_strategy=filename)"
        )
    if default_caption:
        return default_caption
    raise ValueError(
        f"Unsupported caption_strategy={caption_strategy!r} without default caption"
    )


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


class FluxLoraImageDataset(Dataset):
    def __init__(self, data_config: FluxLoraDataConfig):
        self.data_dir = Path(data_config.instance_data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Training data directory not found: {self.data_dir}"
            )

        self.caption_strategy = data_config.caption_strategy
        self.default_caption = data_config.default_caption
        self.resolution = data_config.resolution
        self.crop = data_config.crop

        self.samples: list[tuple[Path, str]] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            caption = _load_caption(path, self.caption_strategy, self.default_caption)
            self.samples.append((path, caption))

        if not self.samples:
            raise ValueError(f"No training images found in {self.data_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        logger.info(
            "Loaded %d training images from %s", len(self.samples), self.data_dir
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FluxLoraSample:
        path, caption = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.crop:
            image = _center_square_crop(image)
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "caption": caption}
