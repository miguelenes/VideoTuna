"""CPU tests for Wan I2V pair dataset loading."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision.io import write_video

from videotuna.data.datasets import DatasetFromCSV
from videotuna.training.wan_lora.config import WanLoraTrainConfig, load_wan_lora_config

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_I2V_CONFIG = REPO_ROOT / "configs" / "domain" / "wan_i2v_lora.yaml"


def _write_test_mp4(path, num_frames: int = 90, height: int = 480, width: int = 832):
    frames = torch.randint(0, 255, (num_frames, 3, height, width), dtype=torch.uint8)
    write_video(
        str(path),
        frames.permute(0, 2, 3, 1),
        fps=24,
        video_codec="h264",
    )


@pytest.fixture
def i2v_pair_dataset(tmp_path):
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    images_dir.mkdir()
    videos_dir.mkdir()
    image_path = images_dir / "ref001.jpg"
    video_path = videos_dir / "clip001.mp4"
    Image.new("RGB", (832, 480), color=(10, 20, 30)).save(image_path)
    video_path.write_bytes(b"placeholder")
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "image_path,video_path,caption\n"
        f"{image_path},{video_path},"
        '"sks_style, slow pan"\n',
        encoding="utf-8",
    )
    fake_video = torch.randint(0, 255, (81, 3, 480, 832), dtype=torch.uint8)
    with (
        patch("videotuna.data.datasets.get_video_frame_count", return_value=100),
        patch("videotuna.data.datasets.read_video_frames", return_value=fake_video),
    ):
        yield DatasetFromCSV(
            str(csv_path),
            height=480,
            width=832,
            num_frames=81,
            frame_interval=1,
            image_to_video=False,
            train=True,
        )


def test_i2v_pair_dataset_emits_image_and_video(i2v_pair_dataset):
    with (
        patch("videotuna.data.datasets.get_video_frame_count", return_value=100),
        patch(
            "videotuna.data.datasets.read_video_frames",
            return_value=torch.randint(0, 255, (81, 3, 480, 832), dtype=torch.uint8),
        ),
    ):
        sample = i2v_pair_dataset[0]
    assert sample["caption"] == "sks_style, slow pan"
    assert sample["video"].shape == (3, 81, 480, 832)
    assert sample["image"].shape == (3, 1, 480, 832)


def test_image_to_video_clones_first_frame(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_path = videos_dir / "clip001.mp4"
    video_path.write_bytes(b"placeholder")
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        f"path,caption\n{video_path},sks_style clip\n",
        encoding="utf-8",
    )
    dataset = DatasetFromCSV(
        str(csv_path),
        height=480,
        width=832,
        num_frames=81,
        image_to_video=True,
        train=True,
    )
    with (
        patch("videotuna.data.datasets.get_video_frame_count", return_value=100),
        patch(
            "videotuna.data.datasets.read_video_frames",
            return_value=torch.randint(0, 255, (81, 3, 480, 832), dtype=torch.uint8),
        ),
    ):
        sample = dataset[0]
    assert torch.allclose(sample["image"], sample["video"][:, :1])


def test_pair_csv_requires_image_path_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("video_path,caption\na.mp4,cap\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_path"):
        DatasetFromCSV(str(csv_path), height=480, width=832, num_frames=16)


def test_image_to_video_true_rejects_pair_csv(tmp_path):
    csv_path = tmp_path / "pair.csv"
    csv_path.write_text(
        "image_path,video_path,caption\nimg.jpg,vid.mp4,caption\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image_to_video=true"):
        DatasetFromCSV(
            str(csv_path),
            height=480,
            width=832,
            num_frames=16,
            image_to_video=True,
        )


def test_image_to_video_true_requires_path_column(tmp_path):
    csv_path = tmp_path / "no_path.csv"
    csv_path.write_text("video_path,caption\nvid.mp4,caption\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'path' column"):
        DatasetFromCSV(
            str(csv_path),
            height=480,
            width=832,
            num_frames=16,
            image_to_video=True,
        )


def test_i2v_pair_dataset_integration_with_pyav(tmp_path):
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    images_dir.mkdir()
    videos_dir.mkdir()
    image_path = images_dir / "ref001.jpg"
    video_path = videos_dir / "clip001.mp4"
    Image.new("RGB", (832, 480), color=(10, 20, 30)).save(image_path)
    _write_test_mp4(video_path, num_frames=90)
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "image_path,video_path,caption\n"
        f"{image_path},{video_path},"
        '"sks_style, slow pan"\n',
        encoding="utf-8",
    )
    dataset = DatasetFromCSV(
        str(csv_path),
        height=480,
        width=832,
        num_frames=81,
        frame_interval=1,
        image_to_video=False,
        train=True,
        video_backend="av",
    )
    sample = dataset[0]
    assert sample["caption"] == "sks_style, slow pan"
    assert sample["video"].shape == (3, 81, 480, 832)
    assert sample["image"].shape == (3, 1, 480, 832)


# --- I2V CSV-layout contract tests ---


def test_i2v_mode_rejects_first_frame_csv_when_image_to_video_false(tmp_path):
    """i2v_mode + image_to_video=false must use image_path,video_path,caption."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("path,caption\nvid.mp4,sks_style clip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path,caption columns"):
        DatasetFromCSV(
            str(csv_path),
            height=480,
            width=832,
            num_frames=16,
            image_to_video=False,
            i2v_mode=True,
        )


def test_i2v_config_validates_with_i2v_mode_true():
    """The canonical wan_i2v_lora.yaml must pass validation."""
    cfg = load_wan_lora_config(WAN_I2V_CONFIG)
    assert cfg.flow.params.task == "i2v-14B"
    ds_params = cfg.train.data.params["train"]["params"]
    assert ds_params["i2v_mode"] is True


def test_i2v_config_fails_when_i2v_mode_is_false():
    """i2v-14B task with i2v_mode=false must be rejected at config load."""
    cfg = load_wan_lora_config(WAN_I2V_CONFIG)
    payload = cfg.model_dump(mode="json")
    payload["train"]["data"]["params"]["train"]["params"]["i2v_mode"] = False
    with pytest.raises(ValueError, match="i2v_mode must be true"):
        WanLoraTrainConfig.model_validate(payload)


def test_i2v_config_fails_when_i2v_mode_is_missing():
    """i2v-14B task without i2v_mode in dataset params must be rejected."""
    cfg = load_wan_lora_config(WAN_I2V_CONFIG)
    payload = cfg.model_dump(mode="json")
    del payload["train"]["data"]["params"]["train"]["params"]["i2v_mode"]
    with pytest.raises(ValueError, match="i2v_mode must be true"):
        WanLoraTrainConfig.model_validate(payload)


def test_i2v_config_rejects_extra_dataset_params():
    """i2v-14B dataset params must match the strict DatasetFromCSVParams schema."""
    cfg = load_wan_lora_config(WAN_I2V_CONFIG)
    payload = cfg.model_dump(mode="json")
    payload["train"]["data"]["params"]["train"]["params"]["unknown_key"] = 42
    with pytest.raises(ValueError, match="Invalid train.data.params.train.params"):
        WanLoraTrainConfig.model_validate(payload)
