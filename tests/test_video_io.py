"""Tests for videotuna.utils.video_io."""

import pytest
import torch
from torchvision.io import write_video

from videotuna.utils.video_io import (
    AvVideoReader,
    get_video_frame_count,
    get_video_fps,
    read_video_frames,
    sample_frame_indices,
)


@pytest.fixture
def tiny_mp4(tmp_path):
    path = tmp_path / "test.mp4"
    num_frames = 24
    frames = torch.randint(0, 255, (num_frames, 3, 64, 64), dtype=torch.uint8)
    write_video(
        str(path),
        frames.permute(0, 2, 3, 1),
        fps=8,
        video_codec="h264",
    )
    return path, num_frames


def test_sample_frame_indices_length():
    indices = sample_frame_indices(100, num_frames=16, frame_interval=1, begin_index=0)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] <= 99


def test_sample_frame_indices_with_interval():
    indices = sample_frame_indices(200, num_frames=8, frame_interval=4, begin_index=10)
    assert len(indices) == 8
    assert indices[0] == 10
    assert indices[-1] <= 10 + 8 * 4


def test_sample_frame_indices_rejects_short_video():
    with pytest.raises(ValueError):
        sample_frame_indices(10, num_frames=16, frame_interval=1)


def test_sample_frame_indices_random_begin():
    runs = [sample_frame_indices(120, 16, 1)[0] for _ in range(20)]
    assert min(runs) >= 0
    assert max(runs) <= 120 - 16


def test_get_video_frame_count(tiny_mp4):
    path, num_frames = tiny_mp4
    count = get_video_frame_count(str(path))
    assert count >= num_frames - 2
    assert count <= num_frames + 2


def test_get_video_fps(tiny_mp4):
    path, _ = tiny_mp4
    fps = get_video_fps(str(path))
    assert fps > 0


def test_read_video_frames_subset(tiny_mp4):
    path, num_frames = tiny_mp4
    indices = [0, num_frames // 2, num_frames - 1]
    frames = read_video_frames(str(path), indices, backend="av")
    assert frames.shape == (3, 3, 64, 64)
    assert frames.dtype == torch.uint8


def test_read_video_frames_rejects_missing_index(tiny_mp4):
    path, num_frames = tiny_mp4
    with pytest.raises(RuntimeError, match="Failed to decode"):
        read_video_frames(str(path), [num_frames + 50], backend="av")


def test_av_video_reader_batch_shape(tiny_mp4):
    path, num_frames = tiny_mp4
    reader = AvVideoReader(str(path))
    batch = reader.get_batch([0, num_frames - 1])
    arr = batch.asnumpy()
    assert arr.shape == (2, 64, 64, 3)
    assert arr.dtype.name == "uint8"
    assert len(reader) >= num_frames - 2
    assert reader.get_avg_fps() > 0
