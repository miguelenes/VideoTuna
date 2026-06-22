"""Tests for videotuna.utils.video_io."""

import numpy as np
import pytest

from videotuna.utils.video_io import sample_frame_indices


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
