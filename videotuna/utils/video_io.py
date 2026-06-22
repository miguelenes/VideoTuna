"""Video frame sampling and decoding with decord / torchcodec / pyav fallbacks."""

from __future__ import annotations

import random
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from einops import rearrange

VideoBackend = Literal["auto", "decord", "torchcodec", "pyav"]


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_interval: int = 1,
    begin_index: Optional[int] = None,
) -> np.ndarray:
    """Sample frame indices matching TemporalRandomCrop randomness."""
    sample_length = num_frames * frame_interval
    rand_end = max(0, total_frames - sample_length - 1)
    if begin_index is None:
        begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + sample_length, total_frames)
    if end_index - begin_index < num_frames:
        raise ValueError(
            f"The video has not enough frames. total={total_frames}, "
            f"need sample_length={sample_length}"
        )
    return np.linspace(begin_index, end_index - 1, num_frames, dtype=int)


def get_video_frame_count(video_path: str) -> int:
    """Return total frame count using decord (lightweight metadata read)."""
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    reader = VideoReader(video_path, ctx=cpu(0))
    return len(reader)


def _read_decord(video_path: str, indices: Sequence[int]) -> torch.Tensor:
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    reader = VideoReader(video_path, ctx=cpu(0))
    idx = np.asarray(indices, dtype=np.int64)
    vframes = reader.get_batch(idx)
    return rearrange(vframes, "t h w c -> t c h w")


def _read_torchcodec(video_path: str, indices: Sequence[int]) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, device="cpu")
    idx = [int(i) for i in indices]
    frames = decoder.get_frames_at(indices=idx)
    data = frames.data
    if data.ndim == 4 and data.shape[-1] in (1, 3, 4):
        return rearrange(data, "t h w c -> t c h w")
    return data


def _read_pyav(video_path: str, indices: Sequence[int]) -> torch.Tensor:
    from torchvision.io import read_video

    video, _, _ = read_video(video_path, output_format="TCHW")
    idx = torch.as_tensor(indices, dtype=torch.long)
    return video.index_select(0, idx)


def read_video_frames(
    video_path: str,
    indices: Sequence[int],
    backend: VideoBackend = "auto",
) -> torch.Tensor:
    """Decode selected frames as TCHW uint8/float tensor."""
    backends: list[str]
    if backend == "auto":
        backends = ["decord", "torchcodec", "pyav"]
    else:
        backends = [backend]

    last_error: Optional[Exception] = None
    for name in backends:
        try:
            if name == "decord":
                return _read_decord(video_path, indices)
            if name == "torchcodec":
                return _read_torchcodec(video_path, indices)
            if name == "pyav":
                return _read_pyav(video_path, indices)
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to decode {video_path} with backends {backends}"
    ) from last_error


def init_video_worker() -> None:
    """Call once per DataLoader worker before decoding."""
    try:
        import decord

        decord.bridge.set_bridge("torch")
    except ImportError:
        pass
