"""Tests for WanVideoModelFlow.log_images and ImageLogger Trackio wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from videotuna.flow.wanvideo import WanVideoModelFlow
from videotuna.utils.callbacks import ImageLogger
from videotuna.utils.training_metrics import log_preview_to_trackio


class DummyFlow(WanVideoModelFlow):
    """Minimal test double that inherits log_images without running __init__."""

    def __init__(self):
        pass


def make_t2v_flow() -> DummyFlow:
    flow = DummyFlow()
    flow.task = "t2v-14B"
    flow.seed = 42
    flow.cfg = MagicMock()
    flow.cfg.param_dtype = torch.float32
    flow.cfg.sample_shift = 5.0

    wan_t2v = MagicMock()
    wan_t2v.device = torch.device("cpu")
    wan_t2v.vae_stride = (4, 8, 8)
    wan_t2v.patch_size = (1, 2, 2)
    wan_t2v.num_train_timesteps = 1000
    wan_t2v.boundary = 0.7
    wan_t2v.sample_neg_prompt = ""
    wan_t2v.vae.model.z_dim = 16
    wan_t2v.text_encoder.return_value = [torch.randn(1, 512, 4096)]
    wan_t2v.vae.decode.return_value = [torch.randn(3, 5, 8, 8)]

    flow.wan_t2v = wan_t2v
    flow.low_denoiser = MagicMock()
    flow.high_denoiser = MagicMock()
    return flow


def test_log_images_returns_inputs_and_captions_on_cpu(monkeypatch):
    monkeypatch.setattr("videotuna.flow.wanvideo.gpu_is_available", lambda: False)

    flow = make_t2v_flow()
    batch = {
        "video": torch.randn(2, 3, 5, 8, 8),
        "caption": ["cat video", "dog video"],
    }

    batch_logs = flow.log_images(batch)

    assert "inputs" in batch_logs
    assert "caption" in batch_logs
    assert "samples" not in batch_logs
    assert batch_logs["inputs"].shape == (2, 3, 5, 8, 8)
    assert batch_logs["caption"] == ["cat video", "dog video"]
    flow.wan_t2v.vae.decode.assert_not_called()


def test_log_images_generates_samples_on_gpu(monkeypatch):
    monkeypatch.setattr("videotuna.flow.wanvideo.gpu_is_available", lambda: True)

    flow = make_t2v_flow()
    batch = {
        "video": torch.randn(2, 3, 5, 8, 8),
        "caption": ["cat video", "dog video"],
    }

    latent_shape = (16, 2, 1, 1)
    noise_pred = torch.randn(*latent_shape)
    flow.low_denoiser.return_value = [noise_pred]
    flow.high_denoiser.return_value = [noise_pred]

    scheduler_mock = MagicMock()
    scheduler_mock.timesteps = torch.tensor([999.0, 500.0])
    scheduler_mock.step.return_value = [torch.randn(*latent_shape)]
    monkeypatch.setattr(
        "videotuna.flow.wanvideo.FlowUniPCMultistepScheduler",
        lambda **_: scheduler_mock,
    )

    batch_logs = flow.log_images(batch, unconditional_guidance_scale=7.0)

    assert "samples" in batch_logs
    assert batch_logs["samples"].shape == (2, 3, 5, 8, 8)
    flow.wan_t2v.vae.decode.assert_called()


def test_log_images_respects_max_preview_samples(monkeypatch):
    monkeypatch.setattr("videotuna.flow.wanvideo.gpu_is_available", lambda: True)

    flow = make_t2v_flow()
    batch = {
        "video": torch.randn(4, 3, 5, 8, 8),
        "caption": ["a", "b", "c", "d"],
    }

    latent_shape = (16, 2, 1, 1)
    noise_pred = torch.randn(*latent_shape)
    flow.low_denoiser.return_value = [noise_pred]
    flow.high_denoiser.return_value = [noise_pred]

    scheduler_mock = MagicMock()
    scheduler_mock.timesteps = torch.tensor([999.0])
    scheduler_mock.step.return_value = [torch.randn(*latent_shape)]
    monkeypatch.setattr(
        "videotuna.flow.wanvideo.FlowUniPCMultistepScheduler",
        lambda **_: scheduler_mock,
    )

    batch_logs = flow.log_images(batch, max_preview_samples=2)

    assert batch_logs["samples"].shape[0] == 2


def test_image_logger_tensor_to_pil_image():
    logger = ImageLogger(batch_frequency=1, save_dir="/tmp")
    video = torch.randn(2, 3, 5, 8, 8)  # 5D video
    pil = logger._tensor_to_pil_image(video)
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"

    images = torch.randn(2, 3, 8, 8)  # 4D images
    pil = logger._tensor_to_pil_image(images)
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"


def test_image_logger_log_to_trackio():
    logger = ImageLogger(batch_frequency=1, save_dir="/tmp", metrics_backend="trackio")
    pl_module = MagicMock()
    pl_module.global_step = 10

    batch_logs = {
        "inputs": torch.randn(2, 3, 5, 8, 8),
        "caption": ["a", "b"],
    }

    with patch("videotuna.utils.callbacks.log_preview_to_trackio") as mock_trackio:
        logger.log_to_trackio(pl_module, batch_logs, "train")
        mock_trackio.assert_called_once()
        pos_args = mock_trackio.call_args[0]
        kw_args = mock_trackio.call_args[1]
        assert isinstance(pos_args[0], Image.Image)
        assert kw_args["step"] == 10
        assert kw_args["tag"] == "preview/train/inputs"


def test_image_logger_log_to_trackio_disabled_for_tensorboard():
    logger = ImageLogger(
        batch_frequency=1, save_dir="/tmp", metrics_backend="tensorboard"
    )
    pl_module = MagicMock()
    pl_module.global_step = 10

    with patch("videotuna.utils.callbacks.log_preview_to_trackio") as mock_trackio:
        logger.log_to_trackio(pl_module, {"inputs": torch.randn(1, 3, 8, 8)}, "train")
        mock_trackio.assert_not_called()


def test_log_preview_to_trackio_logs_image():
    image = Image.new("RGB", (8, 8), color="red")
    mock_trackio = MagicMock()
    with (
        patch.dict("sys.modules", {"trackio": mock_trackio}),
        patch("videotuna.utils.training_metrics.trackio_available", return_value=True),
    ):
        log_preview_to_trackio(image, step=5, tag="preview/test")
        mock_trackio.log.assert_called_once()
        args, _ = mock_trackio.log.call_args
        assert "preview/test" in args[0]
