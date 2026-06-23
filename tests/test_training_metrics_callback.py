"""Tests for training metrics callback."""

import json
import os
import tempfile
import time
from unittest import mock

import pytest
import torch

from videotuna.utils.callbacks import TrainingMetricsCallback
from videotuna.utils.common_utils import instantiate_from_config


def test_training_metrics_callback_writes_metrics_json():
    callback = TrainingMetricsCallback()
    trainer = mock.MagicMock()
    trainer.current_epoch = 0
    trainer.global_rank = 0
    trainer.global_step = 42
    pl_module = mock.MagicMock()
    pl_module.logdir = None
    trainer.logger = mock.MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        callback.save_dir = tmpdir
        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_epoch_end(trainer, pl_module)

        metrics_path = os.path.join(tmpdir, "metrics.json")
        assert os.path.isfile(metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert len(data["epochs"]) == 1
        assert "epoch_time_s" in data["epochs"][0]
        assert "peak_vram_gb" in data["epochs"][0]
        assert "steps" in data
        assert data["steps"] == []
        trainer.logger.log_metrics.assert_called_once()
        logged = trainer.logger.log_metrics.call_args[0][0]
        assert "epoch_time_s" in logged
        assert "peak_vram_gb" in logged


def test_training_metrics_callback_ema_correctness():
    callback = TrainingMetricsCallback(ema_decay=0.5)
    trainer = mock.MagicMock()
    trainer.global_rank = 0
    trainer.global_step = 0
    pl_module = mock.MagicMock()
    trainer.optimizers = [mock.MagicMock()]
    trainer.optimizers[0].param_groups = [{"lr": 1e-4}]
    trainer.logger = mock.MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        callback.save_dir = tmpdir

        callback._batch_start_time = time.time()
        outputs = torch.tensor(1.0)
        callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=0)
        assert callback._loss_ema is not None
        assert callback._loss_ema == pytest.approx(1.0)

        trainer.global_step = 1
        callback._batch_start_time = time.time()
        outputs = torch.tensor(2.0)
        callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=1)
        expected = 0.5 * 1.0 + 0.5 * 2.0
        assert callback._loss_ema == pytest.approx(expected)


def test_training_metrics_callback_divergence_warning():
    callback = TrainingMetricsCallback(ema_decay=0.5)
    trainer = mock.MagicMock()
    trainer.global_rank = 0
    trainer.global_step = 0
    pl_module = mock.MagicMock()
    trainer.optimizers = [mock.MagicMock()]
    trainer.optimizers[0].param_groups = [{"lr": 1e-4}]
    trainer.logger = mock.MagicMock()

    mock_warning = mock.MagicMock()
    with mock.patch("videotuna.utils.callbacks.mainlogger.warning", mock_warning):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback.save_dir = tmpdir

            callback._batch_start_time = time.time()
            outputs = torch.tensor(1.0)
            callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=0)
            assert mock_warning.call_count == 0

            trainer.global_step = 1
            callback._batch_start_time = time.time()
            outputs = torch.tensor(10.0)
            callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=1)

            mock_warning.assert_called_once()
            msg = mock_warning.call_args[0][0]
            assert "divergence" in msg


def test_training_metrics_callback_step_json_format():
    callback = TrainingMetricsCallback(step_log_interval=1)
    trainer = mock.MagicMock()
    trainer.global_rank = 0
    trainer.global_step = 0
    pl_module = mock.MagicMock()
    pl_module.logdir = None
    trainer.optimizers = [mock.MagicMock()]
    trainer.optimizers[0].param_groups = [{"lr": 2e-4}]
    trainer.logger = mock.MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        callback.save_dir = tmpdir
        callback.on_train_epoch_start(trainer, pl_module)

        callback._batch_start_time = time.time() - 1.0
        outputs = torch.tensor(0.5)
        callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=0)

        callback._batch_start_time = time.time() - 0.5
        trainer.global_step = 1
        outputs = torch.tensor(0.8)
        callback.on_train_batch_end(trainer, pl_module, outputs, None, batch_idx=1)

        metrics_path = os.path.join(tmpdir, "metrics.json")
        with open(metrics_path) as f:
            data = json.load(f)

        assert "steps" in data
        assert "epochs" in data
        assert len(data["steps"]) == 2

        step_keys = {
            "step",
            "loss",
            "loss_ema",
            "lr",
            "steps_per_second",
            "peak_vram_gb",
        }
        for entry in data["steps"]:
            assert set(entry.keys()) == step_keys, f"missing keys in {entry}"
            assert isinstance(entry["step"], int)
            assert isinstance(entry["loss"], float)
            assert isinstance(entry["loss_ema"], float)
            assert isinstance(entry["lr"], float)
            assert isinstance(entry["steps_per_second"], float)
            assert isinstance(entry["peak_vram_gb"], float)


def test_training_metrics_callback_params_from_config():
    cfg = {
        "target": "videotuna.utils.callbacks.TrainingMetricsCallback",
        "params": {
            "ema_decay": 0.5,
            "step_log_interval": 5,
            "save_dir": "/tmp/test_metrics",
        },
    }
    callback = instantiate_from_config(cfg)
    assert callback.ema_decay == 0.5
    assert callback.step_log_interval == 5
    assert callback.save_dir == "/tmp/test_metrics"


def test_training_metrics_callback_default_parameters():
    callback = TrainingMetricsCallback()
    assert callback.ema_decay == 0.99
    assert callback.step_log_interval == 10
    assert callback.save_dir is None
