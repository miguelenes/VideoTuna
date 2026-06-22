"""Mocked Wan training step smoke test."""

from unittest import mock

import pytest
import torch

from videotuna.flow.wanvideo import WanVideoModelFlow


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Wan training step mock"
)
def test_wan_training_step_mocked():
    flow = mock.MagicMock(spec=WanVideoModelFlow)
    flow.first_stage_key = "video"
    flow.cond_stage_key = "caption"
    flow.wan_t2v = mock.MagicMock()
    expected_loss = torch.tensor(0.5, requires_grad=True)
    flow.wan_t2v.training_step.return_value = expected_loss
    flow.task = "t2v-14B"

    batch = {"video": torch.randn(1, 3, 4, 32, 32), "caption": ["test"]}
    loss = WanVideoModelFlow.training_step(flow, batch, 0)

    flow.wan_t2v.training_step.assert_called_once_with(batch, 0, "video", "caption")
    assert loss is expected_loss


def test_wan_training_step_delegates_to_i2v():
    flow = mock.MagicMock(spec=WanVideoModelFlow)
    flow.first_stage_key = "video"
    flow.cond_stage_key = "caption"
    flow.wan_i2v = mock.MagicMock()
    flow.wan_i2v.training_step.return_value = torch.tensor(1.0)
    flow.task = "i2v-14B"

    batch = {"video": torch.randn(1, 3, 4, 32, 32), "caption": ["test"]}
    loss = WanVideoModelFlow.training_step(flow, batch, 0)

    flow.wan_i2v.training_step.assert_called_once()
    assert float(loss) == 1.0
