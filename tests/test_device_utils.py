"""Tests for unified compute backend detection."""

from unittest import mock

import pytest
import torch

from videotuna.utils import device_utils


def test_gpu_is_available_alias():
    assert device_utils.cuda_is_available() == device_utils.gpu_is_available()


def test_resolve_inference_device_cpu_when_no_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        assert device_utils.resolve_inference_device() == torch.device("cpu")


def test_resolve_inference_device_cuda_when_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        assert device_utils.resolve_inference_device() == torch.device("cuda")


def test_resolve_inference_device_rejects_cuda_without_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        with pytest.raises(RuntimeError, match="no GPU accelerator"):
            device_utils.resolve_inference_device("cuda")


def test_detect_compute_backend_cpu():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=False):
        assert device_utils.detect_compute_backend() == "cpu"


def test_detect_compute_backend_cuda():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=True):
        with mock.patch.object(device_utils, "_torch_hip_version", return_value=None):
            assert device_utils.detect_compute_backend() == "cuda"


def test_detect_compute_backend_rocm():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=True):
        with mock.patch.object(device_utils, "_torch_hip_version", return_value="6.2.4"):
            assert device_utils.detect_compute_backend() == "rocm"


def test_describe_compute_environment_rocm():
    with mock.patch.object(device_utils, "_detect_compute_backend_raw", return_value="rocm"):
        with mock.patch.object(
            device_utils.torch.cuda, "get_device_name", return_value="gfx1100"
        ):
            with mock.patch.object(device_utils, "_torch_hip_version", return_value="6.2.4"):
                with mock.patch.object(device_utils.torch, "__version__", "2.6.0"):
                    desc = device_utils.describe_compute_environment()
    assert "ROCm available" in desc
    assert "gfx1100" in desc
    assert "HIP 6.2.4" in desc


def test_require_accelerator_for_flow_raises_without_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        with pytest.raises(RuntimeError, match="GPU accelerator"):
            device_utils.require_accelerator_for_flow(
                "videotuna.flow.wanvideo.WanVideoModelFlow"
            )


def test_require_accelerator_for_flow_stepvideo_blocked_on_rocm():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(device_utils, "detect_compute_backend", return_value="rocm"):
            with pytest.raises(RuntimeError, match="StepVideo inference is not supported"):
                device_utils.require_accelerator_for_flow(
                    "videotuna.flow.stepvideo.StepVideoModelFlow"
                )


def test_require_accelerator_for_flow_allow_cpu():
    device_utils.require_accelerator_for_flow(
        "videotuna.flow.wanvideo.WanVideoModelFlow",
        allow_cpu=True,
    )


def test_compute_backend_env_rocm_mismatch():
    with mock.patch.dict("os.environ", {"VIDEOTUNA_COMPUTE_BACKEND": "rocm"}):
        with mock.patch.object(device_utils, "_torch_hip_version", return_value=None):
            with pytest.raises(RuntimeError, match="not built with HIP"):
                device_utils.detect_compute_backend()


def test_require_xfuser_sequence_parallel_on_rocm():
    with mock.patch.object(device_utils, "detect_compute_backend", return_value="rocm"):
        with pytest.raises(RuntimeError, match="xfuser requires NVIDIA CUDA"):
            device_utils.require_xfuser_sequence_parallel("TestFlow")


def test_accelerator_helpers_noop_on_cpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        assert device_utils.accelerator_device_string() == "cpu"
        device_utils.empty_accelerator_cache()
        device_utils.synchronize_accelerator()
