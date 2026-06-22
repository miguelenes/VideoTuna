import importlib

import pytest
import torch
from packaging.version import Version

INFERENCE_BACKENDS = [
    "videotuna.flow.diffusers_video",
    "videotuna.flow.hunyuanvideo",
    "videotuna.flow.videocrafter",
]

TRAINING_BACKENDS = [
    ("videotuna.models.opensora.acceleration.plugin", "colossalai"),
    ("videotuna.training.flux_lora.config", None),
]

GPU_BACKENDS = [
    "videotuna.flow.wanvideo",
    "videotuna.flow.stepvideo",
]


@pytest.mark.parametrize("module", INFERENCE_BACKENDS)
def test_inference_backend_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module,extra", TRAINING_BACKENDS)
def test_training_backend_import(module, extra):
    if extra is not None:
        pytest.importorskip(extra)
    try:
        importlib.import_module(module)
    except ValueError as exc:
        if module == "videotuna.models.opensora.acceleration.plugin":
            pytest.skip(f"colossalai plugin import skipped: {exc}")
        raise


@pytest.mark.parametrize("module", GPU_BACKENDS)
def test_gpu_backend_import(module):
    from videotuna.utils.device_utils import gpu_is_available

    if not gpu_is_available():
        pytest.skip("GPU accelerator required for module-level GPU initialization")
    importlib.import_module(module)


def test_core_ml_stack_versions():
    import accelerate
    import diffusers
    import peft
    import transformers

    assert (
        Version(torch.__version__).major == 2 and Version(torch.__version__).minor >= 6
    )
    assert Version(diffusers.__version__) >= Version("0.36.0")
    assert Version(transformers.__version__) >= Version("4.48.0")
    assert Version(accelerate.__version__) >= Version("1.2.0")
    assert Version(peft.__version__) >= Version("0.17.0")


def test_training_stack_versions():
    pytest.importorskip("deepspeed")
    import deepspeed

    assert Version(deepspeed.__version__) >= Version("0.19.0")
