import importlib

import pytest
import torch
from packaging.version import Version

BACKENDS = [
    "videotuna.flow.diffusers_video",
    "videotuna.flow.hunyuanvideo",
    "videotuna.flow.videocrafter",
    "videotuna.models.opensora.acceleration.plugin",
    "videotuna.third_party.flux.training.model",
    "videotuna.models.cogvideo_sat.arguments",
]

GPU_BACKENDS = [
    "videotuna.flow.wanvideo",
    "videotuna.flow.stepvideo",
]


@pytest.mark.parametrize("module", BACKENDS)
def test_backend_import(module):
    try:
        importlib.import_module(module)
    except ValueError as exc:
        if module == "videotuna.models.opensora.acceleration.plugin":
            pytest.skip(f"colossalai plugin import skipped: {exc}")
        raise


@pytest.mark.parametrize("module", GPU_BACKENDS)
def test_gpu_backend_import(module):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for module-level GPU initialization")
    importlib.import_module(module)


def test_core_ml_stack_versions():
    import accelerate
    import deepspeed
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
    assert Version(deepspeed.__version__) >= Version("0.19.0")
