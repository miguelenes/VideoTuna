import warnings

import pytest

try:
    from sentry_sdk.hub import SentryHubDeprecationWarning
except ImportError:
    SentryHubDeprecationWarning = DeprecationWarning  # type: ignore[misc,assignment]


@pytest.fixture(autouse=True)
def _suppress_third_party_import_warnings():
    """Optional third-party deps emit noisy warnings on import-only smoke tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SentryHubDeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message="Please install the latest tensornvme",
            category=UserWarning,
            module=r"colossalai\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Please install apex from source",
            category=UserWarning,
            module=r"colossalai\..*",
        )
        yield


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: tests that require a GPU (skipped when torch.cuda.is_available() is False)",
    )
    config.addinivalue_line(
        "markers",
        "rocm: tests that require an AMD ROCm GPU",
    )
    config.addinivalue_line(
        "markers",
        "cpu_smoke: slow CPU integration tests (optional nightly)",
    )


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        from torch import version as torch_version
    except (ImportError, OSError, ValueError):
        return

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    is_rocm = (
        torch.cuda.is_available() and getattr(torch_version, "hip", None) is not None
    )
    if not is_rocm:
        skip_rocm = pytest.mark.skip(reason="ROCm not available")
        for item in items:
            if "rocm" in item.keywords:
                item.add_marker(skip_rocm)
