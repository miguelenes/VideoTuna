import warnings

import pytest


@pytest.fixture(autouse=True)
def _suppress_third_party_import_warnings():
    """Optional third-party deps emit noisy warnings on import-only smoke tests."""
    with warnings.catch_warnings():
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
