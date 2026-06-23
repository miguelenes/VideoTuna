"""Tests for training DataLoader configuration."""

import os
import unittest.mock as mock

import pytest
import torch
from torch.utils.data import Dataset

from videotuna.data.lightningdata import DataModuleFromConfig


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"video": torch.zeros(3, 2, 8, 8), "caption": f"item-{idx}"}


@pytest.fixture
def tiny_datamodule_config():
    return {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "train": {
            "target": "tests.test_training_dataloader._TinyDataset",
            "params": {},
        },
    }


def test_datamodule_dataloader_kwargs(tiny_datamodule_config):
    dm = DataModuleFromConfig(**tiny_datamodule_config)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.batch_size == 2
    assert loader.pin_memory is True
    assert loader.num_workers == 0


def test_datamodule_collate_default_batch(tiny_datamodule_config):
    dm = DataModuleFromConfig(**tiny_datamodule_config)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["video"].shape[0] == 2
    assert len(batch["caption"]) == 2


def test_default_num_workers_not_batch_scaled():
    dm = DataModuleFromConfig(
        batch_size=8,
        train={
            "target": "tests.test_training_dataloader._TinyDataset",
            "params": {},
        },
    )
    assert dm.num_workers == 4


# ---------------------------------------------------------------------------
# DatasetFromCSV.__getitem__ error-handling tests
# ---------------------------------------------------------------------------


def _make_csv_dataset(tmp_path, size=6, max_retries=10):
    """Build a minimal DatasetFromCSV pointing at non-existent mp4 paths."""
    import pandas as pd

    from videotuna.data.datasets import DatasetFromCSV
    from videotuna.data.transforms import get_transforms_video

    csv_file = tmp_path / "data.csv"
    rows = [
        {"path": str(tmp_path / f"v{i}.mp4"), "caption": f"cap{i}"} for i in range(size)
    ]
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    transform = {"video": get_transforms_video()}
    ds = DatasetFromCSV(
        str(csv_file),
        transform=transform,
        max_retries=max_retries,
    )
    return ds


def test_skip_count_increments_on_failures(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=6, max_retries=3)

    def always_fail(index):
        raise ValueError("injected failure")

    ds.getitem = always_fail

    with pytest.raises(RuntimeError, match="Too many bad data"):
        ds[0]

    assert ds._skip_count == 3


def test_retry_limit_default(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=6)
    assert ds.max_retries == 10


def test_retry_limit_custom_exhausted(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=6, max_retries=2)

    def always_fail(index):
        raise ValueError("injected failure")

    ds.getitem = always_fail

    with pytest.raises(RuntimeError, match="Too many bad data after 2 retries"):
        ds[0]

    assert ds._skip_count == 2


def test_access_count_increments(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=6, max_retries=2)

    def always_fail(index):
        raise ValueError("injected failure")

    ds.getitem = always_fail

    assert ds._access_count == 0
    with pytest.raises(RuntimeError):
        ds[0]
    assert ds._access_count == 1
    with pytest.raises(RuntimeError):
        ds[1]
    assert ds._access_count == 2


def test_final_exception_includes_offending_sample_context(tmp_path):
    """Final RuntimeError must contain the offending sample index and path."""
    ds = _make_csv_dataset(tmp_path, size=3, max_retries=2)

    def always_fail(index):
        raise ValueError("injected failure")

    ds.getitem = always_fail
    ds.safe_data_list = set()

    with pytest.raises(RuntimeError) as exc_info:
        ds[0]

    msg = str(exc_info.value)
    assert (
        "Last offending sample:" in msg
    ), f"Expected offending sample context in: {msg}"
    assert "index=" in msg, f"Expected index in exception: {msg}"
    assert ".mp4" in msg, f"Expected path fragment in exception: {msg}"


def test_off_by_one_fallback_index_upper_bound(tmp_path):
    """Fallback random.randint must use len(ds)-1, not len(ds)."""
    ds = _make_csv_dataset(tmp_path, size=6, max_retries=3)

    def always_fail(index):
        raise ValueError("injected failure")

    ds.getitem = always_fail
    ds.safe_data_list = set()

    randint_calls = []

    original_randint = __import__("random").randint

    def spy_randint(a, b):
        randint_calls.append((a, b))
        return original_randint(a, b)

    with mock.patch("videotuna.data.datasets.random.randint", side_effect=spy_randint):
        with pytest.raises(RuntimeError):
            ds[0]

    assert len(randint_calls) > 0, "randint should have been called for fallback"
    for a, b in randint_calls:
        assert b == len(ds) - 1, f"Expected upper bound {len(ds)-1}, got {b}"


def test_validate_dataset_detects_missing_files(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=4)
    issues = ds.validate_dataset(raise_on_error=False)
    assert len(issues) == 4
    for issue in issues:
        assert "index" in issue
        assert "path" in issue
        assert "error" in issue


def test_validate_dataset_raises_by_default(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=4)
    with pytest.raises(RuntimeError, match="Dataset validation failed"):
        ds.validate_dataset()


def test_validate_dataset_passes_good_entries(tmp_path):
    ds = _make_csv_dataset(tmp_path, size=4)

    with (
        mock.patch("videotuna.data.datasets.os.path.exists", return_value=True),
        mock.patch(
            "videotuna.data.datasets.get_video_frame_count",
            return_value=1000,
        ),
    ):
        issues = ds.validate_dataset(raise_on_error=False)

    assert issues == []


def test_datamodule_setup_validate_on_setup_raises(tmp_path):
    """validate_on_setup=True should propagate RuntimeError from validate_dataset."""
    import pandas as pd

    from videotuna.data.datasets import DatasetFromCSV

    csv_file = tmp_path / "data.csv"
    pd.DataFrame([{"path": str(tmp_path / "missing.mp4"), "caption": "cap"}]).to_csv(
        csv_file, index=False
    )

    dm = DataModuleFromConfig(
        batch_size=1,
        num_workers=0,
        validate_on_setup=True,
        train={
            "target": "videotuna.data.datasets.DatasetFromCSV",
            "params": {
                "csv_path": str(csv_file),
                "max_retries": 1,
            },
        },
    )
    with pytest.raises(RuntimeError, match="Dataset validation failed"):
        dm.setup()


def test_datamodule_setup_validate_off_by_default(tmp_path):
    """validate_on_setup=False (default) must not call validate_dataset."""
    import pandas as pd

    csv_file = tmp_path / "data.csv"
    pd.DataFrame([{"path": str(tmp_path / "missing.mp4"), "caption": "cap"}]).to_csv(
        csv_file, index=False
    )

    dm = DataModuleFromConfig(
        batch_size=1,
        num_workers=0,
        train={
            "target": "videotuna.data.datasets.DatasetFromCSV",
            "params": {
                "csv_path": str(csv_file),
                "max_retries": 1,
            },
        },
    )
    dm.setup()
    assert "train" in dm.datasets
