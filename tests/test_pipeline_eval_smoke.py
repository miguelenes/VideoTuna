"""Smoke test: pipeline_eval fixture through DatasetFromCSV + DataModuleFromConfig.

Fixture layout::

    tests/fixtures/pipeline_eval/
        metadata.csv          # path,caption (relative to fixture dir)
        videos/
            cat_001.mp4       # 64x64, 8 frames, 24 fps
            cat_002.mp4       # 64x64, 8 frames, 24 fps

Asserts:
  - batch contains exactly 2 captions
  - video tensor shape is (2, 3, 8, 64, 64)  — matching the CPU smoke convention
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pipeline_eval"
FIXTURE_CSV = FIXTURE_DIR / "metadata.csv"


@pytest.mark.cpu_smoke
def test_pipeline_eval_fixture_batch_shape():
    from videotuna.data.lightningdata import DataModuleFromConfig

    dm = DataModuleFromConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        train={
            "target": "videotuna.data.datasets.DatasetFromCSV",
            "params": {
                "csv_path": str(FIXTURE_CSV),
                "data_root": str(FIXTURE_DIR),
                "height": 64,
                "width": 64,
                "num_frames": 8,
                "frame_interval": 1,
                "train": True,
                "split_val": False,
            },
        },
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert (
        len(batch["caption"]) == 2
    ), f"Expected 2 captions, got {len(batch['caption'])}"
    assert batch["video"].shape == (
        2,
        3,
        8,
        64,
        64,
    ), f"Expected video shape (2, 3, 8, 64, 64), got {batch['video'].shape}"
