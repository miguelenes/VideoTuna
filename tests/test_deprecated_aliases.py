"""Tests for deprecated CLI alias warnings."""

from __future__ import annotations

import subprocess
import sys
import warnings
from unittest.mock import patch

import pytest

from videotuna.utils.deprecation import REMOVAL_VERSION


def _assert_no_deprecation_warnings(fn) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations


@pytest.mark.parametrize(
    ("legacy_fn", "canonical"),
    [
        ("train_flux_lora", "train-domain-t2i"),
        ("train_wan2_1_t2v_lora", "train-domain-t2v"),
        ("train_wan2_1_i2v_lora", "train-domain-i2v"),
    ],
)
def test_legacy_training_aliases_warn(legacy_fn, canonical):
    import scripts

    fn = getattr(scripts, legacy_fn)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        with pytest.warns(DeprecationWarning) as record:
            with pytest.raises(SystemExit) as exc:
                fn()
        assert exc.value.code == 0

    assert len(record) == 1
    message = str(record[0].message)
    assert canonical in message
    assert REMOVAL_VERSION in message
    assert mock_run.called


@pytest.mark.parametrize(
    "canonical_fn",
    ["train_domain_t2i", "train_domain_t2v", "train_domain_i2v"],
)
def test_canonical_training_aliases_do_not_warn(canonical_fn):
    import scripts

    fn = getattr(scripts, canonical_fn)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        def run():
            with pytest.raises(SystemExit) as exc:
                fn()
            assert exc.value.code == 0

        _assert_no_deprecation_warnings(run)

    assert mock_run.called


def test_inference_flux_lora_entry_warns():
    from videotuna.cli.inference_app import inference_flux_lora_entry

    with patch(
        "videotuna.cli.inference_app.inference_domain_t2i_entry",
        side_effect=SystemExit(0),
    ):
        with pytest.warns(DeprecationWarning) as record:
            with pytest.raises(SystemExit):
                inference_flux_lora_entry()

    message = str(record[0].message)
    assert "inference-domain-t2i" in message
    assert REMOVAL_VERSION in message


def test_inference_domain_t2i_entry_does_not_warn():
    from videotuna.cli.inference_app import inference_domain_t2i_entry

    with patch(
        "cyclopts.App.__call__",
        side_effect=SystemExit(0),
    ):

        def run():
            with pytest.raises(SystemExit):
                inference_domain_t2i_entry()

        _assert_no_deprecation_warnings(run)


def test_inference_new_main_warns():
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "inference_new.py"
    with patch(
        "videotuna.cli.inference_app.generic_inference_entry",
        side_effect=SystemExit(0),
    ):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )

    combined = result.stdout + result.stderr
    assert "DeprecationWarning" in combined or "deprecated" in combined.lower()
    assert "inference-run" in combined
    assert REMOVAL_VERSION in combined


def test_generic_inference_entry_does_not_warn():
    from videotuna.cli.inference_app import generic_inference_entry

    with patch(
        "cyclopts.App.__call__",
        side_effect=SystemExit(0),
    ):

        def run():
            with pytest.raises(SystemExit):
                generic_inference_entry()

        _assert_no_deprecation_warnings(run)
