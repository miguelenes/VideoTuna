"""Tests for scripts/verify_hf_download.py."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

verify_hf_download = importlib.import_module("scripts.verify_hf_download")

_HF_XET_OK = (True, "1.5.1")
_FAST_KNOB_WARNING = (
    "VIDEOTUNA_FAST_HF_DOWNLOAD=1 is set but HF_XET_HIGH_PERFORMANCE is not"
)


@pytest.fixture(autouse=True)
def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VIDEOTUNA_FAST_HF_DOWNLOAD",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_OFFLINE",
        "HF_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_main_success_with_metadata_smoke(capsys):
    mock_info = SimpleNamespace(sha="abc123def4567890")
    with patch.object(verify_hf_download, "_check_import", return_value=_HF_XET_OK):
        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api_cls.return_value.model_info.return_value = mock_info
            rc = verify_hf_download.main([])

    assert rc == 0
    out = capsys.readouterr()
    assert "hf_xet: OK" in out.out
    assert "Hub metadata OK: gpt2" in out.out
    assert "HF download verification OK" in out.out


def test_main_warns_when_fast_knob_without_xet_var(capsys):
    with patch.dict("os.environ", {"VIDEOTUNA_FAST_HF_DOWNLOAD": "1"}, clear=False):
        with patch.object(verify_hf_download, "_check_import", return_value=_HF_XET_OK):
            with patch("huggingface_hub.HfApi") as mock_api_cls:
                mock_api_cls.return_value.model_info.return_value = SimpleNamespace(
                    sha="abc123def4567890"
                )
                rc = verify_hf_download.main([])

    assert rc == 0
    err = capsys.readouterr().err
    assert _FAST_KNOB_WARNING in err


def test_main_warns_on_deprecated_hf_transfer(capsys):
    with patch.dict("os.environ", {"HF_HUB_ENABLE_HF_TRANSFER": "1"}, clear=False):
        with patch.object(verify_hf_download, "_check_import", return_value=_HF_XET_OK):
            with patch("huggingface_hub.HfApi") as mock_api_cls:
                mock_api_cls.return_value.model_info.return_value = SimpleNamespace(
                    sha="abc123def4567890"
                )
                rc = verify_hf_download.main([])

    assert rc == 0
    err = capsys.readouterr().err
    assert "HF_HUB_ENABLE_HF_TRANSFER is deprecated" in err


def test_main_skips_metadata_when_offline(capsys):
    with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}, clear=False):
        with patch.object(verify_hf_download, "_check_import", return_value=_HF_XET_OK):
            with patch("huggingface_hub.HfApi") as mock_api_cls:
                rc = verify_hf_download.main([])

    assert rc == 0
    out = capsys.readouterr().out
    assert "HF_HUB_OFFLINE=1 — skipping hub metadata smoke" in out
    mock_api_cls.assert_not_called()


def test_main_fails_on_metadata_error(capsys):
    with patch.object(verify_hf_download, "_check_import", return_value=_HF_XET_OK):
        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api_cls.return_value.model_info.side_effect = RuntimeError(
                "network down"
            )
            rc = verify_hf_download.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Hub metadata smoke failed: network down" in err
