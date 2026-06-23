#!/usr/bin/env python3
"""Verify Hugging Face hub connectivity and hf-xet download configuration."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

_TRUTHY = frozenset({"1", "on", "yes", "true"})


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _env_display(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return "(unset)"
    return value


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        return True, str(version)
    except ImportError:
        return False, "not installed"


def _print_env_diagnostics() -> None:
    print("HF download environment:")
    for name in (
        "VIDEOTUNA_FAST_HF_DOWNLOAD",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_OFFLINE",
        "HF_HOME",
    ):
        print(f"  {name}: {_env_display(name)}")


def _collect_config_warnings() -> list[str]:
    warnings: list[str] = []
    if _env_truthy("VIDEOTUNA_FAST_HF_DOWNLOAD") and not _env_truthy(
        "HF_XET_HIGH_PERFORMANCE"
    ):
        warnings.append(
            "VIDEOTUNA_FAST_HF_DOWNLOAD=1 is set but HF_XET_HIGH_PERFORMANCE is not; "
            "cloud bootstrap maps this automatically — locally set "
            "HF_XET_HIGH_PERFORMANCE=1 in .env (see .env.example)."
        )
    if _env_truthy("HF_HUB_ENABLE_HF_TRANSFER"):
        warnings.append(
            "HF_HUB_ENABLE_HF_TRANSFER is deprecated; "
            "use HF_XET_HIGH_PERFORMANCE=1 instead."
        )
    return warnings


def _run_metadata_smoke() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info("gpt2")
    print(f"Hub metadata OK: gpt2 (sha={info.sha[:12]}...)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Hugging Face hub connectivity and hf-xet download config."
    )
    parser.parse_args(argv)

    errors: list[str] = []
    _print_env_diagnostics()

    hf_xet_ok, hf_xet_detail = _check_import("hf_xet")
    status = "OK" if hf_xet_ok else "MISSING"
    print(f"  hf_xet: {status} ({hf_xet_detail})")
    if not hf_xet_ok:
        print(
            "WARNING: hf-xet not importable; hub downloads use fallback path",
            file=sys.stderr,
        )

    for warning in _collect_config_warnings():
        print(f"WARNING: {warning}", file=sys.stderr)

    if _env_truthy("HF_HUB_OFFLINE"):
        print("HF_HUB_OFFLINE=1 — skipping hub metadata smoke")
    else:
        try:
            _run_metadata_smoke()
        except Exception as exc:
            errors.append(f"Hub metadata smoke failed: {exc}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("HF download verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
