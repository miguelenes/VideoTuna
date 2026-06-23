"""Redirect deprecated Diffusers inference scripts to inference_new.py."""

from __future__ import annotations

import subprocess
import sys
import warnings


def run_diffusers_inference(config: str, extra_args: list[str] | None = None) -> int:
    message = (
        f"This script is deprecated. Use:\n"
        f"  poetry run inference-run --config {config}\n"
        f"or the matching preset command (e.g. validate-domain-t2v)."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    cmd = [
        sys.executable,
        "scripts/inference_new.py",
        "--config",
        config,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, check=False)
    return result.returncode
