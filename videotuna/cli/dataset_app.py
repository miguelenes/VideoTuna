from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

from cyclopts import App, Parameter

from videotuna.data.validation.runner import VALID_PHASES, validate_datasets
from videotuna.utils.cli_console import install_pretty_tracebacks
from videotuna.utils.logging_config import bound_logger, configure_logging

app = App(
    name="validate-datasets",
    help="""Validate dataset structure and content before training.

Scans Flux T2I directories and Wan T2V/I2V CSV manifests for common
issues: missing files, empty or missing trigger-token captions, wrong
image/video dimensions, insufficient frame counts, and orphan files.

When --normalize is passed, re-encodes Wan video clips to the expected
480x832 / 81-frame format using ffmpeg.

Exit codes: 0 = all pass, 2 = validation errors found.
""",
)

logger = bound_logger(phase="data_validation", flow="data")


@Parameter(name="*")
@dataclass
class DatasetValidationOptions:
    phase: Annotated[
        tuple[str, ...],
        Parameter(name="phase", help=f"Phase(s): all, {', '.join(VALID_PHASES)}"),
    ] = ("all",)
    data_root: Annotated[
        tuple[str, ...],
        Parameter(name="data-root", help="Override data roots"),
    ] = ()
    trigger_token: Annotated[
        str,
        Parameter(name="trigger-token", help="Trigger token for captions"),
    ] = "sks_style"
    strict: Annotated[
        bool,
        Parameter(name="strict", help="Treat dimension mismatches as errors"),
    ] = False
    normalize: Annotated[
        bool,
        Parameter(name="normalize", help="Re-encode Wan videos via ffmpeg"),
    ] = False
    output_dir: Annotated[
        Path,
        Parameter(name="output-dir", help="Directory for outputs and reports"),
    ] = Path("results/data_validation")
    report_path: Annotated[
        Optional[Path],
        Parameter(name="report-path", help="Path for JSON report"),
    ] = None
    wan_height: Annotated[
        int,
        Parameter(name="wan-height", help="Expected Wan height"),
    ] = 480
    wan_width: Annotated[
        int,
        Parameter(name="wan-width", help="Expected Wan width"),
    ] = 832
    wan_frames: Annotated[
        int,
        Parameter(name="wan-frames", help="Expected Wan frame count"),
    ] = 81
    wan_fps: Annotated[int, Parameter(name="wan-fps", help="Target FPS")] = 16


@app.default()
def _validate_datasets_command(opts: DatasetValidationOptions) -> None:
    configure_logging()
    exit_code = validate_datasets(
        phases=opts.phase,
        data_roots=opts.data_root,
        trigger_token=opts.trigger_token,
        strict=opts.strict,
        normalize=opts.normalize,
        output_dir=opts.output_dir,
        report_path=opts.report_path,
        wan_expected_height=opts.wan_height,
        wan_expected_width=opts.wan_width,
        wan_expected_frames=opts.wan_frames,
        wan_reencode_fps=opts.wan_fps,
    )
    raise SystemExit(exit_code)


def validate_datasets_entry() -> None:
    install_pretty_tracebacks()
    raise SystemExit(app(sys.argv[1:]))
