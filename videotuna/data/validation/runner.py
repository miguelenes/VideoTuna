from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from videotuna.data.validation.checks import ffmpeg_available, reencode_clip
from videotuna.data.validation.flux_validator import FluxDatasetValidator
from videotuna.data.validation.report import PhaseReport, ValidationReport
from videotuna.data.validation.wan_validator import WanDatasetValidator
from videotuna.utils.logging_config import bound_logger

VALID_PHASES = ("flux-t2i", "wan-t2v", "wan-i2v")

logger = bound_logger(phase="data_validation", flow="data")


def resolve_phase_configs(
    phases: tuple[str, ...],
    data_roots: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Resolve configuration for each requested phase.

    Uses auto-discovery from domain config files unless ``data_roots``
    are explicitly provided.  When a single ``data_roots`` entry is
    given it overrides the root for *all* selected phases; otherwise
    ``data_roots`` must match the number of phases.
    """
    expanded = _expand_phases(phases)
    if data_roots and len(data_roots) == 1 and len(expanded) > 1:
        roots: dict[str, str] = {p: data_roots[0] for p in expanded}
    elif data_roots and len(data_roots) == len(expanded):
        roots = dict(zip(expanded, data_roots))
    elif data_roots:
        logger.error(
            "Got {} --data-root value(s) for {} phase(s)",
            len(data_roots),
            len(expanded),
        )
        raise SystemExit(2)
    else:
        roots = {}

    configs: dict[str, dict[str, Any]] = {}
    for phase in expanded:
        if phase == "flux-t2i":
            root = roots.get(phase) or _discover_flux_root()
            configs[phase] = {
                "type": "flux",
                "data_root": Path(root) if root else None,
            }
        elif phase == "wan-t2v":
            root = roots.get(phase) or _discover_wan_root("t2v")
            configs[phase] = {
                "type": "wan",
                "mode": "t2v",
                "data_root": Path(root) if isinstance(root, (str, Path)) else root,
            }
        elif phase == "wan-i2v":
            root = roots.get(phase) or _discover_wan_root("i2v")
            configs[phase] = {
                "type": "wan",
                "mode": "i2v",
                "data_root": Path(root) if isinstance(root, (str, Path)) else root,
            }
    return configs


def _expand_phases(phases: tuple[str, ...]) -> list[str]:
    if not phases:
        return []
    if "all" in phases:
        return list(VALID_PHASES)
    result: list[str] = []
    for p in phases:
        if p == "all":
            result.extend(VALID_PHASES)
        elif p in VALID_PHASES:
            result.append(p)
        else:
            logger.error("Unknown phase: {}. Valid: {}", p, ", ".join(VALID_PHASES))
            raise SystemExit(2)
    return result


def _discover_flux_root() -> Optional[str]:
    try:
        path = Path("configs/domain/flux_t2i_data.json")
        if path.is_file():
            data = json.loads(path.read_text())
            if isinstance(data, list):
                for backend in data:
                    if "instance_data_dir" in backend:
                        return backend["instance_data_dir"]
            elif isinstance(data, dict):
                return data.get("instance_data_dir")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    logger.warning("Could not auto-discover Flux data root from configs/domain/")
    return None


def _discover_wan_root(mode: str) -> Optional[Path]:
    fname = f"wan_{'i2v' if mode == 'i2v' else 't2v'}_lora.yaml"
    path = Path(f"configs/domain/{fname}")
    if not path.is_file():
        logger.warning("Could not find {}", path)
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        csv_path = cfg.train.params.csv_path
        resolved = Path(csv_path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved.parent if resolved.name == "metadata.csv" else resolved
    except Exception as exc:
        logger.warning("Failed to read {}: {}", path, exc)
        return None


def validate_datasets(
    phases: tuple[str, ...] = ("all",),
    data_roots: tuple[str, ...] = (),
    *,
    trigger_token: str = "sks_style",
    strict: bool = False,
    normalize: bool = False,
    output_dir: Path = Path("results/data_validation"),
    report_path: Optional[Path] = None,
    wan_expected_height: int = 480,
    wan_expected_width: int = 832,
    wan_expected_frames: int = 81,
    wan_reencode_fps: int = 16,
) -> int:
    """Run dataset validation on the requested phases.

    Returns exit code ``0`` on success (no errors), ``2`` if any
    errors were found.
    """
    if normalize and not ffmpeg_available():
        logger.error(
            "--normalize requires ffmpeg but it is not installed or not on PATH."
        )
        return 2

    phase_configs = resolve_phase_configs(phases, data_roots)
    if not phase_configs:
        logger.warning("No phases to validate. Use --phase to select data phases.")
        return 0

    report = ValidationReport()
    for phase, cfg in phase_configs.items():
        logger.info("Validating phase: {} ...", phase)

        if cfg["type"] == "flux":
            data_root = cfg.get("data_root")
            if data_root is None:
                logger.warning("Skipping flux-t2i: no data root discovered.")
                continue
            validator = FluxDatasetValidator(
                data_dir=Path(data_root),
                trigger_token=trigger_token,
            )
            phase_report = validator.validate()

        elif cfg["type"] == "wan":
            root = cfg.get("data_root")
            if root is None:
                logger.warning("Skipping {}: no data root discovered.", phase)
                continue
            csv_path = root / "metadata.csv" if root.is_dir() else root
            data_root_dir = root if root.is_dir() else root.parent

            validator = WanDatasetValidator(
                csv_path=csv_path,
                data_root=data_root_dir,
                mode=cfg["mode"],
                trigger_token=trigger_token,
                expected_height=wan_expected_height,
                expected_width=wan_expected_width,
                expected_frames=wan_expected_frames,
                strict=strict,
            )
            phase_report = validator.validate()

        report.phases.append(phase_report)
        ps = phase_report.summary
        logger.info(
            "Phase {}: {} total, {} passed, {} failed, {} warnings",
            phase,
            ps.get("total", 0),
            ps.get("passed", 0),
            ps.get("failed", 0),
            ps.get("warnings", 0),
        )

    if normalize:
        _apply_normalization(
            report,
            phase_configs,
            output_dir,
            wan_expected_width,
            wan_expected_height,
            wan_expected_frames,
            wan_reencode_fps,
        )

    report._compute_status()

    rp = report_path or (output_dir / "report.json")
    report.write_json(rp)
    report.write_summary_md(rp.with_name("summary.md"))
    logger.info("Validation report written to {}", rp)

    overall = report.overall_status
    if overall == "fail":
        logger.error(
            "Dataset validation FAILED ({} error(s)).", report.summary["failed"]
        )
        return 2
    if overall == "warn":
        logger.warning(
            "Dataset validation passed with {} warning(s).",
            report.summary["warnings"],
        )
    else:
        logger.info("Dataset validation PASSED.")
    return 0


def _apply_normalization(
    report: ValidationReport,
    phase_configs: dict[str, dict[str, Any]],
    output_dir: Path,
    req_w: int,
    req_h: int,
    req_frames: int,
    fps: int,
) -> None:
    for phase, cfg in phase_configs.items():
        if cfg["type"] != "wan":
            continue
        phase_report = next((p for p in report.phases if p.phase == phase), None)
        if phase_report is None:
            continue

        root = cfg.get("data_root")
        if root is None:
            continue

        passing_paths = [
            it.path
            for it in phase_report.items
            if it.status == "pass" and it.path.endswith(".mp4")
        ]
        if not passing_paths:
            logger.info("No passing videos to normalize for {}", phase)
            continue

        norm_dir = output_dir / phase / "normalized_videos"
        norm_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for src_str in passing_paths:
            src = Path(src_str)
            dst = norm_dir / src.name
            try:
                reencode_clip(src, dst, req_w, req_h, req_frames, fps)
                rows.append(
                    {
                        "path": str(dst),
                        "caption": _get_caption(phase_report, src_str),
                    }
                )
            except RuntimeError as exc:
                logger.warning("Normalization failed for {}: {}", src_str, exc)

        if rows:
            manifest_path = output_dir / phase / "metadata_normalized.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            logger.info(
                "Normalized {} clips for {} → {}",
                len(rows),
                phase,
                manifest_path,
            )

    report.normalization_applied = True
    report.normalization_output_dir = str(output_dir)


def _get_caption(phase_report: PhaseReport, path: str) -> str:
    for item in phase_report.items:
        if item.path == path:
            return item.caption or ""
    return ""


def run_normalize(
    *,
    csv_path: Path,
    data_root: Path,
    output_dir: Path,
    required_width: int = 832,
    required_height: int = 480,
    required_frames: int = 81,
    fps: int = 16,
) -> int:
    """Standalone normalization helper: validate + re-encode + emit manifest.

    Intended for direct library use without the CLI orchestration.
    """
    from videotuna.data.pipeline import DatasetPipeline

    pipeline = DatasetPipeline(
        output_dir=str(output_dir),
        trigger_token=None,
        required_min_frames=required_frames,
        required_height=required_height,
        required_width=required_width,
        train_val_split=0.0,
        reencode=True,
        reencode_fps=fps,
        preview_frames=0,
    )
    try:
        pipeline.run(csv_path=str(csv_path), data_root=str(data_root))
        logger.info("Normalization complete → {}", output_dir)
        return 0
    except Exception as exc:
        logger.error("Normalization failed: {}", exc)
        return 2
