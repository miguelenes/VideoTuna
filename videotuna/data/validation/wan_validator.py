from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from videotuna.data.validation.checks import (
    check_caption_hygiene,
    check_file_present,
    check_orphan_media,
    probe_image,
    probe_video,
)
from videotuna.data.validation.report import Issue, ItemResult, PhaseReport, Severity

WanMode = Literal["t2v", "i2v"]

_REQUIRED_COLUMNS_T2V = {"path", "caption"}
_REQUIRED_COLUMNS_I2V = {"image_path", "video_path", "caption"}


class WanDatasetValidator:
    def __init__(
        self,
        csv_path: Path,
        *,
        data_root: Optional[Path] = None,
        mode: WanMode = "t2v",
        trigger_token: Optional[str] = "sks_style",
        min_caption_length: int = 3,
        max_caption_length: int = 512,
        expected_height: int = 480,
        expected_width: int = 832,
        expected_frames: int = 81,
        strict: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.mode = mode
        self.trigger_token = trigger_token
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.expected_frames = expected_frames
        self.strict = strict

    def validate(self) -> PhaseReport:
        if not self.csv_path.is_file():
            return PhaseReport(
                phase=f"wan-{self.mode}",
                data_root=str(self.csv_path),
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
                items=[
                    ItemResult(
                        path=str(self.csv_path),
                        status="fail",
                        issues=[
                            Issue(
                                code="csv_not_found",
                                severity=Severity.ERROR,
                                message=f"CSV metadata file not found: {self.csv_path}",
                            )
                        ],
                    )
                ],
            )

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            return PhaseReport(
                phase=f"wan-{self.mode}",
                data_root=str(self.csv_path),
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
                items=[
                    ItemResult(
                        path=str(self.csv_path),
                        status="fail",
                        issues=[
                            Issue(
                                code="csv_read_error",
                                severity=Severity.ERROR,
                                message=f"Could not read CSV: {exc}",
                            )
                        ],
                    )
                ],
            )

        schema_issues = self._check_schema(df)
        if schema_issues:
            return PhaseReport(
                phase=f"wan-{self.mode}",
                data_root=str(self.csv_path),
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
                items=[
                    ItemResult(
                        path=str(self.csv_path),
                        status="fail",
                        issues=schema_issues,
                    )
                ],
            )

        items: list[ItemResult] = []
        passed = 0
        failed = 0
        warnings = 0
        all_media_paths: set[str] = set()

        for idx, row in df.iterrows():
            item_issues: list[Issue] = []
            media_paths: list[Path] = []

            if self.mode == "i2v":
                vid_path = self._resolve_path(str(row["video_path"]))
                img_path = self._resolve_path(str(row["image_path"]))
                primary_path = vid_path
                media_paths = [vid_path, img_path]
                all_media_paths.update((str(vid_path), str(img_path)))
            else:
                primary_path = self._resolve_path(str(row["path"]))
                media_paths = [primary_path]
                all_media_paths.add(str(primary_path))

            for mp in media_paths:
                fe = check_file_present(mp)
                if fe:
                    item_issues.append(fe)

            caption = str(row.get("caption", "")).strip()
            hyg_issues = check_caption_hygiene(
                caption,
                path=str(primary_path),
                trigger_token=self.trigger_token,
                min_length=self.min_caption_length,
                max_length=self.max_caption_length,
            )
            item_issues.extend(hyg_issues)

            if not any(i.code in ("missing_file", "not_a_file") for i in item_issues):
                self._check_video(primary_path, item_issues)
                if self.mode == "i2v" and img_path.exists():
                    self._check_image(img_path, item_issues)

            status = "pass"
            if any(i.severity == Severity.ERROR for i in item_issues):
                status = "fail"
                failed += 1
            elif any(i.severity == Severity.WARNING for i in item_issues):
                status = "warn"
                warnings += 1
            else:
                passed += 1

            items.append(
                ItemResult(
                    path=str(primary_path),
                    status=status,
                    issues=item_issues,
                    caption=caption,
                )
            )

        videos_dir = self.data_root / "videos" if self.data_root else Path("videos")
        orphan_issues = check_orphan_media(
            videos_dir,
            all_media_paths,
            media_extensions={".mp4", ".webm", ".mov", ".avi"},
            label="video",
        )
        if orphan_issues:
            warnings += len(orphan_issues)
            items.append(
                ItemResult(
                    path=str(videos_dir),
                    status="warn",
                    issues=orphan_issues,
                )
            )

        if self.mode == "i2v":
            images_dir = self.data_root / "images" if self.data_root else Path("images")
            img_orphans = check_orphan_media(
                images_dir,
                all_media_paths,
                media_extensions={".jpg", ".jpeg", ".png", ".webp"},
                label="image",
            )
            if img_orphans:
                warnings += len(img_orphans)
                items.append(
                    ItemResult(
                        path=str(images_dir),
                        status="warn",
                        issues=img_orphans,
                    )
                )

        return PhaseReport(
            phase=f"wan-{self.mode}",
            data_root=str(self.csv_path),
            summary={
                "total": len(df),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
            },
            items=items,
        )

    def _resolve_path(self, path_rel: str) -> Path:
        p = Path(path_rel)
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    def _check_video(self, path: Path, issues: list[Issue]) -> None:
        dims = probe_video(path)
        if dims is None:
            issues.append(
                Issue(
                    code="video_probe_failed",
                    severity=Severity.ERROR,
                    message=f"Could not probe video metadata: {path.name}",
                )
            )
            return
        vh, vw, vframes = dims
        self._check_dims(path, vh, vw, issues)
        if vframes is not None and vframes < self.expected_frames:
            issues.append(
                Issue(
                    code="video_too_short",
                    severity=Severity.ERROR,
                    message=(
                        f"Video {path.name}: {vframes} frames, "
                        f"need {self.expected_frames}"
                    ),
                    hint=f"Use clips with at least {self.expected_frames} frames, "
                    "or re-encode with --normalize.",
                )
            )

    def _check_dims(
        self,
        path: Path,
        h: int,
        w: int,
        issues: list[Issue],
        *,
        label: str = "Video",
    ) -> None:
        if h < self.expected_height or w < self.expected_width:
            sev = Severity.ERROR if self.strict else Severity.WARNING
            issues.append(
                Issue(
                    code="video_too_small",
                    severity=sev,
                    message=(
                        f"{label} {path.name}: {w}x{h}, "
                        f"expected {self.expected_width}x{self.expected_height}"
                    ),
                    hint=f"Re-encode to {self.expected_width}x{self.expected_height}.",
                )
            )
        elif h != self.expected_height or w != self.expected_width:
            sev = Severity.ERROR if self.strict else Severity.WARNING
            issues.append(
                Issue(
                    code="video_dim_mismatch",
                    severity=sev,
                    message=(
                        f"{label} {path.name}: {w}x{h}, "
                        f"expected {self.expected_width}x{self.expected_height}"
                    ),
                    hint=f"Re-encode to {self.expected_width}x{self.expected_height}.",
                )
            )

    def _check_image(self, path: Path, issues: list[Issue]) -> None:
        img_dims = probe_image(path)
        if img_dims is not None:
            ih, iw = img_dims
            if ih < self.expected_height or iw < self.expected_width:
                issues.append(
                    Issue(
                        code="image_too_small",
                        severity=Severity.ERROR,
                        message=(
                            f"Conditioning image {path.name}: {iw}x{ih}, "
                            f"expected {self.expected_width}x{self.expected_height}"
                        ),
                    )
                )

    def _check_schema(self, df: pd.DataFrame) -> list[Issue]:
        cols = set(df.columns)
        if self.mode == "i2v":
            if "image_path" not in cols:
                return [
                    Issue(
                        code="csv_missing_column",
                        severity=Severity.ERROR,
                        message="I2V CSV must have 'image_path' column",
                        hint="Add an 'image_path' column with conditioning image paths.",
                    )
                ]
            if "video_path" not in cols and "path" not in cols:
                return [
                    Issue(
                        code="csv_missing_column",
                        severity=Severity.ERROR,
                        message="I2V CSV must have 'video_path' or 'path' column",
                    )
                ]
        else:
            if "path" not in cols and "video_path" not in cols:
                return [
                    Issue(
                        code="csv_missing_column",
                        severity=Severity.ERROR,
                        message="T2V CSV must have 'path' or 'video_path' column",
                    )
                ]
        if "caption" not in cols:
            return [
                Issue(
                    code="csv_missing_column",
                    severity=Severity.ERROR,
                    message="CSV must have 'caption' column",
                )
            ]
        return []
