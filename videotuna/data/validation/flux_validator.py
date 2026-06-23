from __future__ import annotations

from pathlib import Path
from typing import Optional

from videotuna.data.validation.checks import (
    DEFAULT_IMAGE_EXTENSIONS,
    check_caption_hygiene,
    check_file_present,
    check_orphan_sidecars,
    probe_image,
)
from videotuna.data.validation.report import Issue, ItemResult, PhaseReport, Severity


class FluxDatasetValidator:
    def __init__(
        self,
        data_dir: Path,
        *,
        trigger_token: Optional[str] = "sks_style",
        min_caption_length: int = 3,
        max_caption_length: int = 512,
        min_resolution: int = 512,
        image_extensions: set[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> None:
        self.data_dir = data_dir
        self.trigger_token = trigger_token
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.min_resolution = min_resolution
        self.image_extensions = image_extensions

    def validate(self) -> PhaseReport:
        if not self.data_dir.is_dir():
            items = [
                ItemResult(
                    path=str(self.data_dir),
                    status="fail",
                    issues=[
                        Issue(
                            code="dir_not_found",
                            severity=Severity.ERROR,
                            message=f"Data directory not found: {self.data_dir}",
                        )
                    ],
                )
            ]
            return PhaseReport(
                phase="flux-t2i",
                data_root=str(self.data_dir),
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
                items=items,
            )

        image_files = sorted(
            p
            for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.image_extensions
        )
        items: list[ItemResult] = []
        passed = 0
        failed = 0
        warnings = 0

        for img_path in image_files:
            item_issues: list[Issue] = []

            sidecar = img_path.with_suffix(".txt")
            missing_sidecar = not sidecar.is_file()

            has_file_issue = check_file_present(img_path)
            if has_file_issue:
                item_issues.append(has_file_issue)

            if missing_sidecar:
                item_issues.append(
                    Issue(
                        code="missing_sidecar",
                        severity=Severity.ERROR,
                        message=f"No sidecar caption file for {img_path.name}",
                        hint=(
                            f"Create {sidecar.name} with captions "
                            f"containing '{self.trigger_token}'."
                        ),
                    )
                )

            dims = probe_image(img_path)
            if dims is not None:
                h, w = dims
                if h < self.min_resolution or w < self.min_resolution:
                    item_issues.append(
                        Issue(
                            code="image_too_small",
                            severity=Severity.ERROR,
                            message=(
                                f"Image {img_path.name}: {w}x{h} below min "
                                f"{self.min_resolution}x{self.min_resolution}"
                            ),
                            hint=(
                                f"Use images at least "
                                f"{self.min_resolution}x{self.min_resolution} pixels."
                            ),
                        )
                    )
            else:
                item_issues.append(
                    Issue(
                        code="image_probe_failed",
                        severity=Severity.ERROR,
                        message=f"Could not read image metadata: {img_path.name}",
                    )
                )

            if not missing_sidecar:
                caption = sidecar.read_text(encoding="utf-8").strip()
                hyg_issues = check_caption_hygiene(
                    caption,
                    path=str(sidecar),
                    trigger_token=self.trigger_token,
                    min_length=self.min_caption_length,
                    max_length=self.max_caption_length,
                )
                item_issues.extend(hyg_issues)

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
                    path=str(img_path),
                    status=status,
                    issues=item_issues,
                    caption=caption if not missing_sidecar else None,
                )
            )

        orphan_issues = check_orphan_sidecars(self.data_dir, self.image_extensions)
        if orphan_issues:
            warnings += len(orphan_issues)
            items.append(
                ItemResult(
                    path=str(self.data_dir),
                    status="warn",
                    issues=orphan_issues,
                )
            )

        return PhaseReport(
            phase="flux-t2i",
            data_root=str(self.data_dir),
            summary={
                "total": len(image_files),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
            },
            items=items,
        )
