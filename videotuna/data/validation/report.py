from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    hint: Optional[str] = None


@dataclass
class ItemResult:
    path: str
    status: str  # "pass" | "fail" | "warn"
    issues: list[Issue] = field(default_factory=list)
    caption: Optional[str] = None


@dataclass
class PhaseReport:
    phase: str
    data_root: str
    summary: dict[str, Any]
    items: list[ItemResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.summary.get("failed", 0) == 0

    @property
    def total(self) -> int:
        return self.summary.get("total", 0)


@dataclass
class ValidationReport:
    generated_at: str = ""
    overall_status: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    phases: list[PhaseReport] = field(default_factory=list)
    normalization_applied: bool = False
    normalization_output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        if not self.overall_status:
            self._compute_status()

    def _compute_status(self) -> None:
        total_failed = sum(p.summary.get("failed", 0) for p in self.phases)
        total_warn = sum(p.summary.get("warnings", 0) for p in self.phases)
        total_items = sum(p.summary.get("total", 0) for p in self.phases)
        total_passed = sum(p.summary.get("passed", 0) for p in self.phases)
        self.summary = {
            "phases": len(self.phases),
            "total": total_items,
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warn,
        }
        if total_failed > 0:
            self.overall_status = "fail"
        elif total_warn > 0:
            self.overall_status = "warn"
        else:
            self.overall_status = "pass"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "normalization_applied": self.normalization_applied,
            "normalization_output_dir": self.normalization_output_dir,
            "summary": self.summary,
            "phases": [],
        }
        for p in self.phases:
            d["phases"].append(
                {
                    "phase": p.phase,
                    "data_root": p.data_root,
                    "summary": p.summary,
                    "items": [
                        {
                            "path": it.path,
                            "status": it.status,
                            "issues": [
                                {
                                    "code": iss.code,
                                    "severity": iss.severity.value,
                                    "message": iss.message,
                                    "hint": iss.hint,
                                }
                                for iss in it.issues
                            ],
                        }
                        for it in p.items
                    ],
                }
            )
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def summary_markdown(self) -> str:
        lines = [
            "# Data Validation Report",
            f"**Generated:** {self.generated_at}",
            f"**Overall status:** `{self.overall_status.upper()}`",
        ]
        if self.normalization_applied:
            dst = self.normalization_output_dir or ""
            lines.append(f"**Normalization:** applied → {dst}")
        lines.append("")
        lines.append("## Summary")
        s = self.summary
        lines.append(f"- **Phases:** {s.get('phases', 0)}")
        lines.append(f"- **Total items:** {s.get('total', 0)}")
        lines.append(f"- **Passed:** {s.get('passed', 0)}")
        lines.append(f"- **Failed:** {s.get('failed', 0)}")
        lines.append(f"- **Warnings:** {s.get('warnings', 0)}")
        lines.append("")

        for p in self.phases:
            lines.append(f"### Phase: `{p.phase}`")
            lines.append(f"**Data root:** `{p.data_root}`")
            lines.append(f"**Result:** {'✅ PASS' if p.passed else '❌ FAIL'}")
            lines.append("")
            ps = p.summary
            lines.append(f"- Total: {ps.get('total', 0)}")
            lines.append(f"- Passed: {ps.get('passed', 0)}")
            lines.append(f"- Failed: {ps.get('failed', 0)}")
            lines.append(f"- Warnings: {ps.get('warnings', 0)}")
            lines.append("")

            failed_items = [it for it in p.items if it.status == "fail"]
            warn_items = [it for it in p.items if it.status == "warn"]
            if failed_items:
                lines.append("#### Failures")
                for it in failed_items:
                    for iss in it.issues:
                        lines.append(f"- `{it.path}` — **{iss.code}**: {iss.message}")
                        if iss.hint:
                            lines.append(f"  ⤷ {iss.hint}")
                lines.append("")
            if warn_items:
                lines.append("#### Warnings")
                for it in warn_items:
                    for iss in it.issues:
                        lines.append(f"- `{it.path}` — {iss.message}")
                lines.append("")

        return "\n".join(lines)

    def write_summary_md(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.summary_markdown(), encoding="utf-8")
