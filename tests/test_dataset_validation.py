"""Tests for the dataset validation and preprocessing workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videotuna.data.validation.checks import (
    check_caption_hygiene,
    check_file_present,
    check_orphan_sidecars,
    probe_image,
)
from videotuna.data.validation.flux_validator import FluxDatasetValidator
from videotuna.data.validation.report import (
    Issue,
    ItemResult,
    PhaseReport,
    Severity,
    ValidationReport,
)
from videotuna.data.validation.runner import resolve_phase_configs, validate_datasets
from videotuna.data.validation.wan_validator import WanDatasetValidator

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class TestCheckFilePresent:
    def test_file_exists(self, tmp_path: Path) -> None:
        f = tmp_path / "exists.txt"
        f.write_text("hello")
        assert check_file_present(f) is None

    def test_file_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.txt"
        issue = check_file_present(f)
        assert issue is not None
        assert issue.code == "missing_file"
        assert issue.severity == Severity.ERROR

    def test_not_a_file(self, tmp_path: Path) -> None:
        issue = check_file_present(tmp_path)
        assert issue is not None
        assert issue.code == "not_a_file"


class TestCheckCaptionHygiene:
    def test_empty_caption(self) -> None:
        issues = check_caption_hygiene("", path="test")
        assert any(i.code == "empty_caption" for i in issues)

    def test_whitespace_only(self) -> None:
        issues = check_caption_hygiene("   ", path="test")
        assert any(i.code == "empty_caption" for i in issues)

    def test_missing_trigger_token(self) -> None:
        issues = check_caption_hygiene(
            "a portrait", trigger_token="sks_style", path="test"
        )
        assert any(i.code == "missing_trigger_token" for i in issues)

    def test_trigger_token_present(self) -> None:
        issues = check_caption_hygiene(
            "sks_style, portrait", trigger_token="sks_style", path="test"
        )
        assert not any(i.code == "missing_trigger_token" for i in issues)

    def test_no_trigger_check_when_none(self) -> None:
        issues = check_caption_hygiene("portrait", trigger_token=None, path="test")
        assert not any(i.code == "missing_trigger_token" for i in issues)

    def test_control_characters(self) -> None:
        issues = check_caption_hygiene("sks_style, port\u0000rait", path="test")
        assert any(i.code == "control_characters" for i in issues)

    def test_too_short(self) -> None:
        issues = check_caption_hygiene("ab", path="test", min_length=3)
        assert any(i.code == "caption_too_short" for i in issues)

    def test_too_long(self) -> None:
        issues = check_caption_hygiene("x" * 600, path="test", max_length=512)
        assert any(i.code == "caption_too_long" for i in issues)

    def test_placeholder_caption(self) -> None:
        placeholders = ("caption", "TODO", "text", "placeholder", "Enter caption here")
        for placeholder in placeholders:
            issues = check_caption_hygiene(placeholder, path="test")
            assert any(
                i.code == "placeholder_caption" for i in issues
            ), f"missed {placeholder}"

    def test_clean_caption(self) -> None:
        issues = check_caption_hygiene(
            "sks_style, a beautiful portrait with soft lighting",
            trigger_token="sks_style",
            path="test",
        )
        assert len(issues) == 0


class TestCheckOrphanSidecars:
    def test_no_orphans(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"dummy")
        txt = tmp_path / "photo.txt"
        txt.write_text("caption")
        issues = check_orphan_sidecars(tmp_path)
        assert len(issues) == 0

    def test_finds_orphan_txt(self, tmp_path: Path) -> None:
        (tmp_path / "orphan.txt").write_text("no image")
        issues = check_orphan_sidecars(tmp_path)
        assert any(i.code == "orphan_sidecar" for i in issues)

    def test_empty_dir(self, tmp_path: Path) -> None:
        issues = check_orphan_sidecars(tmp_path)
        assert len(issues) == 0

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        issues = check_orphan_sidecars(tmp_path / "nope")
        assert len(issues) == 0


class TestProbeImage:
    def test_probe_png(self, tmp_path: Path) -> None:
        from PIL import Image

        img = tmp_path / "test.png"
        Image.new("RGB", (832, 480)).save(img)
        dims = probe_image(img)
        assert dims == (480, 832)

    def test_probe_fail(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.png"
        f.write_bytes(b"not-an-image")
        assert probe_image(f) is None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_issue_creation(self) -> None:
        issue = Issue(
            code="test_code",
            severity=Severity.ERROR,
            message="something went wrong",
            hint="fix it",
        )
        assert issue.code == "test_code"
        assert issue.severity == Severity.ERROR

    def test_item_result(self) -> None:
        item = ItemResult(
            path="/fake/path",
            status="fail",
            issues=[Issue("err", Severity.ERROR, "fail")],
        )
        assert item.status == "fail"
        assert len(item.issues) == 1

    def test_phase_report_passed_property(self) -> None:
        p = PhaseReport(
            phase="test",
            data_root="/x",
            summary={"total": 5, "passed": 5, "failed": 0, "warnings": 0},
        )
        assert p.passed is True

    def test_phase_report_failed_property(self) -> None:
        p = PhaseReport(
            phase="test",
            data_root="/x",
            summary={"total": 5, "passed": 4, "failed": 1, "warnings": 0},
        )
        assert p.passed is False

    def test_validation_report_status_pass(self) -> None:
        r = ValidationReport()
        r.phases.append(
            PhaseReport(
                phase="test",
                data_root="/x",
                summary={"total": 1, "passed": 1, "failed": 0, "warnings": 0},
            )
        )
        r._compute_status()
        assert r.overall_status == "pass"

    def test_validation_report_status_fail(self) -> None:
        r = ValidationReport()
        r.phases.append(
            PhaseReport(
                phase="test",
                data_root="/x",
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
            )
        )
        r._compute_status()
        assert r.overall_status == "fail"

    def test_validation_report_status_warn(self) -> None:
        r = ValidationReport()
        r.phases.append(
            PhaseReport(
                phase="test",
                data_root="/x",
                summary={"total": 1, "passed": 1, "failed": 0, "warnings": 1},
            )
        )
        r._compute_status()
        assert r.overall_status == "warn"

    def test_to_json_roundtrip(self, tmp_path: Path) -> None:
        r = ValidationReport(generated_at="2025-01-01T00:00:00")
        r.phases.append(
            PhaseReport(
                phase="flux-t2i",
                data_root="/fake/data",
                summary={"total": 2, "passed": 1, "failed": 1, "warnings": 0},
                items=[
                    ItemResult("/ok.jpg", "pass"),
                    ItemResult(
                        "/bad.jpg",
                        "fail",
                        issues=[
                            Issue("missing_file", Severity.ERROR, "not found"),
                        ],
                    ),
                ],
            )
        )
        r._compute_status()

        p = tmp_path / "report.json"
        r.write_json(p)
        assert p.is_file()

        data = json.loads(p.read_text())
        assert data["overall_status"] == "fail"
        assert len(data["phases"]) == 1
        assert len(data["phases"][0]["items"]) == 2

    def test_summary_markdown_includes_phases(self) -> None:
        r = ValidationReport()
        r.phases.append(
            PhaseReport(
                phase="flux-t2i",
                data_root="/d",
                summary={"total": 1, "passed": 1, "failed": 0, "warnings": 0},
            )
        )
        r._compute_status()
        md = r.summary_markdown()
        assert "flux-t2i" in md
        assert "PASS" in md

    def test_summary_markdown_lists_failures(self) -> None:
        r = ValidationReport()
        r.phases.append(
            PhaseReport(
                phase="wan-t2v",
                data_root="/d",
                summary={"total": 1, "passed": 0, "failed": 1, "warnings": 0},
                items=[
                    ItemResult(
                        "/bad.mp4",
                        "fail",
                        issues=[
                            Issue("missing_file", Severity.ERROR, "not found"),
                        ],
                    )
                ],
            )
        )
        r._compute_status()
        md = r.summary_markdown()
        assert "FAIL" in md
        assert "missing_file" in md


# ---------------------------------------------------------------------------
# FluxDatasetValidator
# ---------------------------------------------------------------------------


class TestFluxDatasetValidator:
    def test_passes_good_dir(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (1024, 1024)).save(str(tmp_path / "001.jpg"))
        (tmp_path / "001.txt").write_text("sks_style, portrait", encoding="utf-8")
        Image.new("RGB", (800, 800)).save(str(tmp_path / "002.png"))
        (tmp_path / "002.txt").write_text("sks_style, landscape", encoding="utf-8")

        validator = FluxDatasetValidator(
            tmp_path,
            trigger_token="sks_style",
            min_resolution=512,
        )
        report = validator.validate()
        assert report.passed
        assert report.summary["total"] == 2
        assert report.summary["passed"] == 2

    def test_flags_missing_sidecar(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (512, 512)).save(str(tmp_path / "img.jpg"))
        validator = FluxDatasetValidator(tmp_path)
        report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "missing_sidecar" for item in report.items for i in item.issues
        )

    def test_flags_orphan_caption(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (512, 512)).save(str(tmp_path / "img.jpg"))
        (tmp_path / "img.txt").write_text("sks_style, test")
        (tmp_path / "orphan.txt").write_text("no image")
        validator = FluxDatasetValidator(tmp_path)
        report = validator.validate()
        assert any(
            i.code == "orphan_sidecar" for item in report.items for i in item.issues
        )

    def test_flags_bad_caption(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (512, 512)).save(str(tmp_path / "img.jpg"))
        (tmp_path / "img.txt").write_text("")  # empty caption
        validator = FluxDatasetValidator(tmp_path, trigger_token="sks_style")
        report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "empty_caption" for item in report.items for i in item.issues
        )

    def test_image_too_small(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (100, 100)).save(str(tmp_path / "small.jpg"))
        (tmp_path / "small.txt").write_text("sks_style, tiny", encoding="utf-8")
        validator = FluxDatasetValidator(
            tmp_path,
            trigger_token="sks_style",
            min_resolution=512,
        )
        report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "image_too_small" for item in report.items for i in item.issues
        )

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        validator = FluxDatasetValidator(tmp_path / "nope")
        report = validator.validate()
        assert not report.passed
        assert any(i.code == "dir_not_found" for i in report.items[0].issues)

    def test_skips_non_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("# notes")
        validator = FluxDatasetValidator(tmp_path)
        report = validator.validate()
        assert report.summary["total"] == 0
        assert report.passed


# ---------------------------------------------------------------------------
# WanDatasetValidator
# ---------------------------------------------------------------------------


class TestWanT2VValidator:
    def test_passes_good_csv(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "clip001.mp4").write_bytes(b"dummy")
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "path,caption\n" f"{video_dir}/clip001.mp4,sks_style test clip\n",
            encoding="utf-8",
        )
        validator = WanDatasetValidator(csv_path, data_root=tmp_path, mode="t2v")
        # patch probe_video to succeed
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_video",
                lambda _: (480, 832, 81),
            )
            report = validator.validate()
        assert report.passed

    def test_flags_missing_video(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "path,caption\n" f"{tmp_path}/videos/nonexistent.mp4,sks_style test\n",
        )
        validator = WanDatasetValidator(csv_path, data_root=tmp_path, mode="t2v")
        report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "missing_file" for item in report.items for i in item.issues
        )

    def test_flags_missing_trigger_token(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "clip.mp4").write_bytes(b"dummy")
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "path,caption\n" f"{video_dir}/clip.mp4,no trigger here\n",
        )
        validator = WanDatasetValidator(
            csv_path,
            data_root=tmp_path,
            mode="t2v",
            trigger_token="sks_style",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_video",
                lambda _: (480, 832, 81),
            )
            report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "missing_trigger_token"
            for item in report.items
            for i in item.issues
        )

    def test_short_frame_count(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "clip.mp4").write_bytes(b"dummy")
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "path,caption\n" f"{video_dir}/clip.mp4,sks_style short\n",
        )
        validator = WanDatasetValidator(
            csv_path,
            data_root=tmp_path,
            mode="t2v",
            expected_frames=81,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_video",
                lambda _: (480, 832, 30),  # only 30 frames
            )
            report = validator.validate()
        assert not report.passed
        assert any(
            i.code == "video_too_short" for item in report.items for i in item.issues
        )

    def test_wrong_dimensions_warn_by_default(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "clip.mp4").write_bytes(b"dummy")
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "path,caption\n" f"{video_dir}/clip.mp4,sks_style clip\n",
        )
        validator = WanDatasetValidator(
            csv_path,
            data_root=tmp_path,
            mode="t2v",
            expected_height=480,
            expected_width=832,
            strict=False,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_video",
                lambda _: (480, 1280, 81),  # wider than expected
            )
            report = validator.validate()
        # non-strict: should warn not fail
        assert report.passed
        assert any(
            i.code == "video_dim_mismatch" for item in report.items for i in item.issues
        )

    def test_csv_not_found(self, tmp_path: Path) -> None:
        validator = WanDatasetValidator(tmp_path / "nope.csv", mode="t2v")
        report = validator.validate()
        assert not report.passed
        assert any(i.code == "csv_not_found" for i in report.items[0].issues)

    def test_csv_schema_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col1,col2\nval1,val2\n")
        validator = WanDatasetValidator(csv_path, mode="t2v")
        report = validator.validate()
        assert not report.passed
        assert any(i.code == "csv_missing_column" for i in report.items[0].issues)


class TestWanI2VValidator:
    def test_passes_good_i2v_csv(self, tmp_path: Path) -> None:
        videos = tmp_path / "videos"
        images = tmp_path / "images"
        videos.mkdir(parents=True)
        images.mkdir(parents=True)
        (videos / "clip.mp4").write_bytes(b"dummy")
        (images / "ref.jpg").write_bytes(b"dummy")
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(
            "image_path,video_path,caption\n"
            f"{images}/ref.jpg,{videos}/clip.mp4,sks_style i2v clip\n",
        )
        validator = WanDatasetValidator(csv_path, data_root=tmp_path, mode="i2v")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_video",
                lambda _: (480, 832, 81),
            )
            mp.setattr(
                "videotuna.data.validation.wan_validator.probe_image",
                lambda _: (480, 832),
            )
            report = validator.validate()
        assert report.passed
        assert report.summary["total"] == 1

    def test_missing_image_path_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("path,caption\n/path/to/vid,sks_style clip\n")
        validator = WanDatasetValidator(csv_path, mode="i2v")
        report = validator.validate()
        assert not report.passed
        assert any(i.code == "csv_missing_column" for i in report.items[0].issues)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestResolvePhaseConfigs:
    def test_all_phases(self) -> None:
        configs = resolve_phase_configs(("all",), ())
        assert set(configs.keys()) == {"flux-t2i", "wan-t2v", "wan-i2v"}

    def test_single_phase(self) -> None:
        configs = resolve_phase_configs(("flux-t2i",), ())
        assert set(configs.keys()) == {"flux-t2i"}

    def test_data_root_override(self) -> None:
        configs = resolve_phase_configs(("flux-t2i",), ("/custom/path",))
        cfg = configs["flux-t2i"]
        assert cfg["data_root"] == Path("/custom/path")


class TestValidateDatasets:
    def test_no_phases(self) -> None:
        code = validate_datasets(phases=())
        assert code == 0

    def test_flux_t2i_nonexistent_dir(self, tmp_path: Path) -> None:
        code = validate_datasets(
            phases=("flux-t2i",),
            data_roots=(str(tmp_path / "nope"),),
        )
        assert code == 2  # should fail

    def test_flux_t2i_clean_dir(self, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (1024, 1024)).save(str(tmp_path / "test.jpg"))
        (tmp_path / "test.txt").write_text("sks_style, test image", encoding="utf-8")
        code = validate_datasets(
            phases=("flux-t2i",),
            data_roots=(str(tmp_path),),
        )
        assert code == 0

    def test_wan_t2v_nonexistent_csv(self, tmp_path: Path) -> None:
        code = validate_datasets(
            phases=("wan-t2v",),
            data_roots=(str(tmp_path),),
        )
        assert code == 2

    def test_output_dir_creates_report(self, tmp_path: Path) -> None:
        out = tmp_path / "val_out"
        code = validate_datasets(
            phases=(),
            output_dir=out,
        )
        assert code == 0
