from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from videotuna.data.validation.report import Issue, Severity

DEFAULT_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_VIDEO_EXTENSIONS: set[str] = {".mp4", ".webm", ".mov", ".avi", ".m4v"}
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PLACEHOLDER_PATTERNS = re.compile(
    r"^(caption|text|todo|placeholder|description|prompt|enter\s+.*)$",
    re.IGNORECASE,
)


def check_file_present(path: Path) -> Optional[Issue]:
    if not path.exists():
        return Issue(
            code="missing_file",
            severity=Severity.ERROR,
            message=f"File not found: {path}",
            hint=f"Ensure {path} exists in the expected location.",
        )
    if not path.is_file():
        return Issue(
            code="not_a_file",
            severity=Severity.ERROR,
            message=f"Path is not a file: {path}",
        )
    return None


def check_caption_hygiene(
    caption: str,
    path: str = "",
    *,
    trigger_token: Optional[str] = None,
    min_length: int = 1,
    max_length: int = 512,
) -> list[Issue]:
    issues: list[Issue] = []
    text = caption.strip() if caption else ""

    if not text:
        issues.append(
            Issue(
                code="empty_caption",
                severity=Severity.ERROR,
                message=f"Empty caption — {path}",
                hint="Provide a non-empty caption describing the content.",
            )
        )
        return issues

    if trigger_token and trigger_token not in text:
        issues.append(
            Issue(
                code="missing_trigger_token",
                severity=Severity.ERROR,
                message=f"Caption missing trigger token '{trigger_token}' — {path}",
                hint=f"Include '{trigger_token}' in the caption text.",
            )
        )

    control_matches = _CONTROL_CHAR_RE.findall(text)
    if control_matches:
        issues.append(
            Issue(
                code="control_characters",
                severity=Severity.WARNING,
                message=(
                    f"Caption contains {len(control_matches)} "
                    f"control character(s) — {path}"
                ),
                hint="Remove or replace control characters in the caption.",
            )
        )

    if len(text) < min_length:
        issues.append(
            Issue(
                code="caption_too_short",
                severity=Severity.ERROR,
                message=f"Caption length {len(text)} < minimum {min_length} — {path}",
                hint=f"Caption must be at least {min_length} characters.",
            )
        )

    if len(text) > max_length:
        issues.append(
            Issue(
                code="caption_too_long",
                severity=Severity.WARNING,
                message=f"Caption length {len(text)} exceeds {max_length} — {path}",
                hint=f"Consider trimming the caption to under {max_length} characters.",
            )
        )

    if _PLACEHOLDER_PATTERNS.match(text):
        issues.append(
            Issue(
                code="placeholder_caption",
                severity=Severity.ERROR,
                message=f"Caption appears to be a placeholder — {path}",
                hint="Replace placeholder text with a real description.",
            )
        )

    return issues


def check_orphan_sidecars(
    data_dir: Path,
    media_extensions: set[str] = DEFAULT_IMAGE_EXTENSIONS,
    sidecar_ext: str = ".txt",
) -> list[Issue]:
    issues: list[Issue] = []
    if not data_dir.is_dir():
        return issues

    media_stems: set[str] = set()
    for fp in data_dir.iterdir():
        if fp.suffix.lower() in media_extensions:
            media_stems.add(fp.stem)

    for fp in sorted(data_dir.iterdir()):
        if fp.suffix.lower() == sidecar_ext and fp.stem not in media_stems:
            issues.append(
                Issue(
                    code="orphan_sidecar",
                    severity=Severity.WARNING,
                    message=f"Orphan sidecar file with no matching media: {fp.name}",
                    hint=f"Add media for {fp.name} or remove the orphan file.",
                )
            )

    return issues


def check_orphan_media(
    data_dir: Path,
    known_paths: set[str],
    media_extensions: set[str],
    label: str = "",
) -> list[Issue]:
    issues: list[Issue] = []
    if not data_dir.is_dir():
        return issues

    for fp in sorted(data_dir.iterdir()):
        if fp.suffix.lower() in media_extensions and str(fp) not in known_paths:
            issues.append(
                Issue(
                    code="orphan_media",
                    severity=Severity.WARNING,
                    message=(
                        f"Orphan {label} file not referenced in metadata: {fp.name}"
                    ),
                    hint=(
                        f"Either reference {fp.name} in your metadata CSV or remove it."
                    ),
                )
            )

    return issues


def probe_image(path: Path) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image as PilImage

        with PilImage.open(str(path)) as img:
            w, h = img.size
        return h, w
    except Exception:
        return None


def probe_video(path: Path) -> Optional[tuple[int, int, int]]:
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            h = stream.codec_context.height
            w = stream.codec_context.width
            if stream.frames and stream.frames > 0:
                frames = int(stream.frames)
            elif stream.duration and stream.time_base and stream.average_rate:
                frames = int(stream.duration * stream.time_base * stream.average_rate)
            else:
                frames = None
        if h is None or w is None or frames is None:
            return None
        return h, w, frames
    except Exception:
        return None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def reencode_clip(
    src: Path,
    dst: Path,
    required_width: int,
    required_height: int,
    required_frames: int,
    fps: int = 16,
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"scale={required_width}:{required_height}",
        "-r",
        str(fps),
        "-frames:v",
        str(required_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(f"ffmpeg re-encode failed for {src}: {exc}") from exc
