"""Validated dataset ingestion and preparation pipeline.

Ingest image/video manifests, enforce trigger-token and caption rules,
verify frame counts and resolution, optionally re-encode clips with ffmpeg,
and generate reproducible train/validation splits plus smoke-test previews.

Usage::

    from videotuna.data.pipeline import DatasetPipeline

    pipeline = DatasetPipeline(
        output_dir="data/t2v/domain",
        trigger_token="sks_style",
        required_min_frames=81,
        required_height=480,
        required_width=832,
        train_val_split=0.2,
        seed=42,
        reencode=False,
        preview_frames=1,
    )
    pipeline.run(csv_path="raw/metadata.csv", data_root="raw")

    # Or from a folder with sidecar .txt captions:
    pipeline.run_from_folder(video_dir="raw/videos", caption_strategy="filename")
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_CaptionStrategy = Literal["filename", "csv"]


class DatasetPipeline:
    """Validated ingest → validate → split → preview pipeline for LoRA datasets.

    Args:
        output_dir: Directory to write validated manifest and splits.
        trigger_token: Optional token that every caption must contain.
            Rows whose captions lack it are logged and dropped.
        required_min_frames: Minimum frame count a video must have.
            Videos with fewer frames are dropped.
        required_height: Minimum frame height in pixels.
        required_width: Minimum frame width in pixels.
        train_val_split: Fraction of data to reserve for validation.
            Set to 0.0 to skip splitting.
        seed: RNG seed for reproducible splits.
        reencode: If True, re-encode each accepted clip to the required
            resolution and frame count using ffmpeg.
        reencode_fps: Target fps for re-encoded clips. Defaults to 24.
        preview_frames: Number of first-frame PNG previews to write per clip.
            Set to 0 to disable.
    """

    def __init__(
        self,
        output_dir: Union[str, os.PathLike],
        trigger_token: Optional[str] = None,
        required_min_frames: int = 1,
        required_height: int = 1,
        required_width: int = 1,
        train_val_split: float = 0.2,
        seed: int = 42,
        reencode: bool = False,
        reencode_fps: int = 24,
        preview_frames: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.trigger_token = trigger_token
        self.required_min_frames = required_min_frames
        self.required_height = required_height
        self.required_width = required_width
        self.train_val_split = train_val_split
        self.seed = seed
        self.reencode = reencode
        self.reencode_fps = reencode_fps
        self.preview_frames = preview_frames

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: Union[str, os.PathLike],
        data_root: Union[str, os.PathLike, None] = None,
    ) -> dict[str, Path]:
        """Run the full pipeline from a CSV manifest.

        Args:
            csv_path: Path to the input CSV.  Must have ``path`` and
                ``caption`` columns; optional ``fps``, ``frames``,
                ``height``, ``width`` for pre-computed metadata.
            data_root: If provided, relative ``path`` values in the CSV
                are resolved against this directory.

        Returns:
            Dictionary with keys ``manifest``, ``train``, ``val``
            (or only ``manifest`` when ``train_val_split == 0``), and
            ``previews`` pointing to the respective output paths.
        """
        df = self._load_manifest(csv_path, data_root)
        return self._process(df)

    def run_from_folder(
        self,
        video_dir: Union[str, os.PathLike],
        caption_strategy: _CaptionStrategy = "filename",
    ) -> dict[str, Path]:
        """Run the full pipeline from a folder of clips with sidecar captions.

        Args:
            video_dir: Directory containing video (or image) files.
            caption_strategy: How to resolve captions:
                ``"filename"`` — looks for ``<stem>.txt`` sidecars next
                to each clip.  ``"csv"`` is a no-op placeholder (use
                :meth:`run` with an explicit CSV instead).

        Returns:
            Same as :meth:`run`.
        """
        df = self._load_folder_manifest(video_dir, caption_strategy)
        return self._process(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(
        self,
        csv_path: Union[str, os.PathLike],
        data_root: Union[str, os.PathLike, None],
    ) -> pd.DataFrame:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self._assert_schema(df, csv_path)

        if "path" not in df.columns and "video_path" in df.columns:
            df = df.rename(columns={"video_path": "path"})

        if data_root is not None:
            root = Path(data_root)
            df["path"] = df["path"].apply(
                lambda p: str(root / p) if not Path(p).is_absolute() else p
            )

        return df

    def _load_folder_manifest(
        self,
        video_dir: Union[str, os.PathLike],
        caption_strategy: _CaptionStrategy,
    ) -> pd.DataFrame:
        from videotuna.data.datasets_utils import is_image, is_video

        video_dir = Path(video_dir)
        rows: list[dict] = []
        for fp in sorted(video_dir.iterdir()):
            if not (is_video(str(fp)) or is_image(str(fp))):
                continue
            if caption_strategy == "filename":
                sidecar = fp.with_suffix(".txt")
                if not sidecar.exists():
                    logger.warning("No sidecar caption for %s — skipping", fp)
                    continue
                caption = sidecar.read_text().strip()
            else:
                caption = ""
            rows.append({"path": str(fp), "caption": caption})

        if not rows:
            raise ValueError(f"No supported media files found under {video_dir}")

        return pd.DataFrame(rows)

    @staticmethod
    def _assert_schema(df: pd.DataFrame, path: Union[str, Path]) -> None:
        cols = set(df.columns)
        if not ({"path", "video_path", "image_path"} & cols):
            raise ValueError(
                f"{path}: CSV must have a 'path', 'video_path', or 'image_path' column."
            )
        if "caption" not in cols:
            raise ValueError(f"{path}: CSV must have a 'caption' column.")

    def _validate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate each row; return filtered DataFrame with only good rows."""
        from videotuna.data.datasets_utils import is_image, is_video
        from videotuna.utils.video_io import get_video_frame_count

        kept: list[int] = []
        for idx, row in df.iterrows():
            path = str(row["path"])

            # ── existence check ────────────────────────────────────────
            if not Path(path).exists():
                logger.warning("Row %s: file not found — %s", idx, path)
                continue

            # ── caption check ──────────────────────────────────────────
            caption = str(row.get("caption", "")).strip()
            if not caption:
                logger.warning("Row %s: empty caption — %s", idx, path)
                continue
            if self.trigger_token and self.trigger_token not in caption:
                logger.warning(
                    "Row %s: caption missing trigger token '%s' — %s",
                    idx,
                    self.trigger_token,
                    path,
                )
                continue

            # ── video/image metadata check ─────────────────────────────
            if is_video(path):
                h = row.get("height") if pd.notna(row.get("height", None)) else None
                w = row.get("width") if pd.notna(row.get("width", None)) else None
                frames = (
                    row.get("frames") if pd.notna(row.get("frames", None)) else None
                )

                if h is None or w is None or frames is None:
                    h, w, frames = self._probe_video(path)
                    if h is None:
                        logger.warning(
                            "Row %s: could not probe video metadata — %s", idx, path
                        )
                        continue

                try:
                    frame_count = get_video_frame_count(path)
                except Exception as exc:
                    logger.warning(
                        "Row %s: frame count probe failed (%s) — %s", idx, exc, path
                    )
                    continue

                if frame_count < self.required_min_frames:
                    logger.warning(
                        "Row %s: only %d frames (need %d) — %s",
                        idx,
                        frame_count,
                        self.required_min_frames,
                        path,
                    )
                    continue
                if int(h) < self.required_height or int(w) < self.required_width:
                    logger.warning(
                        "Row %s: resolution %dx%d below minimum %dx%d — %s",
                        idx,
                        h,
                        w,
                        self.required_height,
                        self.required_width,
                        path,
                    )
                    continue

            elif is_image(path):
                h = row.get("height") if pd.notna(row.get("height", None)) else None
                w = row.get("width") if pd.notna(row.get("width", None)) else None

                if h is None or w is None:
                    h, w = self._probe_image(path)
                    if h is None:
                        logger.warning(
                            "Row %s: could not probe image metadata — %s", idx, path
                        )
                        continue

                if int(h) < self.required_height or int(w) < self.required_width:
                    logger.warning(
                        "Row %s: image %dx%d below minimum %dx%d — %s",
                        idx,
                        h,
                        w,
                        self.required_height,
                        self.required_width,
                        path,
                    )
                    continue
            else:
                logger.warning("Row %s: unsupported file type — %s", idx, path)
                continue

            kept.append(idx)

        result = df.loc[kept].reset_index(drop=True)
        n_dropped = len(df) - len(result)
        if n_dropped:
            logger.info("Dropped %d rows during validation.", n_dropped)
        return result

    @staticmethod
    def _probe_video(path: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Return (height, width, frames) by probing with PyAV."""
        try:
            import av

            with av.open(path) as container:
                stream = container.streams.video[0]
                h = stream.codec_context.height
                w = stream.codec_context.width
                if stream.frames and stream.frames > 0:
                    frames = int(stream.frames)
                elif stream.duration and stream.time_base and stream.average_rate:
                    frames = int(
                        stream.duration * stream.time_base * stream.average_rate
                    )
                else:
                    frames = None
            return h, w, frames
        except Exception:
            return None, None, None

    @staticmethod
    def _probe_image(path: str) -> tuple[Optional[int], Optional[int]]:
        """Return (height, width) for an image using Pillow."""
        try:
            from PIL import Image as PilImage

            with PilImage.open(path) as img:
                w, h = img.size
            return h, w
        except Exception:
            return None, None

    def _reencode_clip(self, src: str, dst: str) -> bool:
        """Re-encode *src* to *dst* at target resolution and frame count.

        Returns True on success, False on ffmpeg failure.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-vf",
            f"scale={self.required_width}:{self.required_height}",
            "-r",
            str(self.reencode_fps),
            "-frames:v",
            str(self.required_min_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            dst,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("ffmpeg re-encode failed for %s: %s", src, exc)
            return False

    def _write_previews(self, df: pd.DataFrame, preview_dir: Path) -> None:
        """Write first-frame PNG previews for each row in *df*."""
        from videotuna.data.datasets_utils import is_video
        from videotuna.utils.video_io import read_video_frames

        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in df.iterrows():
            path = str(row["path"])
            stem = Path(path).stem
            out = preview_dir / f"{stem}_preview.png"
            try:
                if is_video(path):
                    frame = read_video_frames(path, [0])[0]  # CHW uint8
                    from PIL import Image as PilImage

                    arr = frame.permute(1, 2, 0).numpy()
                    PilImage.fromarray(arr).save(out)
                else:
                    shutil.copy(path, out)
            except Exception as exc:
                logger.warning("Preview failed for row %s (%s): %s", idx, path, exc)

    def _write_outputs(self, df: pd.DataFrame) -> dict[str, Path]:
        """Write manifest, optional splits, and previews; return path map."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = self.output_dir / "metadata.csv"
        df.to_csv(manifest_path, index=False)
        logger.info("Wrote validated manifest → %s (%d rows)", manifest_path, len(df))

        result: dict[str, Path] = {"manifest": manifest_path}

        if self.train_val_split > 0.0 and len(df) >= 2:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(df))
            n_val = max(1, int(len(df) * self.train_val_split))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_path = self.output_dir / "metadata_train.csv"
            val_path = self.output_dir / "metadata_val.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            logger.info(
                "Wrote split: train=%d → %s, val=%d → %s",
                len(train_df),
                train_path,
                len(val_df),
                val_path,
            )
            result["train"] = train_path
            result["val"] = val_path

        if self.preview_frames > 0:
            preview_dir = self.output_dir / "previews"
            self._write_previews(df, preview_dir)
            result["previews"] = preview_dir

        return result

    def _process(self, df: pd.DataFrame) -> dict[str, Path]:
        """Validate, optionally re-encode, then write outputs."""
        df = self._validate_rows(df)

        if df.empty:
            raise RuntimeError(
                "All rows were dropped during validation. Check your manifest."
            )

        if self.reencode:
            reencode_dir = self.output_dir / "reencoded"
            reencode_dir.mkdir(parents=True, exist_ok=True)
            new_paths: list[str] = []
            for _, row in df.iterrows():
                src = str(row["path"])
                dst = str(reencode_dir / Path(src).name)
                ok = self._reencode_clip(src, dst)
                new_paths.append(dst if ok else src)
            df = df.copy()
            df["path"] = new_paths
            df["fps"] = self.reencode_fps
            df["frames"] = self.required_min_frames
            df["height"] = self.required_height
            df["width"] = self.required_width

        return self._write_outputs(df)
