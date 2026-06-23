"""Regression tests for the manifest-driven inference pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from videotuna.base.inference_base import InferenceBase
from videotuna.base.inference_manifest import InferenceManifest, InferenceSample


class DummyFlow(InferenceBase):
    """Minimal concrete InferenceBase for unit testing."""


@pytest.fixture
def flow():
    return DummyFlow()


def test_inference_manifest_serialization():
    manifest = InferenceManifest(
        model_family="wan",
        mode="t2v",
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        lora_path="/tmp/lora.safetensors",
    )
    manifest.add_sample(
        InferenceSample(
            sample_id="t2v-0000-000-42",
            prompt="a cat",
            output_path="/tmp/out.mp4",
            seed=42,
            mode="t2v",
            index=0,
            sample_index=0,
            height=720,
            width=1280,
            frames=81,
            num_inference_steps=4,
            guidance_scale=5.0,
        )
    )
    data = manifest.as_dict()
    assert data["version"] == "1.0"
    assert data["model_family"] == "wan"
    assert data["mode"] == "t2v"
    assert data["model_id"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert data["lora_path"] == "/tmp/lora.safetensors"
    assert len(data["samples"]) == 1
    sample = data["samples"][0]
    assert sample["sample_id"] == "t2v-0000-000-42"
    assert sample["prompt"] == "a cat"
    assert sample["output_path"] == "/tmp/out.mp4"


def test_manifest_write_roundtrip(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        manifest = InferenceManifest(
            model_family="flux",
            mode="t2i",
            model_id="black-forest-labs/FLUX.1-dev",
        )
        manifest.add_sample(
            InferenceSample(
                sample_id="t2i-0000-000-1",
                prompt="a dog",
                output_path=os.path.join(tmp, "0000_000_a_dog_abcdef12.jpg"),
                seed=1,
                mode="t2i",
                index=0,
                sample_index=0,
            )
        )
        path = flow.save_manifest(manifest, tmp)
        assert path == os.path.join(tmp, "manifest.json")
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["model_family"] == "flux"
        assert loaded["samples"][0]["prompt"] == "a dog"


def test_generate_filename_is_deterministic_and_unique(flow: InferenceBase):
    f1 = flow._generate_filename("a cat", 42, None, "t2v", 0, 0)
    f2 = flow._generate_filename("a cat", 42, None, "t2v", 0, 0)
    f3 = flow._generate_filename("a cat", 42, None, "t2v", 0, 1)
    assert f1 == f2
    assert f1 != f3
    assert f1.endswith(".mp4")


def test_generate_filename_includes_hash(flow: InferenceBase):
    filename = flow._generate_filename("prompt text", 7, "/img.png", "i2v", 3, 2)
    assert "prompt_text" in filename
    assert filename.endswith(".mp4")
    assert "0003_002_" in filename


def test_assign_output_paths_prevents_collisions(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        samples = [
            InferenceSample(
                sample_id=f"s-{i}",
                prompt="same prompt",
                seed=1,
                mode="t2v",
                index=0,
                sample_index=i,
            )
            for i in range(3)
        ]
        # Force identical generated filenames by monkey-patching _generate_filename.
        flow._generate_filename = (
            lambda prompt, seed, image_path, mode, index, sample_index: "collision.mp4"
        )
        flow.assign_output_paths(samples, tmp)
        paths = [s.output_path for s in samples]
        assert len(set(paths)) == 3
        assert all(p.endswith(".mp4") for p in paths)
        assert "_001" in paths[1] or "_002" in paths[2]


def test_build_manifest_inputs_t2v_from_single_prompt(flow: InferenceBase):
    args = SimpleNamespace(
        num_inference_steps=None,
        ddim_steps=None,
        unconditional_guidance_scale=None,
        guidance_scale=None,
        height=None,
        width=None,
        frames=None,
        savefps=None,
        fps=None,
    )

    samples = flow.build_manifest_inputs("hello world", "t2v", args)
    assert len(samples) == 1
    assert samples[0].prompt == "hello world"
    assert samples[0].mode == "t2v"
    assert samples[0].image_path is None


def test_build_manifest_inputs_t2v_from_txt_file(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = os.path.join(tmp, "prompts.txt")
        Path(prompt_file).write_text("prompt a\n\nprompt b\n", encoding="utf-8")
        args = SimpleNamespace(
            num_inference_steps=8,
            ddim_steps=None,
            unconditional_guidance_scale=None,
            guidance_scale=None,
            height=512,
            width=512,
            frames=None,
            savefps=None,
            fps=None,
        )

        samples = flow.build_manifest_inputs(prompt_file, "t2v", args)
        assert len(samples) == 2
        assert [s.prompt for s in samples] == ["prompt a", "prompt b"]
        assert samples[0].num_inference_steps == 8
        assert samples[0].height == 512


def test_build_manifest_inputs_i2v_from_directory(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        Path(os.path.join(tmp, "prompts.txt")).write_text(
            "prompt a\nprompt b\n", encoding="utf-8"
        )
        Path(os.path.join(tmp, "a.png")).write_bytes(b"")
        Path(os.path.join(tmp, "b.png")).write_bytes(b"")
        args = SimpleNamespace(
            num_inference_steps=None,
            ddim_steps=None,
            unconditional_guidance_scale=None,
            guidance_scale=None,
            height=None,
            width=None,
            frames=None,
            savefps=None,
            fps=None,
        )

        samples = flow.build_manifest_inputs(tmp, "i2v", args)
        assert len(samples) == 2
        assert samples[0].image_path == os.path.join(tmp, "a.png")
        assert samples[1].image_path == os.path.join(tmp, "b.png")


def test_build_manifest_inputs_i2v_mismatch_raises(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        Path(os.path.join(tmp, "prompts.txt")).write_text(
            "prompt a\nprompt b\n", encoding="utf-8"
        )
        Path(os.path.join(tmp, "a.png")).write_bytes(b"")
        args = SimpleNamespace(
            num_inference_steps=None,
            ddim_steps=None,
            unconditional_guidance_scale=None,
            guidance_scale=None,
            height=None,
            width=None,
            frames=None,
            savefps=None,
            fps=None,
        )

        with pytest.raises(ValueError, match="mismatch"):
            flow.build_manifest_inputs(tmp, "i2v", args)


def test_build_manifest_inputs_per_sample_seed(flow: InferenceBase):
    args = SimpleNamespace(
        num_inference_steps=None,
        ddim_steps=None,
        unconditional_guidance_scale=None,
        guidance_scale=None,
        height=None,
        width=None,
        frames=None,
        savefps=None,
        fps=None,
    )

    samples = flow.build_manifest_inputs(
        "prompt", "t2v", args, n_samples=3, seed=10, per_sample_seed=True
    )
    assert [s.seed for s in samples] == [10, 11, 12]


def test_build_manifest_inputs_global_seed(flow: InferenceBase):
    args = SimpleNamespace(
        num_inference_steps=None,
        ddim_steps=None,
        unconditional_guidance_scale=None,
        guidance_scale=None,
        height=None,
        width=None,
        frames=None,
        savefps=None,
        fps=None,
    )

    samples = flow.build_manifest_inputs(
        "prompt", "t2v", args, n_samples=3, seed=10, per_sample_seed=False
    )
    assert [s.seed for s in samples] == [10, 10, 10]


def test_save_output_t2i(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        sample = InferenceSample(
            sample_id="t2i-0000-000-1",
            prompt="a flower",
            output_path=os.path.join(tmp, "out.jpg"),
            seed=1,
            mode="t2i",
            index=0,
            sample_index=0,
        )
        image = mock.MagicMock()
        flow.save_output(sample, image)
        image.save.assert_called_once_with(sample.output_path)


def test_save_output_video_tensor(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        sample = InferenceSample(
            sample_id="t2v-0000-000-1",
            prompt="a cat",
            output_path=os.path.join(tmp, "out.mp4"),
            seed=1,
            mode="t2v",
            index=0,
            sample_index=0,
        )
        video = torch.randn(3, 4, 8, 8)
        with mock.patch.object(flow, "save_video") as mock_save_video:
            flow.save_output(sample, video, fps=12)
        mock_save_video.assert_called_once_with(video, sample.output_path, fps=12)


def test_save_output_diffusers_frame_list(flow: InferenceBase):
    with tempfile.TemporaryDirectory() as tmp:
        sample = InferenceSample(
            sample_id="t2v-0000-000-1",
            prompt="a cat",
            output_path=os.path.join(tmp, "out.mp4"),
            seed=1,
            mode="t2v",
            index=0,
            sample_index=0,
        )
        with mock.patch("diffusers.utils.export_to_video") as mock_export:
            flow.save_output(sample, ["frame1", "frame2"], fps=15)
        mock_export.assert_called_once_with(
            ["frame1", "frame2"], sample.output_path, fps=15
        )


def test_create_manifest_includes_config(flow: InferenceBase):
    from omegaconf import OmegaConf

    args = OmegaConf.create({"seed": 42, "savedir": "/tmp"})
    manifest = flow.create_manifest(
        model_id="model", lora_path="lora", model_family="wan", mode="t2v", config=args
    )
    assert manifest.config == {"seed": 42, "savedir": "/tmp"}
