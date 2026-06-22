"""Tier-A Diffusers inference compatibility audit (no GPU weights)."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from videotuna.utils import device_utils
from videotuna.utils.inference_cli import apply_cpu_smoke_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "configs" / "inference" / "presets"
DIFFUSERS_FLOW = device_utils._DIFFUSERS_FLOW

CPU_SMOKE_CAPS = {
    "frames": 2,
    "height": 256,
    "width": 256,
    "num_inference_steps": 4,
}

TIER_A_PRODUCTION = [
    pytest.param(
        REPO_ROOT / "configs/inference/cogvideox_t2v_2b.yaml",
        "cogvideox_2b_cpu_smoke.yaml",
        "cpu_smoke",
        id="cogvideox-2b",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/cogvideox1.5_t2v_5b.yaml",
        "cogvideox_1_5_cpu_smoke.yaml",
        "gpu_required",
        id="cogvideox-1.5",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/flux1_schnell.yaml",
        "flux_schnell_cpu_smoke.yaml",
        "cpu_smoke",
        id="flux-schnell",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/flux_dev.yaml",
        None,
        "gpu_required",
        id="flux-2-dev",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/mochi_t2v.yaml",
        "mochi_cpu_smoke.yaml",
        "gpu_required",
        id="mochi",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/ltx_video.yaml",
        "ltx_cpu_smoke.yaml",
        "gpu_required",
        id="ltx",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/hunyuanvideo1.5_t2v_720p.yaml",
        "hunyuan1_5_cpu_smoke.yaml",
        "gpu_required",
        id="hunyuan-1.5-diffusers",
    ),
    pytest.param(
        REPO_ROOT / "configs/inference/wan2_2_t2v_a14b.yaml",
        "wan2_2_cpu_smoke.yaml",
        "gpu_required",
        id="wan-2.2-diffusers",
    ),
]


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "prod_path,smoke_preset,expected_tier",
    TIER_A_PRODUCTION,
)
def test_tier_a_production_config_tier(
    prod_path: Path,
    smoke_preset: str | None,
    expected_tier: str,
):
    cfg = _load_config(prod_path)
    flow = cfg["flow"]
    inf = cfg["inference"]
    tier = device_utils.get_flow_tier(
        flow["target"],
        model_family=flow.get("params", {}).get("model_family"),
        model_variant=flow.get("params", {}).get("model_variant"),
        height=inf.get("height"),
        width=inf.get("width"),
    )
    assert tier == expected_tier
    if smoke_preset is not None:
        smoke_path = PRESETS_DIR / smoke_preset
        assert smoke_path.exists(), f"Missing CPU smoke preset: {smoke_preset}"


@pytest.mark.parametrize(
    "preset_name",
    [
        "cogvideox_2b_cpu_smoke.yaml",
        "cogvideox_1_5_cpu_smoke.yaml",
        "flux_schnell_cpu_smoke.yaml",
        "mochi_cpu_smoke.yaml",
        "ltx_cpu_smoke.yaml",
        "hunyuan1_5_cpu_smoke.yaml",
        "wan2_2_cpu_smoke.yaml",
    ],
)
def test_cpu_smoke_preset_within_caps(preset_name: str):
    path = PRESETS_DIR / preset_name
    cfg = OmegaConf.load(path)
    assert cfg.inference.device == "cpu"
    inf = OmegaConf.create(OmegaConf.to_container(cfg.inference, resolve=True))
    flow = OmegaConf.create(OmegaConf.to_container(cfg.flow, resolve=True))
    apply_cpu_smoke_limits(inf, flow)
    assert int(inf.height) <= CPU_SMOKE_CAPS["height"]
    assert int(inf.width) <= CPU_SMOKE_CAPS["width"]
    steps = getattr(inf, "num_inference_steps", None)
    if steps is None:
        steps = getattr(inf, "ddim_steps", None)
    if steps is not None:
        assert int(steps) <= CPU_SMOKE_CAPS["num_inference_steps"]
    frames = getattr(inf, "frames", None)
    if frames is not None:
        assert int(frames) <= CPU_SMOKE_CAPS["frames"]
    assert cfg.flow.target == DIFFUSERS_FLOW
