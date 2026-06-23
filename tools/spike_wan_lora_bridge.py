#!/usr/bin/env python3
"""GPU/CPU spike: inventory native Wan LoRA keys and optional smoke inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videotuna.testing.wan_lora_ckpt import build_synthetic_wan_lora_ckpt  # noqa: E402
from videotuna.utils.wan_lora_bridge import (  # noqa: E402
    MIN_REMAP_COVERAGE,
    WanBridgeConfig,
    analyze_native_wan_lora_ckpt,
    apply_native_wan_lora_to_pipeline,
    build_bridge_key_map,
    compute_remap_coverage,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
    verify_runtime_export_parity,
)


def _build_synthetic_ckpt(path: Path, *, num_blocks: int = 2, rank: int = 16) -> None:
    """Write a synthetic denoiser ckpt with production-style key names."""
    import torch

    build_synthetic_wan_lora_ckpt(path, num_blocks=num_blocks, rank=rank)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    count = len(raw.get("state_dict", {}))
    print(f"Wrote synthetic checkpoint: {path} ({count} tensors)")


def _inventory_only(ckpt: Path, *, min_coverage: float) -> int:
    native = load_native_wan_lora_state_dict(ckpt)
    transformed, total, coverage = compute_remap_coverage(native)
    info = analyze_native_wan_lora_ckpt(ckpt)
    info["remap_coverage"] = coverage
    print(json.dumps(info, indent=2))
    print(f"Remap coverage: {transformed}/{total} keys transformed ({coverage:.1%})")
    return 0 if coverage >= min_coverage else 1


def _key_diff_report(ckpt: Path, *, config: WanBridgeConfig) -> int:
    """Print full per-key remap status table."""
    native_state = load_native_wan_lora_state_dict(ckpt)
    key_map = build_bridge_key_map(native_state, config=config)

    remapped = [e for e in key_map if e.status == "remapped"]
    fallback_ = [e for e in key_map if e.status == "fallback"]
    unmapped = [e for e in key_map if e.status == "unmapped"]

    print(f"\nKey diff ({len(key_map)} total keys):")
    print(f"  Remapped:  {len(remapped)}")
    print(f"  Fallback:  {len(fallback_)}")
    print(f"  Unmapped:  {len(unmapped)}")
    print(f"  Coverage:  {len(remapped) / len(key_map):.1%}")
    print()

    for e in key_map:
        dst = e.diffusers_key or "(dropped)"
        line = f"  [{e.status:>9}]  {e.native_key}  →  {dst}"
        if e.pattern:
            line += f"  ({e.pattern})"
        print(line)

    return 0 if len(remapped) / len(key_map) >= config.min_coverage else 1


def _parity_check(ckpt: Path, *, config: WanBridgeConfig) -> int:
    """Compare runtime bridge remap with offline export remap."""
    parity = verify_runtime_export_parity(ckpt, config=config)
    print("\nParity check (runtime vs offline export):")
    print(f"  Keys match:         {parity.keys_match}")
    print(f"  Runtime keys:       {parity.runtime_key_count}")
    print(f"  Export keys:        {parity.export_key_count}")
    if parity.only_in_export:
        print(f"\n  Keys in export but NOT runtime ({len(parity.only_in_export)}):")
        for k in parity.only_in_export:
            print(f"    {k}")
    return 0 if parity.keys_match else 1


def _expert_map(ckpt: Path) -> int:
    """Show key-to-expert assignment (high-noise / low-noise / both)."""
    native_state = load_native_wan_lora_state_dict(ckpt)
    remapped, unmapped, _ = (
        apply_native_wan_lora_to_pipeline(None, ckpt) if False else (None, None, None)
    )
    # In inventory mode: both experts receive identical remapped keys.
    key_map = build_bridge_key_map(native_state)
    remapped_count = sum(1 for e in key_map if e.status == "remapped")
    unmapped_count = sum(1 for e in key_map if e.status == "unmapped")

    print(f"\nExpert mapping ({len(key_map)} total keys):")
    print(f"  Both experts (high + low): {remapped_count}")
    print(f"  No expert (unmapped):      {unmapped_count}")
    print()
    for e in key_map:
        expert = "high+low" if e.status == "remapped" else "—"
        line = f"  [{expert:>9}]  {e.native_key}"
        if e.diffusers_key:
            line += f"  →  {e.diffusers_key}"
        print(line)
    return 0


def _load_on_pipeline(ckpt: Path, model_id: str, *, config: WanBridgeConfig) -> int:
    import torch
    from diffusers import WanPipeline

    from videotuna.utils.device_utils import resolve_inference_device

    device = resolve_inference_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading pipeline {model_id} (dtype={dtype})...")
    pipeline = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
    reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt, bridge_config=config)
    for report in reports:
        print(json.dumps(report.as_dict(), indent=2))
        if report.loaded_lora_params == 0:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Spike: Wan 2.1 native LoRA → 2.2 Diffusers bridge inventory/load test."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Native denoiser .ckpt path (or omit with --synthetic)",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        metavar="PATH",
        help="Write a synthetic production-key ckpt to PATH and use it",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_REMAP_COVERAGE,
        help=f"Minimum remap coverage threshold (default: {MIN_REMAP_COVERAGE})",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print full per-key remap status table",
    )
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="Compare runtime bridge remap vs offline export remap",
    )
    parser.add_argument(
        "--expert-map",
        action="store_true",
        help="Show which keys go to high-noise / low-noise / both",
    )
    parser.add_argument(
        "--load-pipeline",
        action="store_true",
        help="Load Wan 2.2 pipeline and apply bridge (requires GPU + weights)",
    )
    parser.add_argument(
        "--model-id",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers model id for --load-pipeline",
    )
    args = parser.parse_args()

    config = WanBridgeConfig(min_coverage=args.min_coverage)
    ckpt = args.input
    if args.synthetic:
        _build_synthetic_ckpt(args.synthetic)
        ckpt = args.synthetic

    if ckpt is None:
        parser.error("Provide --input or --synthetic")

    if not is_native_wan_lora_ckpt(ckpt):
        print(f"Not a native Wan LoRA checkpoint: {ckpt}", file=sys.stderr)
        return 1

    if args.diff:
        return _key_diff_report(ckpt, config=config)
    if args.check_parity:
        return _parity_check(ckpt, config=config)
    if args.expert_map:
        return _expert_map(ckpt)

    rc = _inventory_only(ckpt, min_coverage=args.min_coverage)
    if args.load_pipeline:
        rc = _load_on_pipeline(ckpt, args.model_id, config=config) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
