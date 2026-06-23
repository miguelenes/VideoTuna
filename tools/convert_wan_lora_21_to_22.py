#!/usr/bin/env python3
"""
Export Wan 2.1 native Lightning LoRA checkpoints to Diffusers safetensors.

Known limitations:
- Best-effort key remap (2.1 single denoiser → 2.2 dual-expert). Same tensors are
  written for high-noise and low-noise exports; load low-noise with
  load_into_transformer_2=True.
- Architecture deltas between Wan 2.1 native and Wan 2.2 Diffusers may reduce
  visual fidelity; run validate-domain-t2v for QA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videotuna.utils.wan_lora_bridge import (  # noqa: E402
    MIN_REMAP_COVERAGE,
    WanBridgeConfig,
    analyze_native_wan_lora_ckpt,
    build_bridge_key_map,
    export_diffusers_lora_state_dicts,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
    verify_runtime_export_parity,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Native denoiser .ckpt from train-domain-t2v",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for high_noise.safetensors and low_noise.safetensors",
    )
    parser.add_argument(
        "--mode",
        choices=("t2v", "i2v"),
        default="t2v",
        help="Wan task mode (T2V default; I2V uses same block remap)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help=f"Override minimum remap coverage (default: {MIN_REMAP_COVERAGE})",
    )
    args = parser.parse_args()

    if not is_native_wan_lora_ckpt(args.input):
        print(f"Error: not a native Wan LoRA checkpoint: {args.input}", file=sys.stderr)
        return 1

    config = (
        WanBridgeConfig(min_coverage=args.min_coverage)
        if args.min_coverage is not None
        else WanBridgeConfig()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        exports = export_diffusers_lora_state_dicts(
            args.input,
            mode=args.mode,
            bridge_config=config,
            include_key_diff=True,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    meta = analyze_native_wan_lora_ckpt(args.input)

    high_path = args.output_dir / "high_noise.safetensors"
    low_path = args.output_dir / "low_noise.safetensors"
    save_file(exports["high_noise"], high_path)
    save_file(exports["low_noise"], low_path)

    parity = verify_runtime_export_parity(args.input, config=config)

    # Write full key-diff artifact.
    native_state = load_native_wan_lora_state_dict(args.input)
    key_diff = build_bridge_key_map(native_state, config=config)
    key_diff_path = args.output_dir / "key_diff.json"
    key_diff_path.write_text(
        json.dumps(
            [
                {
                    "native_key": e.native_key,
                    "diffusers_key": e.diffusers_key,
                    "status": e.status,
                    "pattern": e.pattern,
                }
                for e in key_diff
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "source": str(args.input),
        "mode": args.mode,
        "high_noise": str(high_path),
        "low_noise": str(low_path),
        "bridge_config": {
            "min_coverage": config.min_coverage,
            "min_coverage_export": config.min_coverage_export,
            "allow_fallback_remap": config.allow_fallback_remap,
        },
        "remap_coverage": meta["remap_coverage"],
        "unmapped_keys": meta["unmapped_keys"],
        "parity": {
            "keys_match": parity.keys_match,
            "runtime_key_count": parity.runtime_key_count,
            "export_key_count": parity.export_key_count,
            "only_in_export": parity.only_in_export,
        },
        "key_diff": str(key_diff_path),
        "analysis": meta,
        "load_hint": (
            "pipeline.load_lora_weights("
            "output_dir, weight_name='high_noise.safetensors'); "
            "pipeline.load_lora_weights(..., weight_name='low_noise.safetensors', "
            "load_into_transformer_2=True)"
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported high-noise LoRA: {high_path}")
    print(f"Exported low-noise LoRA:  {low_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Key diff:  {key_diff_path}")
    print(f"Remap coverage: {meta['remap_coverage']:.1%}")
    if meta["unmapped_keys"]:
        print(
            f"Warning: {len(meta['unmapped_keys'])} unmapped keys (first 5): "
            f"{meta['unmapped_keys'][:5]}",
            file=sys.stderr,
        )
    if not parity.keys_match:
        print(
            f"Warning: runtime vs export key set mismatch — "
            f"{len(parity.only_in_export)} keys in export not in runtime. "
            f"Check key_diff.json for details.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
