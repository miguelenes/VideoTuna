#!/usr/bin/env python3
"""
Benchmark attention backends on a small CogVideoX diffusers inference smoke run.

Example:
    poetry run benchmark-attn-backends
    poetry run benchmark-attn-backends --json-out results/bench_attn.json
    VIDEOTUNA_ATTN_BACKEND=sdpa poetry run benchmark-attn-backends --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers import CogVideoXPipeline

from videotuna.utils.attention import (
    apply_diffusers_attention_backend,
    is_flash_attn_available,
)
from videotuna.utils.device_utils import (
    detect_compute_backend,
    empty_accelerator_cache,
    gpu_is_available,
    resolve_inference_device,
    synchronize_accelerator,
)


def _verify_torch_vision_stack() -> None:
    """Fail fast when torch and torchvision are from different accelerator builds."""
    import torch
    import torchvision

    torch_build = torch.__version__
    tv_build = torchvision.__version__
    hip = getattr(torch.version, "hip", None)
    if hip is not None and "+cu" in tv_build:
        raise RuntimeError(
            f"torch/torchvision build mismatch: torch={torch_build} (ROCm), "
            f"torchvision={tv_build} (CUDA). Run: poetry run install-rocm"
        )
    if hip is None and "+rocm" in torch_build.lower():
        raise RuntimeError(
            f"torch reports ROCm build ({torch_build}) but HIP is unavailable. "
            "Run: poetry run install-rocm"
        )


def _compute_capability() -> str | None:
    if not gpu_is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _run_backend(
    backend: str,
    model_path: str,
    prompt: str,
    num_inference_steps: int,
    seed: int,
    compute_backend: str,
    height: int | None = None,
    width: int | None = None,
    num_frames: int = 49,
) -> Dict[str, Any]:
    os.environ["VIDEOTUNA_ATTN_BACKEND"] = backend

    if not gpu_is_available():
        raise RuntimeError(
            "A GPU accelerator (NVIDIA CUDA or AMD ROCm) is required for benchmarks."
        )

    device = resolve_inference_device()
    empty_accelerator_cache()
    torch.cuda.reset_peak_memory_stats()

    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    apply_diffusers_attention_backend(pipe.transformer)

    generator = torch.Generator(device=device).manual_seed(seed)

    pipe_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": 1,
        "generator": generator,
        "output_type": "latent",
    }
    if height is not None:
        pipe_kwargs["height"] = height
    if width is not None:
        pipe_kwargs["width"] = width
    if height is not None:
        pipe_kwargs["num_frames"] = num_frames

    # Warm-up (excludes compile / first-kernel overhead from timed region).
    _ = pipe(**pipe_kwargs)

    synchronize_accelerator()
    torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    _ = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="latent",
        **{k: v for k, v in pipe_kwargs.items() if k not in ("prompt", "num_inference_steps", "generator", "output_type")},
    )
    synchronize_accelerator()
    elapsed = time.perf_counter() - start

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    frames_per_sec = round(num_frames / elapsed, 3) if elapsed > 0 and height else None

    del pipe
    empty_accelerator_cache()

    result: Dict[str, Any] = {
        "backend": backend,
        "compute_backend": compute_backend,
        "seconds": round(elapsed, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "num_inference_steps": num_inference_steps,
        "model_path": model_path,
        "compute_capability": _compute_capability(),
    }
    if height is not None:
        result["height"] = height
        result["width"] = width
        result["num_frames"] = num_frames
        result["frames_per_sec"] = frames_per_sec
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark VideoTuna attention backends."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VIDEOTUNA_BENCH_MODEL", "THUDM/CogVideoX-2b"),
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A cat riding a bicycle through a sunny park.",
        help="Short prompt for the smoke benchmark.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Diffusion steps for the timed run (after warm-up).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to test (default: eager sdpa; flash on CUDA when available).",
    )
    parser.add_argument(
        "--resolutions",
        default=None,
        help="Comma-separated heights for a resolution matrix (width keeps 16:9 aspect).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write JSON results to this file path.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of a table."
    )
    args = parser.parse_args(argv)

    _verify_torch_vision_stack()
    compute_backend = detect_compute_backend()
    backends = args.backends or ["eager", "sdpa"]
    if (
        compute_backend == "cuda"
        and is_flash_attn_available()
        and "flash" not in backends
    ):
        backends.append("flash")

    heights: List[int | None] = [None]
    if args.resolutions:
        heights = [int(h.strip()) for h in args.resolutions.split(",") if h.strip()]

    results: List[Dict[str, Any]] = []
    for height in heights:
        width = int(height * 16 / 9) if height else None
        for backend in backends:
            label = backend if height is None else f"{backend}@{height}p"
            print(f"Running backend={label} ({compute_backend}) ...", file=sys.stderr)
            try:
                results.append(
                    _run_backend(
                        backend=backend,
                        model_path=args.model_path,
                        prompt=args.prompt,
                        num_inference_steps=args.num_inference_steps,
                        seed=args.seed,
                        compute_backend=compute_backend,
                        height=height,
                        width=width,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "backend": backend,
                        "compute_backend": compute_backend,
                        "height": height,
                        "error": str(exc),
                    }
                )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nCompute backend: {compute_backend}\n")
        print("| Backend | Seconds | Peak VRAM (GB) | Frames/s |")
        print("| --- | ---: | ---: | ---: |")
        for row in results:
            if "error" in row:
                print(f"| {row['backend']} | ERROR | {row['error']} | |")
            else:
                vram = row["peak_vram_gb"]
                vram_str = f"{vram:.3f}" if vram is not None else "n/a"
                fps = row.get("frames_per_sec")
                fps_str = f"{fps:.3f}" if fps is not None else "n/a"
                label = row["backend"]
                if row.get("height"):
                    label = f"{label} ({row['height']}p)"
                print(f"| {label} | {row['seconds']:.3f} | {vram_str} | {fps_str} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
