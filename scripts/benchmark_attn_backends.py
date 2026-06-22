#!/usr/bin/env python3
"""
Benchmark attention backends on a small CogVideoX diffusers inference smoke run.

Example:
    poetry run benchmark-attn-backends
    VIDEOTUNA_ATTN_BACKEND=sdpa poetry run benchmark-attn-backends --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from diffusers import CogVideoXPipeline

from videotuna.utils.attention import (
    apply_diffusers_attention_backend,
    is_flash_attn_available,
)


def _run_backend(
    backend: str,
    model_path: str,
    prompt: str,
    num_inference_steps: int,
    seed: int,
) -> Dict[str, Any]:
    os.environ["VIDEOTUNA_ATTN_BACKEND"] = backend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for attention backend benchmarks.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    apply_diffusers_attention_backend(pipe.transformer)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Warm-up (excludes compile / first-kernel overhead from timed region).
    _ = pipe(
        prompt=prompt,
        num_inference_steps=1,
        generator=generator,
        output_type="latent",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.perf_counter()
    _ = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="latent",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "backend": backend,
        "seconds": round(elapsed, 3),
        "peak_vram_gb": round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
        "num_inference_steps": num_inference_steps,
        "model_path": model_path,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark VideoTuna attention backends.")
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
        help="Backends to test (default: eager sdpa flash when available).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args(argv)

    backends = args.backends or ["eager", "sdpa"]
    if is_flash_attn_available() and "flash" not in backends:
        backends.append("flash")

    results: List[Dict[str, Any]] = []
    for backend in backends:
        print(f"Running backend={backend} ...", file=sys.stderr)
        try:
            results.append(
                _run_backend(
                    backend=backend,
                    model_path=args.model_path,
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                )
            )
        except Exception as exc:
            results.append({"backend": backend, "error": str(exc)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n| Backend | Seconds | Peak VRAM (GB) |")
        print("| --- | ---: | ---: |")
        for row in results:
            if "error" in row:
                print(f"| {row['backend']} | ERROR | {row['error']} |")
            else:
                vram = row["peak_vram_gb"]
                vram_str = f"{vram:.3f}" if vram is not None else "n/a"
                print(f"| {row['backend']} | {row['seconds']:.3f} | {vram_str} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
