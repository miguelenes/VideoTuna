"""Deprecated: use scripts/inference_new.py --config configs/inference/flux_dev.yaml --lorackpt ..."""

import os
import sys

sys.path.insert(0, os.getcwd())

from videotuna.utils.diffusers_inference_shim import run_diffusers_inference

if __name__ == "__main__":
    config = "configs/inference/flux_dev.yaml"
    extra = [
        "--enable_sequential_cpu_offload",
        "--enable_vae_tiling",
        "--enable_vae_slicing",
        "--dtype",
        "fp16",
    ]
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--prompt" and i + 1 < len(argv):
            extra.extend(["--prompt_file", argv[i + 1]])
            i += 2
            continue
        if arg == "--out_path" and i + 1 < len(argv):
            extra.extend(["--savedir", argv[i + 1]])
            i += 2
            continue
        if arg == "--lora_path" and i + 1 < len(argv):
            extra.extend(["--lorackpt", argv[i + 1]])
            i += 2
            continue
        if arg == "--guidance_scale" and i + 1 < len(argv):
            extra.extend(["--unconditional_guidance_scale", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("--") and i + 1 < len(argv):
            extra.extend([arg, argv[i + 1]])
            i += 2
            continue
        extra.append(arg)
        i += 1
    sys.exit(run_diffusers_inference(config, extra))
