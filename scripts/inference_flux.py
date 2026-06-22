"""Deprecated: use scripts/inference_new.py --config configs/inference/flux_dev.yaml"""

import os
import sys

sys.path.insert(0, os.getcwd())

from videotuna.utils.diffusers_inference_shim import run_diffusers_inference

if __name__ == "__main__":
    config = "configs/inference/flux_dev.yaml"
    extra = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--model_type" and i + 1 < len(argv):
            if argv[i + 1] == "schnell":
                config = "configs/inference/flux_schnell.yaml"
            i += 2
            continue
        if arg == "--prompt" and i + 1 < len(argv):
            extra.extend(["--prompt_file", argv[i + 1]])
            i += 2
            continue
        if arg == "--out_path" and i + 1 < len(argv):
            extra.extend(["--savedir", argv[i + 1]])
            i += 2
            continue
        if arg == "--guidance_scale" and i + 1 < len(argv):
            extra.extend(["--unconditional_guidance_scale", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("--") and i + 1 < len(argv) and arg not in (
            "--enable_vae_tiling",
            "--enable_vae_slicing",
            "--enable_model_cpu_offload",
            "--enable_sequential_cpu_offload",
            "--compile",
            "--fuse_qkv",
            "--enable_attention_cache",
        ):
            extra.extend([arg, argv[i + 1]])
            i += 2
            continue
        extra.append(arg)
        i += 1
    sys.exit(run_diffusers_inference(config, extra))
