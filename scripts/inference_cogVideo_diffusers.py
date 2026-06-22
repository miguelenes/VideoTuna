"""Deprecated: use scripts/inference_new.py --config configs/inference/cogvideox_t2v_2b.yaml"""

import os
import sys

sys.path.insert(0, os.getcwd())

from videotuna.utils.diffusers_inference_shim import run_diffusers_inference

if __name__ == "__main__":
    config = "configs/inference/cogvideox_t2v_2b.yaml"
    extra = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--generate_type" and i + 1 < len(argv):
            mode = argv[i + 1]
            if mode == "i2v":
                config = "configs/inference/cogvideox_i2v_5b.yaml"
            elif mode == "t2v":
                config = "configs/inference/cogvideox_t2v_5b.yaml"
            i += 2
            continue
        if arg in ("--model_path", "--ckpt_path") and i + 1 < len(argv):
            extra.extend(["--ckpt_path", argv[i + 1]])
            i += 2
            continue
        if arg == "--output_path" and i + 1 < len(argv):
            extra.extend(["--savedir", argv[i + 1]])
            i += 2
            continue
        if arg == "--model_input" and i + 1 < len(argv):
            path = argv[i + 1]
            if path.endswith(".txt"):
                extra.extend(["--prompt_file", path])
            else:
                extra.extend(["--prompt_dir", path])
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
