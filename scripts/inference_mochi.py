"""Deprecated: use scripts/inference_new.py --config configs/inference/mochi_t2v.yaml"""

import os
import sys

sys.path.insert(0, os.getcwd())

from videotuna.utils.diffusers_inference_shim import run_diffusers_inference

if __name__ == "__main__":
    config = "configs/inference/mochi_t2v.yaml"
    extra = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--ckpt_path", "--model_path") and i + 1 < len(argv):
            extra.extend(["--ckpt_path", argv[i + 1]])
            i += 2
            continue
        if arg == "--prompt_file" and i + 1 < len(argv):
            extra.extend(["--prompt_file", argv[i + 1]])
            i += 2
            continue
        if arg == "--savedir" and i + 1 < len(argv):
            extra.extend(["--savedir", argv[i + 1]])
            i += 2
            continue
        if arg == "--fps" and i + 1 < len(argv):
            extra.extend(["--savefps", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("--") and i + 1 < len(argv):
            extra.extend([arg, argv[i + 1]])
            i += 2
            continue
        extra.append(arg)
        i += 1
    sys.exit(run_diffusers_inference(config, extra))
