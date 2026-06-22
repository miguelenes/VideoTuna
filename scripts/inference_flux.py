import argparse
import os

import torch
from diffusers import FluxPipeline

from videotuna.utils.common_utils import monitor_resources, save_metrics
from videotuna.utils.inference_cli import (
    add_standard_inference_flags,
    apply_compile_env,
)
from videotuna.utils.inference_utils import load_prompts_from_txt


def inference(args):
    apply_compile_env(bool(getattr(args, "compile", False)))
    flux_dtype = (
        torch.float16 if getattr(args, "dtype", None) == "fp16" else torch.bfloat16
    )
    if args.model_type == "dev":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", dtype=flux_dtype
        )
    elif args.model_type == "schnell":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", dtype=flux_dtype
        )
    else:
        raise ValueError("model_type must be either 'dev' or 'schnell'")

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if args.enable_vae_slicing:
        pipe.vae.enable_slicing()
    if args.enable_vae_tiling:
        pipe.vae.enable_tiling()
    if not args.enable_sequential_cpu_offload and not args.enable_model_cpu_offload:
        pipe.to(flux_dtype)
    if args.prompt.endswith(".txt"):
        # model_input is a file for t2i
        prompts = load_prompts_from_txt(prompt_file=args.prompt)
        os.makedirs(args.out_path, exist_ok=True)
        out_paths = [
            os.path.join(args.out_path, f"{i:05d}_{prompts[i]}.jpg")
            for i in range(len(prompts))
        ]
    else:
        prompts = [prompt]
        out_paths = [args.out_path]
    per_sample = []
    for prompt, out_path in zip(prompts, out_paths):
        result_with_metrics = generate(args, pipe, prompt)
        out = result_with_metrics["result"]
        per_sample.append(result_with_metrics)
        out.save(out_path)
    save_metrics(
        metrics={"per_sample": per_sample, "frames": 1},
        savedir=args.out_path,
        config=args,
    )


@monitor_resources(return_metrics=True)
def generate(args, pipe, prompt):
    out = pipe(
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=256,
    ).images[0]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="dev", choices=["dev", "schnell"]
    )
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("--out_path", type=str, default="./image.png")
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    add_standard_inference_flags(parser, include_fp8=False)
    args = parser.parse_args()
    inference(args)
