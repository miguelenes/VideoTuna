# AMD ROCm install

VideoTuna supports AMD GPUs on Linux x86_64 via PyTorch ROCm wheels. ROCm uses the same `torch.cuda` API as NVIDIA CUDA (HIP backend).

## Prerequisites

- Linux x86_64
- AMD GPU (e.g. RX 7900 XTX, MI300 series)
- ROCm driver **≥ 6.2** (PyTorch 2.6 wheels target **ROCm 6.2.4**)
- Python 3.11+

Verify the driver:

```bash
rocminfo | head
```

## Install

```bash
poetry install -E rocm
poetry run install-rocm
```

`install-rocm` removes CUDA-only packages, uninstalls any existing torch/torchvision wheels, then installs matching **ROCm** builds of `torch==2.6.0` and `torchvision==0.21.0` from `https://download.pytorch.org/whl/rocm6.2.4`.

**Important:** The committed `poetry.lock` pins NVIDIA CUDA torch. Any later `poetry install` may restore `+cu126` wheels — re-run `poetry run install-rocm` on AMD machines before inference.

Verify:

```bash
poetry run python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, or `cpu` |
| `VIDEOTUNA_ATTN_BACKEND` | Use `sdpa` or `eager` on ROCm (`flash` is not supported) |
| `HIP_VISIBLE_DEVICES` | GPU selection (like `CUDA_VISIBLE_DEVICES`) |

## Smoke tests

```bash
export VIDEOTUNA_ATTN_BACKEND=sdpa
poetry run benchmark-attn-backends --num-inference-steps 2
poetry run inference-cogvideo-t2v-diffusers --num_inference_steps 2
```

## Model tiers on ROCm

| Tier | Models | Status |
|------|--------|--------|
| **A** | CogVideoX, Flux, Mochi, LTX, Hunyuan 1.5 Diffusers, Wan 2.2 Diffusers | Expected to work with `sdpa` + CPU offload |
| **B** | Native Hunyuan/Wan, Open-Sora, VideoCrafter | Experimental; no flash/xfuser/FP8 |
| **C** | StepVideo, CogVideo SAT (removed; use Diffusers 1.5), multi-GPU xfuser training | Unsupported |

See [checkpoints.md](checkpoints.md) for download links.

## NVIDIA install (default)

```bash
poetry install -E cuda
poetry run install-flash-attn   # optional
```

Training (NVIDIA only): `poetry install -E cuda --with training` then `poetry run install-deepspeed` if needed.

## CPU-only dev

```bash
poetry install -E cpu
poetry run install-cpu-torch
```

See [install-cpu.md](install-cpu.md) for smoke tests and tier matrix.

## Troubleshooting

**`torchvision::nms` / import errors after `install-rocm`**

torch and torchvision must come from the same ROCm index. If torchvision still shows `+cu126`, re-run:

```bash
poetry run install-rocm
```

**`torch.cuda.is_available()` is False**

- Confirm ROCm driver: `rocminfo`
- Re-run `poetry run install-rocm`
- Check `HIP_VISIBLE_DEVICES` is not masking all GPUs

**Out of memory**

- Use `--enable_sequential_cpu_offload`, `--enable_vae_tiling`, `--dtype bf16`
- Prefer Tier-A diffusers presets over native 720p flows

**flash-attn / xformers errors**

- ROCm does not use these packages. Set `export VIDEOTUNA_ATTN_BACKEND=sdpa`

## Lockfile note

The committed `poetry.lock` targets the `cuda` extra. ROCm users rely on `install-rocm` for PyTorch wheels. To regenerate a ROCm lock locally: `poetry lock` after editing extras (advanced).
