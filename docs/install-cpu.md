# CPU-only development install

VideoTuna supports CPU-only installs for **unit tests**, **config validation**, and **tiny smoke inference**. CPU is not practical for 720p / 14B video generation — use NVIDIA CUDA or AMD ROCm for production inference.

## Prerequisites

- Linux, macOS, or Windows
- Python 3.11+
- No GPU required

## Install

```bash
poetry install -E cpu --with dev
poetry run install-cpu-torch
```

`install-cpu-torch` removes CUDA/ROCm wheels and installs CPU-only `torch==2.6.0` + `torchvision==0.21.0` from `https://download.pytorch.org/whl/cpu`.

**Important:** The committed `poetry.lock` pins NVIDIA CUDA torch. Any later plain `poetry install` may restore `+cu126` wheels — re-run `poetry run install-cpu-torch` on CPU-only machines.

Verify:

```bash
poetry run verify-cpu-torch
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | Set `cpu` to force CPU even when a GPU is visible |
| `VIDEOTUNA_CPU_MODE` | `off` (default), `smoke` (tiny runs), `force` (debug init; deprecated alias: `VIDEOTUNA_ALLOW_CPU_INFERENCE=1`) |
| `VIDEOTUNA_ATTN_BACKEND` | Use `eager` or `sdpa` on CPU (`flash` is not supported) |
| `VIDEOTUNA_TORCH_COMPILE` | Keep `0` on CPU (compile is GPU-only) |

## CPU inference vs GPU + CPU offload

| | CPU-only inference | GPU inference + CPU offload |
|--|-------------------|----------------------------|
| **Purpose** | Dev, CI, smoke tests | Reduce VRAM on a GPU machine |
| **Requires GPU** | No | Yes |
| **Flags** | `--cpu-smoke`, `device: cpu` | `--enable_model_cpu_offload`, `--memory-preset low_vram` |
| **Practical for 720p 14B** | No | Yes (slow) |

CPU offload flags move weights between GPU and **host RAM**. They do not replace a GPU for large models.

## Smoke tests

```bash
export VIDEOTUNA_ATTN_BACKEND=eager
poetry run pytest tests/ -m "not gpu and not cpu_smoke" -q

# CogVideoX 2B tiny run (downloads HF weights on first use)
poetry run inference-cogvideo-t2v-diffusers \
  --config configs/inference/presets/cogvideox_2b_cpu_smoke.yaml \
  --cpu-smoke
```

Full Tier-A preset list and commands: [capability-matrix.md](capability-matrix.md).

### CPU smoke presets (`configs/inference/presets/`)

| Preset | Command |
|--------|---------|
| `cogvideox_2b_cpu_smoke.yaml` | `poetry run inference-cogvideo-t2v-diffusers --config … --cpu-smoke` |
| `cogvideox_1_5_cpu_smoke.yaml` | `poetry run inference-cogvideox1.5-t2v --config … --cpu-smoke` |
| `flux_schnell_cpu_smoke.yaml` | `poetry run inference-flux-schnell --config … --cpu-smoke` |
| `mochi_cpu_smoke.yaml` | `poetry run inference-mochi --config … --cpu-smoke` |
| `ltx_cpu_smoke.yaml` | `poetry run inference-ltx-t2v --config … --cpu-smoke` |
| `hunyuan1_5_cpu_smoke.yaml` | `poetry run inference-hunyuan1.5-t2v --config … --cpu-smoke` |
| `wan2_2_cpu_smoke.yaml` | `poetry run inference-wan2.2-t2v-720p --config … --cpu-smoke` |
| `hunyuan_init_cpu_smoke.yaml` | Native Hunyuan init-only (≤256px); `poetry run inference-hunyuan-t2v --config … --cpu-smoke` |

CogVideo SAT was removed — use Diffusers `inference-cogvideox1.5-*` for CogVideoX 1.5.

## Model tiers on CPU

| Tier | Models | Status |
|------|--------|--------|
| **cpu_ok** | Import smoke, config parse, attention/device unit tests | Always |
| **cpu_smoke** | CogVideoX 2B, Flux Schnell, Tier-A presets above (tiny H×W, few steps) | `--cpu-smoke` required |
| **gpu_required** | Production 720p+ Diffusers (Wan, Hunyuan 1.5, CogVideoX 1.5, Mochi, LTX), native Wan/Hunyuan 720p, StepVideo | Clear error; use matching `*_cpu_smoke.yaml` or native init preset |

Preset YAMLs: [`configs/inference/presets/`](../configs/inference/presets/) (`*_cpu_smoke.yaml`). See [capability-matrix.md](capability-matrix.md).

## NVIDIA install (default)

```bash
poetry install -E cuda
poetry run install-flash-attn   # optional
```

## AMD ROCm

See [install-rocm.md](install-rocm.md).

## Apple Silicon

Linux CPU install is separate from Apple Silicon. For Mac arm64, see the Docker path in the [README](../README.md).

## Troubleshooting

**`verify-cpu-torch` reports CUDA build**

Re-run `poetry run install-cpu-torch` after any `poetry install`.

**`flash` / `xformers` import errors on CPU**

Expected — these are CUDA-only optional deps. Use `VIDEOTUNA_ATTN_BACKEND=eager`.

**Offload flags rejected on CPU**

`--enable_model_cpu_offload` and `--memory-preset low_vram` need a GPU. Remove them for CPU smoke runs.

**Wan / Hunyuan / StepVideo blocked at production resolution**

Production 720p configs are `gpu_required` on CPU. Use the matching `*_cpu_smoke.yaml` preset with `--cpu-smoke`, or `hunyuan_init_cpu_smoke.yaml` for native Hunyuan init-only (not full denoise). See [capability-matrix.md](capability-matrix.md).
