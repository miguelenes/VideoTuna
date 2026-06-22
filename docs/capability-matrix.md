# Tier-A inference capability matrix

Cross-platform support for **Tier A** Diffusers models (CUDA / ROCm / CPU smoke). For checkpoint download links see [checkpoints.md](checkpoints.md). For version pins see [MODEL_VERSIONS.md](MODEL_VERSIONS.md).

**Attention backends** (via `VIDEOTUNA_ATTN_BACKEND`):

| Backend | CUDA | ROCm | CPU |
|---------|------|------|-----|
| `auto` | `flash` if installed, else `sdpa` | `sdpa` | `eager` |
| `flash` | yes (optional `install-flash-attn`) | **blocked** | **blocked** |
| `sdpa` | yes | yes (recommended) | falls back to `eager` |
| `eager` | yes | yes | yes (recommended for CPU CI) |

CPU smoke uses `--cpu-smoke` (sets `VIDEOTUNA_CPU_MODE=smoke`, `VIDEOTUNA_ATTN_BACKEND=eager`, caps resolution/steps). GPU offload flags require an accelerator — they are not CPU-only modes.

## T2V / T2I models

| Model | Production preset | CUDA command | ROCm preset + attn | CPU smoke preset | Attn CUDA | Attn ROCm | Attn CPU |
|-------|-------------------|--------------|--------------------|------------------|-----------|-----------|----------|
| CogVideoX 2B | `configs/inference/cogvideox_t2v_2b.yaml` | `poetry run inference-cogvideo-t2v-diffusers` | same + `VIDEOTUNA_ATTN_BACKEND=sdpa` | `presets/cogvideox_2b_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |
| CogVideoX 1.5 T2V | `configs/inference/cogvideox1.5_t2v_5b.yaml` | `poetry run inference-cogvideox1.5-t2v` | same + offload + `sdpa` | `presets/cogvideox_1_5_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |
| Flux 1 Schnell | `configs/inference/flux1_schnell.yaml` | `poetry run inference-flux-schnell` | same + `sdpa` | `presets/flux_schnell_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |
| Flux 2-dev T2I | `configs/inference/flux_dev.yaml` | `poetry run inference-flux2-dev` | same + offload + `sdpa` | `--cpu-smoke` caps main preset | auto→flash/sdpa | `sdpa` | `eager` |
| Mochi T2V | `configs/inference/mochi_t2v.yaml` | `poetry run inference-mochi` | same + offload + `sdpa` | `presets/mochi_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |
| LTX-Video T2V | `configs/inference/ltx_video.yaml` | `poetry run inference-ltx-t2v` | same + offload + `sdpa` | `presets/ltx_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |
| Hunyuan 1.5 T2V (Diffusers) | `configs/inference/hunyuanvideo1.5_t2v_720p.yaml` | `poetry run inference-hunyuan1.5-t2v` | same + offload + `sdpa` | `presets/hunyuan1_5_cpu_smoke.yaml` | diffusers flash_hub / native | `sdpa` | `eager` |
| Wan 2.2 T2V (Diffusers) | `configs/inference/wan2_2_t2v_a14b.yaml` | `poetry run inference-wan2.2-t2v-720p` | `presets/wan2_2_cpu_smoke.yaml` or 720p + offload + `sdpa` | `presets/wan2_2_cpu_smoke.yaml` | auto→flash/sdpa | `sdpa` | `eager` |

### I2V variants

| Model | Production preset | Poetry command |
|-------|-------------------|----------------|
| CogVideoX 1.5 I2V | `configs/inference/cogvideox1.5_i2v_5b.yaml` | `poetry run inference-cogvideox1.5-i2v` |
| Hunyuan 1.5 I2V | `configs/inference/hunyuanvideo1.5_i2v_720p.yaml` | `poetry run inference-hunyuan1.5-i2v` |
| Wan 2.2 I2V | `configs/inference/wan2_2_i2v_a14b.yaml` | `poetry run inference-wan2.2-i2v-720p` |

720p I2V presets are `gpu_required` on CPU without `--cpu-smoke`. Use tiny Diffusers smoke presets or `VIDEOTUNA_CPU_MODE=force` only for native-flow init debug.

## Memory presets (GPU)

| Model | Low VRAM | Balanced | Max speed |
|-------|----------|----------|-----------|
| Wan 2.2 720p | `presets/low_vram_wan2_2_720p.yaml` | `presets/balanced_wan2_2_720p.yaml` | `presets/max_speed_wan2_2_720p.yaml` |
| Hunyuan 1.5 720p | — | `presets/balanced_hunyuan1_5_720p.yaml` | — |
| CogVideoX | — | — | `presets/max_speed_cogvideox.yaml` |

Pass `--memory-preset low_vram|balanced|max_speed` or set in YAML. Requires a GPU.

## Native vs Diffusers Hunyuan (CPU)

| Path | Preset | Purpose |
|------|--------|---------|
| Diffusers 1.5 | `presets/hunyuan1_5_cpu_smoke.yaml` | Tiny Diffusers smoke on CPU |
| Native legacy | `presets/hunyuan_init_cpu_smoke.yaml` | Init-only checkpoint load (≤256px, ≤2 frames) with `--cpu-smoke` |

CogVideo SAT inference was removed — use Diffusers `inference-cogvideox1.5-*` only.

## Canonical smoke commands

### CPU dev

```bash
poetry install -E cpu --with dev
poetry run install-cpu-torch
poetry run verify-cpu-torch
export VIDEOTUNA_ATTN_BACKEND=eager
poetry run pytest tests/ -m "not gpu and not cpu_smoke" -q
poetry run inference-cogvideo-t2v-diffusers \
  --config configs/inference/presets/cogvideox_2b_cpu_smoke.yaml --cpu-smoke
```

### AMD ROCm

```bash
poetry install -E rocm
poetry run install-rocm
export VIDEOTUNA_ATTN_BACKEND=sdpa
poetry run inference-cogvideo-t2v-diffusers --num_inference_steps 2
poetry run inference-flux-schnell \
  --config configs/inference/presets/flux_schnell_cpu_smoke.yaml --cpu-smoke
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --num_inference_steps 2 --enable_model_cpu_offload
```

### NVIDIA CI smoke

From [MODEL_VERSIONS.md](MODEL_VERSIONS.md):

```bash
poetry install -E cuda --with dev
poetry run python scripts/inference_new.py \
  --config configs/inference/cogvideox_t2v_2b.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload
poetry run pytest tests/test_inference_optimization.py tests/test_import_smoke.py -q
```

## Tier B / C (reference)

| Tier | Models | ROCm | CPU |
|------|--------|------|-----|
| B | Native Hunyuan/Wan, Open-Sora, VideoCrafter | Experimental | Init smoke only |
| C | StepVideo, CogVideo SAT (removed), xfuser multi-GPU training | Unsupported | No |

See [install-rocm.md](install-rocm.md) and [install-cpu.md](install-cpu.md).
