# VideoTuna — Agent Instructions

## Project overview

VideoTuna is a unified Python codebase for generative video and image models: text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), and video-to-video (V2V). It supports inference and fine-tuning across Diffusers pipelines and native model flows (Wan, Hunyuan, OpenSora, Flux, CogVideoX, and others). Python 3.11+; Poetry is the default package manager (`poetry run …`), with optional uv.

## Role

You are editing a research-and-production ML repo. Optimize for:

1. **Correct behavior** — inference and training entrypoints must keep working for CUDA, ROCm, and CPU paths.
2. **Scoped diffs** — change only what the task requires; do not revert unrelated in-flight work.
3. **Portable device handling** — respect `videotuna/utils/device_utils.py` and env knobs in `.env.example`.
4. **Safe boundaries** — vendor code under `videotuna/vendor/` follows [`docs/vendor-policy.md`](docs/vendor-policy.md); never commit weights, outputs, or secrets.

Primary instruction file: this `AGENTS.md`. Cursor rules in `.cursor/rules/videotuna.mdc` link here.

## Agent workflow

1. `cd` into the VideoTuna repo root before running commands.
2. Prefer **Poetry** (`poetry run …`) unless the user explicitly uses uv.
3. Keep changes scoped — do not revert unrelated in-flight work.
4. Read [`docs/vendor-policy.md`](docs/vendor-policy.md) before touching vendored upstream code.
5. Do not commit checkpoints, `pretrained_models/`, `outputs/`, or secrets.

## Stack

| Detect | Command prefix |
|--------|----------------|
| `pyproject.toml` + `poetry.lock` | `poetry run …` |
| `uv.lock` (alternative) | `uv run …` |

## Install profiles

| Use case | Poetry | uv |
|----------|--------|-----|
| Inference NVIDIA (default) | `poetry install -E cuda` | `uv sync` |
| Inference AMD ROCm | `poetry install -E rocm` then `poetry run install-rocm` | see [`docs/install-rocm.md`](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu` then `poetry run install-cpu-torch` | see [`docs/install-rocm.md`](docs/install-rocm.md) |
| + Training | `poetry install -E cuda --with training` | `uv sync --group training` |
| + VBench eval | `poetry install --with eval` | `uv sync --group eval` |
| + Dev (pytest, ruff) | `poetry install --with dev` | `uv sync --group dev` |

## Verification (required before finishing)

Every code change **must** pass these minimum gates:

```bash
poetry run test tests/test_import_smoke.py -q   # import smoke (fast, no GPU weights)
poetry run lint                                 # ruff
```

Add targeted tests by change area (see [Testing guidance](#testing-guidance)). Run `poetry run format-check` when Python style may have drifted. Use `poetry run test -q` before large refactors or release prep.

## Commands

All Poetry scripts are defined in `pyproject.toml` under `[tool.poetry.scripts]`. Prefix every command with `poetry run` (or `uv run` when using uv).

### Dev tooling

```bash
poetry run test -q                    # full pytest suite
poetry run test tests/test_import_smoke.py -q
poetry run lint                       # ruff (E, F, C90)
poetry run format                     # apply isort + black
poetry run format-check               # check isort + black (CI)
poetry run type-check                 # mypy (optional)
poetry run coverage-report            # pytest with coverage HTML
```

### CI smoke (no GPU weights required for short-step runs)

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/cogvideox_t2v_2b.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload
poetry run pytest tests/test_inference_optimization.py tests/test_import_smoke.py -q
```

### Inference

Diffusers models use `scripts/inference_new.py` with presets under `configs/inference/`. Legacy/native models use `poetry run inference-<model>` wrappers in `scripts/__init__.py`.

```bash
poetry run inference-cogvideo-t2v-diffusers
poetry run inference-flux2-dev --enable_model_cpu_offload --num_inference_steps 4
poetry run inference-wan2.2-t2v-720p --device cuda:0
poetry run python scripts/inference_new.py --config configs/inference/cogvideox1.5_t2v_5b.yaml
```

See [`README.md`](README.md) for the full model × command matrix and [`docs/MODEL_VERSIONS.md`](docs/MODEL_VERSIONS.md) for version pins.

### Training

Requires the `training` dependency group.

```bash
poetry run train-flux-lora
poetry run train-wan2-1-t2v-lora
poetry run train-hunyuan-t2v-lora
```

### GPU / performance utilities

```bash
poetry run verify-cuda-extras
poetry run benchmark-attn-backends
poetry run install-flash-attn          # NVIDIA only
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

## Testing guidance

Tests live in `tests/`. GPU tests use `@pytest.mark.gpu` and auto-skip when no CUDA/ROCm GPU is available (`tests/conftest.py`). Prefer targeted files over the full suite during iteration.

| Change area | Minimum verification |
|-------------|---------------------|
| Any Python edit | `test_import_smoke.py` + `lint` |
| `videotuna/utils/device_utils.py`, attention, fp8 | + `test_device_utils.py` |
| Inference CLI, memory presets, optimizations | + `test_inference_optimization.py` |
| Diffusers video flow | + `test_diffusers_video_flow.py` (slow — downloads weights) |
| Flux LoRA trainer | + `test_flux_lora_train_smoke.py` (needs `--with training`) |
| Vendor / import paths | + `test_import_smoke.py` (covers module graph) |

**Fast smoke** (default, no weights): `poetry run test tests/test_import_smoke.py -q`

**CI-style smoke** (no GPU weights, short inference): see [CI smoke](#ci-smoke-no-gpu-weights-required-for-short-step-runs) below.

## Environment variables

Copy [`.env.example`](.env.example) and export as needed. Key runtime knobs:

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, `cpu` | `auto` | Override GPU backend detection |
| `VIDEOTUNA_ATTN_BACKEND` | `auto`, `flash`, `sdpa`, `eager` | `auto` | Attention implementation |
| `VIDEOTUNA_ATTN_BACKEND_STRICT` | `0`, `1` | `0` | Fail if flash requested but missing |
| `VIDEOTUNA_TORCH_COMPILE` | `0`, `1` | `0` | `torch.compile` on denoiser |
| `VIDEOTUNA_TORCH_COMPILE_MODE` | `reduce-overhead`, `max-autotune` | `reduce-overhead` | Compile mode |
| `VIDEOTUNA_METRICS_OWNER` | `script`, `flow` | `script` | Who writes `metrics.json` |
| `CUDA_VISIBLE_DEVICES` | GPU indices | all | Restrict visible NVIDIA GPUs |
| `HIP_VISIBLE_DEVICES` | GPU indices | all | Restrict visible AMD GPUs |
| `HF_TOKEN` | token | — | Hugging Face gated model access |
| `DASH_API_KEY` | key | — | DashScope prompt extension (Wan) |

## Project layout

```
videotuna/
  flow/          # Inference orchestration (Diffusers, Wan, Hunyuan, StepVideo, …)
  models/        # Native model implementations (wan/, opensora/, hunyuan/, …)
  training/      # First-party trainers (flux_lora/, …)
  utils/         # device_utils, attention, inference_cli, memory_presets
  vendor/        # Third-party snapshots (git submodule preferred)
scripts/         # CLI entrypoints (inference_new.py, train_new.py, …)
configs/         # YAML configs per model family
tests/           # pytest suite
docs/            # install guides, vendor policy, checkpoint layout
eval/            # VBench evaluation (optional `eval` group)
```

**Inference paths:** Diffusers pipelines → `videotuna/flow/diffusers_video.py` + `configs/inference/`. Native flows → `videotuna/flow/<family>.py` + `configs/00N_<family>/`.

**Training paths:** First-party trainers under `videotuna/training/`; legacy Lightning paths via `scripts/train_new.py`.

## Safety and constraints

### Never commit

- `.env`, API keys (`DASH_API_KEY`, `HF_TOKEN`), tokens, or credentials
- `checkpoints/`, `pretrained_models/`, `outputs/`, `results/`, `wandb/`, or downloaded model weights
- Large generated artifacts or debug dumps under `.jolli/` unless explicitly requested

### GPU and compute

- 720p video models need 24–80 GB VRAM depending on model and offload settings
- Low VRAM: `--enable_model_cpu_offload`, `--device-map auto`, or `configs/inference/presets/low_vram_*.yaml`
- **ROCm:** flash-attn is not supported — set `VIDEOTUNA_ATTN_BACKEND=sdpa`; do not run `install-flash-attn`
- **CPU:** use `poetry install -E cpu` then `poetry run install-cpu-torch`; expect slow inference

### Code and vendor boundaries

- New upstream snapshots go under `videotuna/vendor/<name>/` with `VENDOR.md` (pinned SHA, license, entrypoints). See [`docs/vendor-policy.md`](docs/vendor-policy.md).
- Do not edit vendored upstream unless the task explicitly requires a minimal patch; prefer submodule pins.
- CogVideo SAT paths are removed; use Diffusers CogVideoX 1.5 (`inference-cogvideox1.5-*`).

### Git and releases

- Do not force-push `main`
- Do not amend commits unless the user explicitly requests it
- Do not commit unless the user explicitly asks
- Run [verification gates](#verification-required-before-finishing) before declaring work complete

## MCP

No project-specific MCP servers are required. Optional workspace-level MCP (mem0, Context7, etc.) is configured at the user/workspace level, not in this repo. See [`.cursor/mcp.json`](.cursor/mcp.json).

## Related docs

| Doc | Topic |
|-----|-------|
| [`README.md`](README.md) | Install, inference commands, upgrade notes |
| [`docs/checkpoints.md`](docs/checkpoints.md) | Checkpoint download and layout |
| [`docs/MODEL_VERSIONS.md`](docs/MODEL_VERSIONS.md) | Model version matrix |
| [`docs/install-rocm.md`](docs/install-rocm.md) | AMD ROCm setup |
| [`docs/multi-gpu.md`](docs/multi-gpu.md) | Multi-GPU and device-map |
| [`docs/vendor-policy.md`](docs/vendor-policy.md) | Vendored upstream policy |
