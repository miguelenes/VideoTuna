# PrivTune — Development

## Prerequisites

- **Python 3.11+** (`.python-version` = `3.11`)
- **Poetry** (default) — install via `pip install poetry`
- **uv** (optional alternative)
- **CUDA-capable GPU** (recommended for training; CPU-only for CI/dev validation)

## Install

### CUDA + training (default, recommended)

```shell
conda create -n privtune python=3.11 -y
conda activate privtune
pip install poetry
poetry install -E cuda --with training
poetry run install-deepspeed   # required for Wan LoRA (rebuilds DeepSpeed)
```

### CPU dev / CI (no GPU)

```shell
poetry install -E cpu --with dev --with training
poetry run install-cpu-torch   # replaces CUDA torch with CPU torch
poetry run verify-cpu-torch    # verify the swap worked
```

### AMD ROCm (inference + Flux training only)

```shell
poetry install -E rocm --with training
poetry run install-rocm
```
Wan training requires CUDA. See `docs/install-rocm.md`. ROCm must set `VIDEOTUNA_ATTN_BACKEND=sdpa`.

### + Dev tooling

Add `--with dev` to any of the above to get pytest, ruff, mypy, pre-commit.

### Install profiles quick reference

| Use case | Command |
|----------|---------|
| CUDA + training | `poetry install -E cuda --with training` + `poetry run install-deepspeed` |
| CUDA + training + dev | `poetry install -E cuda --with training --with dev` + `poetry run install-deepspeed` |
| CPU dev / CI | `poetry install -E cpu --with dev --with training` + `poetry run install-cpu-torch` |
| ROCm inference | `poetry install -E rocm --with training` + `poetry run install-rocm` |
| Optional video decode | add `-E video-fast` (torchcodec) |
| Optional Flux tracking | add `-E trackio` |

### Docker

```shell
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
docker compose build privtune
docker compose run --rm privtune bash
# inside container: poetry install -E cuda --with training
```

## Environment variables

All `VIDEOTUNA_*` prefix. See `.env.example` for full list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | Override: `cuda`, `rocm`, `cpu` |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` | Override: `flash`, `sdpa`, `eager` |
| `VIDEOTUNA_TORCH_COMPILE` | `0` | Enable torch.compile on denoiser |
| `VIDEOTUNA_CPU_MODE` | `off` | `smoke` forces CPU inference mode |
| `HF_TOKEN` | — | Required for gated Hugging Face models |

CPU CI defaults: `VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_COMPUTE_BACKEND=cpu`.

## Commands

### Training

```shell
poetry run train-domain-t2i                           # Flux T2I LoRA
poetry run train-domain-t2v                           # Wan 2.1 T2V LoRA
poetry run train-domain-i2v                           # Wan 2.1 I2V LoRA
```

### Inference / Validation

```shell
poetry run inference-domain-t2i --lorackpt <path>     # Flux LoRA smoke
poetry run validate-domain-t2v --trained_ckpt <path>  # Wan 2.2 validation
poetry run validate-domain-i2v --trained_ckpt <path>  # Wan 2.2 I2V validation
poetry run inference-wan2.2-t2v-720p                  # General Wan 2.2 (optional)
```

### Dev

```shell
poetry run test -q                # Run full pytest suite
poetry run test tests/test_foo.py -q   # Single test file (but see note below)
poetry run pytest tests/test_foo.py -q  # Use pytest directly for single file
poetry run lint                   # ruff check (videotuna tests scripts tools)
poetry run format-check           # ruff format check
poetry run format                 # auto-format with ruff
poetry run type-check             # mypy on typed allowlist (4 modules)
poetry run coverage-report        # full coverage report (no gate)
poetry run coverage-gate          # CI smoke tests + 35% coverage floor
poetry run test-smoke             # Pre-commit smoke tests (no coverage)
```

**Important:** `poetry run test <path>` appends arguments to `pytest tests`, so it always collects the whole suite regardless of the path. To run a single file, call `poetry run pytest <path>` directly.

### Verification gate (run before finishing any change)

```shell
poetry run lint
poetry run format-check
poetry run coverage-gate
```

## Test structure

- **Framework:** pytest ^9.1
- **Config:** `[tool.pytest.ini_options]` in `pyproject.toml`
- **Markers:** `gpu` (skipped without CUDA), `rocm` (skipped without ROCm), `cpu_smoke` (optional nightly)
- **Coverage floor:** 35% on `videotuna/training/*,videotuna/utils/*,videotuna/flow/*,videotuna/cli/*`
- **CI smoke tests (13 files):** defined in `scripts/__init__.py` as `CI_SMOKE_TESTS`
- **conftest.py:** suppresses noisy third-party import warnings; auto-skips GPU/ROCm tests

### Key test files by change area

| Area | Tests |
|------|-------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py`, `test_wan_lora_bridge.py`, `test_wan_i2v_lora_bridge.py` |
| `diffusers_video` flow | `test_diffusers_video_flow.py` |
| Device / attention | `test_device_utils.py`, `test_attention_backend.py` |
| Inference CLI / memory | `test_inference_optimization.py` |
| Flux LoRA training | `test_flux_lora_train_smoke.py`, `test_flux_lora_features.py` |
| Wan training | `test_wan_training_step.py`, `test_wan_train_smoke.py` |
| Config validation | `test_domain_finetune_configs.py` |
| Import boundary | `test_import_smoke.py`, `test_vendor_import_boundary.py` |

## Type checking

Mypy is enabled only on a typed allowlist (4 modules): `videotuna.settings`, `videotuna.cli.inference_options`, `videotuna.training.wan_lora.config`, `videotuna.utils.wan_lora_bridge`. Other modules are excluded by design.

## Pre-commit hooks

Configuration in `.pre-commit-config.yaml`:
- `format-check` — ruff format check
- `lint` — ruff lint
- `type-check` — mypy on allowlist
- `test-smoke` — CI smoke tests (no coverage)
- `commitizen` — commit message format
- Standard hooks: merge conflict check, trailing whitespace, EOF fixer, large file check, private key detection

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to `main` and PRs
- CPU-only environment (`VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_COMPUTE_BACKEND=cpu`)
- Steps: ruff auto-fix → install deps (CPU) → lint → format-check → coverage-gate
- Coverage summary posted to PRs

GPU nightly (`.github/workflows/gpu-nightly.yml`): scheduled Monday 08:00 UTC, self-hosted GPU runner.

## Project layout quick reference

```
videotuna/
  cli/                     # CLI entry points (cyclopts)
  flow/                    # Two execution paths (wanvideo + diffusers_video)
  training/flux_lora/      # Flux T2I trainer (Accelerate)
  training/wan_lora/       # Wan LoRA config only
  models/wan/              # Vendored Wan 2.1 native stack
  utils/                   # Shared utilities (bridge, device, callbacks, etc.)
  base/                    # Base classes (GenerationBase, mixins)
  data/                    # Datasets, transforms, Lightning DataModule
  settings.py              # PrivTuneSettings (VIDEOTUNA_* env vars)
configs/domain/            # Training configs
configs/inference/presets/ # Inference presets
docs/                      # ADRs, runbooks, vendor policy
tests/                     # pytest suite (~46 files)
scripts/                   # Poetry command implementations
cloud/vast/                # Vast.ai GPU provisioning
docker/                    # Dockerfiles
tools/                     # Utility scripts (checkpoint conversion, debugging)
```
