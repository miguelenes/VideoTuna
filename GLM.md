# GLM.md

Instructions for GLM (Zhipu AI) when working in this repository.

**Canonical agent instructions:** [`AGENTS.md`](./AGENTS.md) — read that first. This file is the GLM entrypoint: a condensed operational subset plus a deep-context index in one place. Keep it aligned with `AGENTS.md`, `CLAUDE.md`, and `CODEX.md`; when any of those change, update this file to match.

## Project

**PrivTune** (Poetry package: `privtune`, Python import path: `videotuna/`) is a private-domain LoRA training platform for image and video generation. It is a **training platform ONLY** — not a general inference service.

| Phase | Model | Task |
|-------|-------|------|
| 1 | FLUX.1-dev LoRA | Text-to-image style training |
| 2 | Wan 2.1 T2V LoRA | Text-to-video motion training |
| 2.5 | Wan 2.1 I2V LoRA | Image-to-video training (optional) |
| 3 | Wan 2.2 Diffusers | Domain LoRA validation |

Python 3.11+. **Always use `poetry run …`** unless told otherwise. `uv` is supported but secondary.

## Architecture

- **Two inference flows** in `videotuna/flow/` — never mix them:
  - `wanvideo.py` (`WanVideoModelFlow`) — PyTorch Lightning / native Wan 2.1. Used by `train-domain-t2v`, `train-domain-i2v`. Requires GPU.
  - `diffusers_video.py` (`DiffusersVideoFlow`) — Unified Diffusers for Flux T2I, Wan 2.2 T2V/I2V. All production inference + Flux training. CPU smoke mode for config validation.
- **Trainer stacks differ by design** (see [ADR-001](docs/decisions/0001-dual-training-stacks.md)): Flux = Diffusers + PEFT + Accelerate. Wan = Lightning + DeepSpeed ZeRO-3.
- **LoRA bridge** (`videotuna/utils/wan_lora_bridge.py`) remaps Wan 2.1 Lightning keys → Wan 2.2 Diffusers keys via PEFT. Critical path between training and validation; verify remap coverage ≥ 90% when touched (`test_wan_lora_bridge.py`).
- **Device detection** via `videotuna/utils/device_utils.py` — single source of truth for cuda/rocm/cpu. Never call `torch.cuda.is_available()` directly in flow code.
- **Settings** via `videotuna/settings.py` (`PrivTuneSettings`, pydantic-settings, `VIDEOTUNA_*` prefix).
- **CLI** via `videotuna/cli/inference_app.py` (cyclopts). New entry points follow the dataclass-options pattern in `videotuna/cli/inference_options.py`.
- **Vendored model**: `videotuna/models/wan/wan/` is upstream Wan 2.1 (Apache 2.0). Read [docs/vendor-policy.md](docs/vendor-policy.md) before modifying.

## Commands

### Install
```bash
poetry install -E cuda --with training   # default (CUDA + training)
poetry run install-deepspeed             # required for Wan LoRA

# CPU dev / CI:
poetry install -E cpu --with dev --with training
poetry run install-cpu-torch
```

### Dev tools
```bash
poetry run lint          # ruff check
poetry run format-check  # ruff format --check
poetry run type-check    # mypy (typed allowlist only)
```

### Tests
```bash
poetry run test -q                       # full suite
poetry run pytest tests/test_foo.py      # single file (prefer this form)
```
Markers: `gpu` (CUDA-only), `rocm`, `cpu_smoke`.

### Training
```bash
poetry run train-domain-t2i   # Phase 1 — Flux T2I
poetry run train-domain-t2v   # Phase 2 — Wan T2V
poetry run train-domain-i2v   # Phase 2.5 — Wan I2V (optional)
```
Configs: `configs/domain/flux_t2i.json`, `configs/domain/wan_t2v_lora.yaml`, `configs/domain/wan_i2v_lora.yaml`. Legacy aliases: `train-flux-lora`, `train-wan2-1-t2v-lora`.

### Inference / validation
```bash
poetry run inference-domain-t2i --lorackpt <path>     # Phase 1
poetry run validate-domain-t2v --trained_ckpt <path>  # Phase 2 → 3 bridge
poetry run inference-wan2.2-t2v-720p                  # Phase 3 (optional)
```

## Key environment variables

`VIDEOTUNA_*` prefix is retained for compatibility (no `PRIVTUNE_*` aliases).

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | `cuda` / `rocm` / `cpu` |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` | `flash` / `sdpa` / `eager` |
| `VIDEOTUNA_TORCH_COMPILE` | `0` | Denoiser compile |
| `VIDEOTUNA_CPU_MODE` | `off` | `smoke` / `force` for CPU |
| `HF_TOKEN` | — | Gated HF models |

## Conventions

- **Imports**: Python path `videotuna/`, Poetry name `privtune`.
- **Configs**: Training → `configs/domain/`, Inference → `configs/inference/presets/`.
- **GPU tests**: mark with `@pytest.mark.gpu` (skipped without CUDA).
- **Scoped diffs**: change only what the task requires. No incidental refactoring.
- **ruff**: select `E`, `F`, `C90`, `I`; line-length 88; target py311; max-complexity 19.
- **Comments / emojis**: never add unless explicitly requested.
- **Device handling**: always `videotuna/utils/device_utils.py`, never `torch.cuda.is_available()`.

## Verification (required before finishing any change)

```bash
poetry run lint
poetry run format-check
poetry run coverage-gate   # 33% line-coverage floor on videotuna/training/ + videotuna/utils/
```

Additional tests by change area:

| Area | Tests |
|------|-------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py`, `test_wan_lora_bridge.py` |
| `diffusers_video` flow | `test_diffusers_video_flow.py` |
| Device / attention | `test_device_utils.py`, `test_attention_backend.py` |
| Inference CLI / memory | `test_inference_optimization.py` |
| Flux LoRA training | `test_flux_lora_train_smoke.py` |
| Import / vendor boundary | `test_import_smoke.py`, `test_vendor_import_boundary.py` |

## Safety

- **Never commit** `.env`, checkpoints, `outputs/`, `results/`, `data/`, weights, or secrets.
- **ROCm**: set `VIDEOTUNA_ATTN_BACKEND=sdpa`; never run `install-flash-attn`.
- **CPU**: `VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_TORCH_COMPILE=0`.
- **QA = training callbacks + smoke inference.** No generic T2V benchmarking (VBench removed).
- Read [docs/vendor-policy.md](docs/vendor-policy.md) before touching `videotuna/models/wan/wan/`.

## Deep-context index

One place to find bootstrap, hook, and domain references.

| File | What it covers |
|------|----------------|
| [`AGENTS.md`](AGENTS.md) | Canonical agent instructions (primary source) |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code guide + architecture deep-dive + pre-merge checklist |
| [`CODEX.md`](CODEX.md) | Codex condensed operational guide |
| [`CONTEXT.md`](CONTEXT.md) | Repo component map + entry-point index |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture, two training stacks |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Setup, install, test, lint commands |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Coding conventions, PR process |
| [`AGENT_NOTES.md`](AGENT_NOTES.md) | Safety notes, anti-patterns, pitfalls |
| [`.cursor/rules/privtune.mdc`](.cursor/rules/privtune.mdc) | Cursor rule (alwaysApply) |
| [`.env.example`](.env.example) | Environment variable reference |
| [`.mcp.json`](.mcp.json) | MCP servers |
| [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md) | Canonical domain training runbook |
| [`docs/runbooks/wan2.2-inference-profile.md`](docs/runbooks/wan2.2-inference-profile.md) | Wan 2.2 rental GPU presets |
| [`docs/runbooks/cloud-gpu-training.md`](docs/runbooks/cloud-gpu-training.md) | Cloud GPU training runbook |
| [`docs/decisions/0001-dual-training-stacks.md`](docs/decisions/0001-dual-training-stacks.md) | Why two training stacks (Flux vs Wan) |
| [`docs/decisions/0002-wan-training-stack-version-pins.md`](docs/decisions/0002-wan-training-stack-version-pins.md) | Wan stack version pins |
| [`docs/vendor-policy.md`](docs/vendor-policy.md) | Rules for vendored upstream code |
| [`docs/checkpoints.md`](docs/checkpoints.md) | Weight layout |
| [`docs/MODEL_VERSIONS.md`](docs/MODEL_VERSIONS.md) | Model pins |
| [`docs/install-cpu.md`](docs/install-cpu.md) / [`install-rocm.md`](docs/install-rocm.md) | CPU / ROCm install |

## Alignment

This entrypoint mirrors the canonical project context defined by [`AGENTS.md`](AGENTS.md). The original request referenced `.cursor/hooks/session-context.sh` and `.ai/guidelines/README.md`, neither of which exists in this repo; `AGENTS.md` and its sibling entrypoints (`CLAUDE.md`, `CODEX.md`, `.devin/guidelines.md`) are the real alignment source. When those change, update `GLM.md` to match.
