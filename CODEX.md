# CODEX.md

Instructions for OpenAI Codex when working in this repository.

## Project

**PrivTune** (Poetry package: `privtune`, Python import path: `videotuna/`) is a private-domain LoRA training platform for image and video generation. It is a **training platform ONLY** — not a general inference service.

| Phase | Model | Task |
|-------|-------|------|
| 1 | FLUX.1-dev LoRA | Text-to-image style training |
| 2 | Wan 2.1 T2V LoRA | Text-to-video motion training |
| 2.5 | Wan 2.1 I2V LoRA | Image-to-video training (optional) |
| 3 | Wan 2.2 Diffusers | Domain LoRA validation |

Python 3.11+. **Always use `poetry run …`** unless told otherwise.

## Commands

```bash
# Install
poetry install -E cuda --with training
poetry run install-deepspeed   # Wan LoRA

# Dev tools
poetry run lint                # ruff check
poetry run format-check        # ruff format --check
poetry run type-check          # mypy (typed allowlist only)

# Tests
poetry run test -q

# Training
poetry run train-domain-t2i    # Phase 1 — Flux T2I
poetry run train-domain-t2v    # Phase 2 — Wan T2V

# Inference / validation
poetry run inference-domain-t2i --lorackpt <path>
poetry run validate-domain-t2v --trained_ckpt <path>
```

## Architecture

- **Two inference flows** in `videotuna/flow/`: `wanvideo.py` (Lightning / Wan 2.1) and `diffusers_video.py` (Diffusers / Wan 2.2 + Flux). Never cross streams.
- **Device detection** via `videotuna/utils/device_utils.py`. Never use `torch.cuda.is_available()` directly.
- **Settings** via `videotuna/settings.py` (`VIDEOTUNA_*` env vars, pydantic-settings).
- **CLI** via `videotuna/cli/inference_app.py` (cyclopts). New commands follow this pattern.
- **LoRA bridge** (`videotuna/utils/wan_lora_bridge.py`) remaps Wan 2.1 → 2.2 keys.
- **Trainer stacks**: Flux = Diffusers + PEFT + Accelerate. Wan = Lightning + DeepSpeed ZeRO-3.

## Key environment variables

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
- **GPU tests**: Mark with `@pytest.mark.gpu` (skipped without CUDA).
- **Scoped diffs**: Change only what the task requires. No incidental refactoring.

## Verification (required before finishing)

```bash
poetry run lint
poetry run format-check
poetry run coverage-gate   # 33% floor on videotuna/training/ + videotuna/utils/
```

## Safety

- Never commit `.env`, checkpoints, `outputs/`, `results/`, `data/`, weights, or secrets.
- ROCm: `VIDEOTUNA_ATTN_BACKEND=sdpa`. Never run `install-flash-attn`.
- CPU: `VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_TORCH_COMPILE=0`.
- Read `docs/vendor-policy.md` before touching `videotuna/models/wan/wan/`.
- No generic T2V benchmarking (VBench removed). QA = training callbacks + smoke inference.

## References

- [`AGENTS.md`](AGENTS.md) — Full agent instructions
- [`CLAUDE.md`](CLAUDE.md) — Claude Code guide
- [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md)
- [`docs/decisions/0001-dual-training-stacks.md`](docs/decisions/0001-dual-training-stacks.md)
- [`.mcp.json`](.mcp.json) — MCP servers
