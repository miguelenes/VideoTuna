# Devin — Project Context

## Project

**PrivTune** (Poetry: `privtune`, Python import: `videotuna/`) is a private-domain LoRA training platform for image and video generation. It is a **training platform ONLY** — not a general inference service.

| Phase | Model | Goal |
|-------|-------|------|
| 1 — T2I | FLUX.1-dev LoRA | Still-image domain style training |
| 2 — T2V | Wan 2.1 T2V LoRA | Short-video motion training |
| 2.5 — I2V | Wan 2.1 I2V LoRA | Image-to-video domain training (optional) |
| 3 — Validate | Wan 2.2 Diffusers | Domain LoRA validation via `validate-domain-t2v` |

Python 3.11+. **Always use `poetry run …`** unless explicitly told otherwise. `uv` is supported but secondary.

## Architecture

- **Two inference flows** (`videotuna/flow/`): `wanvideo.py` (PyTorch Lightning / native Wan 2.1) and `diffusers_video.py` (Unified Diffusers for Wan 2.2 + Flux). Never mix them.
- **LoRA bridge** (`videotuna/utils/wan_lora_bridge.py`): remaps Wan 2.1 → 2.2 LoRA keys. Critical path between training and validation.
- **Device handling** (`videotuna/utils/device_utils.py`): single source of truth for CUDA/ROCm/CPU. Never check `torch.cuda.is_available()` directly.
- **Settings** (`videotuna/settings.py`): all config via `VIDEOTUNA_*` env vars (pydantic-settings).
- **CLI** (`videotuna/cli/inference_app.py`): cyclopts-based registration for all entry points.

## Commands

### Install

```bash
# CUDA + training (default)
poetry install -E cuda --with training
poetry run install-deepspeed   # required for Wan LoRA

# CPU dev / CI
poetry install -E cpu --with dev --with training
poetry run install-cpu-torch
```

### Dev tools

```bash
poetry run lint            # ruff check
poetry run format-check    # ruff format --check
poetry run format          # ruff format (apply)
poetry run type-check      # mypy (typed allowlist only)
```

### Tests

```bash
poetry run test -q                          # full suite
poetry run test tests/test_foo.py -q        # single file
poetry run test tests/test_foo.py::test_bar # single test
```

Test markers: `gpu` (CUDA-only), `rocm`, `cpu_smoke`.

### Training

```bash
poetry run train-domain-t2i   # Phase 1 — Flux T2I
poetry run train-domain-t2v   # Phase 2 — Wan T2V
poetry run train-domain-i2v   # Phase 2.5 — Wan I2V (optional)
```

Configs: `configs/domain/flux_t2i.json`, `configs/domain/wan_t2v_lora.yaml`, `configs/domain/wan_i2v_lora.yaml`

### Inference / validation

```bash
poetry run inference-domain-t2i --lorackpt <path>    # Phase 1
poetry run validate-domain-t2v --trained_ckpt <path> # Phase 2 → 3 bridge
poetry run validate-domain-i2v --trained_ckpt <path> # Phase 2.5 → 3 bridge
poetry run inference-wan2.2-t2v-720p                 # Phase 3 (optional)
```

## Verification (required before finishing any change)

```bash
poetry run lint
poetry run format-check
poetry run coverage-gate
```

`coverage-gate` enforces **33%** line coverage on `videotuna/training/` + `videotuna/utils/`.

Additional tests by change area:

| Area | Tests |
|------|-------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py` |
| diffusers_video | `test_diffusers_video_flow.py` |
| Device / attention | `test_device_utils.py`, `test_attention_backend.py` |
| Inference CLI / memory | `test_inference_optimization.py` |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | `cuda` / `rocm` / `cpu` |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` | `flash` / `sdpa` / `eager` |
| `VIDEOTUNA_TORCH_COMPILE` | `0` | Denoiser compile |
| `HF_TOKEN` | — | Gated HF models |

## Boundaries

- **Never commit** `.env`, checkpoints, `outputs/`, `results/`, `data/`, weights, or secrets.
- **ROCm**: set `VIDEOTUNA_ATTN_BACKEND=sdpa`; never run `install-flash-attn`.
- **QA = training callbacks + smoke inference.** No generic T2V benchmarking (VBench removed).
- Read `docs/vendor-policy.md` before modifying `videotuna/models/wan/wan/` (vendored code).
- CPU env: set `VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_TORCH_COMPILE=0`.
- GPU training/inference uses real FLUX/Wan weights (not runnable on CPU).

## References

- Canonical instructions: [`AGENTS.md`](../AGENTS.md)
- Claude Code guide: [`CLAUDE.md`](../CLAUDE.md)
- Training runbook: [`docs/runbooks/domain-adult-finetune.md`](../docs/runbooks/domain-adult-finetune.md)
- Architecture decisions: [`docs/decisions/0001-dual-training-stacks.md`](../docs/decisions/0001-dual-training-stacks.md)
- Vendor policy: [`docs/vendor-policy.md`](../docs/vendor-policy.md)
- MCP config: [`.mcp.json`](../.mcp.json)
