# PrivTune — Repo Context

**PrivTune** (Poetry: `privtune`, import: `videotuna/`) is a **training-only platform** for private-domain LoRA adapters. It trains Flux T2I (images) and Wan 2.1 T2V/I2V (video) LoRAs, then validates on Wan 2.2 Diffusers. Not a general inference service.

**License:** CC BY-NC-ND 4.0 (root); Apache 2.0 (`videotuna/models/wan/LICENSE`)

## Quick component map

| Layer | Path | Purpose |
|-------|------|---------|
| **CLI** | `videotuna/cli/` | Cyclopts entry points for all `poetry run *` commands |
| **Flow** | `videotuna/flow/` | Two execution paths (`wanvideo.py` = Lightning/Wan 2.1, `diffusers_video.py` = Diffusers/Wan 2.2+Flux) |
| **Trainers** | `videotuna/training/flux_lora/` | Phase 1: Flux T2I (Accelerate + PEFT) |
| **Trainers config** | `videotuna/training/wan_lora/` | Phase 2: Wan 2.1 LoRA config only |
| **Vendored model** | `videotuna/models/wan/` | Upstream Wan 2.1 native stack (Apache 2.0) |
| **Settings** | `videotuna/settings.py` | `PrivTuneSettings` (pydantic-settings, `VIDEOTUNA_*` prefix) |
| **Device** | `videotuna/utils/device_utils.py` | Single source of truth for compute backend detection |
| **LoRA bridge** | `videotuna/utils/wan_lora_bridge.py` | Wan 2.1 Lightning → 2.2 Diffusers key remapping |
| **Configs** | `configs/domain/` | Training configs (`flux_t2i.json`, `wan_t2v_lora.yaml`) |
| **Configs** | `configs/inference/presets/` | Inference preset YAMLs (smoke + production) |
| **Vendor policy** | `docs/vendor-policy.md` | Rules for vendored upstream code |

## Key entry points (18 `poetry run` scripts)

All defined in `pyproject.toml` `[tool.poetry.scripts]`. Implementation in `scripts/__init__.py` or `videotuna/cli/`.

| Command | Purpose | Phase |
|---------|---------|-------|
| `train-domain-t2i` | Flux T2I LoRA training | 1 |
| `train-domain-t2v` | Wan 2.1 T2V LoRA training | 2 |
| `train-domain-i2v` | Wan 2.1 I2V LoRA training | 2.5 |
| `inference-domain-t2i` | Flux LoRA smoke inference | 1 |
| `validate-domain-t2v` | Wan 2.2 Diffusers LoRA validation | 3 |
| `inference-wan2.2-t2v-720p` | General Wan 2.2 T2V 720p (optional) | 3 |
| `install-deepspeed` | DeepSpeed CUDA rebuild | infra |
| `test` | pytest runner | dev |
| `lint` | ruff check | dev |
| `coverage-gate` | CI smoke tests + coverage floor | CI |

## How to get productive fast

1. **Read `ARCHITECTURE.md`** for the system design and two training stacks.
2. **Read `DEVELOPMENT.md`** for install commands by platform.
3. **Read `AGENTS.md`** for the canonical agent instructions.
4. **Never modify `videotuna/models/wan/wan/`** without reading `docs/vendor-policy.md`.
5. **Use `videotuna/utils/device_utils.py`** for device detection — never `torch.cuda.is_available()`.
6. **Run `poetry run lint && poetry run format-check && poetry run coverage-gate`** before finishing changes.

## Related docs

| Doc | What it covers |
|-----|----------------|
| `AGENTS.md` | Canonical agent instructions |
| `CLAUDE.md` | Claude Code specifics |
| `ARCHITECTURE.md` | System architecture deep-dive |
| `DEVELOPMENT.md` | Setup, install, test, lint commands |
| `CONTRIBUTING.md` | Coding conventions, PR process |
| `AGENT_NOTES.md` | Safety notes, anti-patterns, pitfalls |
| `docs/decisions/0001-dual-training-stacks.md` | Why two training stacks |
| `docs/runbooks/domain-adult-finetune.md` | Full training runbook |
| `docs/vendor-policy.md` | Rules for vendored upstream code |
