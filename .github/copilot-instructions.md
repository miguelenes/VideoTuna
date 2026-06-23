# PrivTune — GitHub Copilot Instructions

Canonical agent instructions: [`AGENTS.md`](../AGENTS.md) at the repo root.

## Tech stack

- **Python 3.11+**, **Poetry** (`poetry run …`); uv is secondary
- **Core**: PyTorch 2.6 (CUDA 12.6), Diffusers `^0.38`, PEFT `^0.17`, Transformers `^4.48`, Accelerate `^1.14`
- **Training group**: DeepSpeed 0.19.2, PyTorch Lightning 2.4.0
- **Dev**: pytest `^9.1`, ruff `^0.6.8`, mypy `^1.11`, coverage `^7.6`
- **CLI**: Cyclopts `^3.0`, pydantic-settings `^2.8`

## Coding conventions

- **ruff**: Select `E`, `F`, `C90`, `I`; line-length 88; target py311; max-complexity 19
- **isort**: First-party = `["videotuna"]`
- **mypy**: Typed allowlist only — `videotuna.settings`, `videotuna.cli.inference_options`, `videotuna.training.wan_lora.config`, `videotuna.utils.wan_lora_bridge`
- **CRITICAL**: Never add code comments unless explicitly requested
- **No emojis** in code files unless the user explicitly asks
- Use `videotuna/utils/device_utils.py` for device detection — never call `torch.cuda.is_available()` directly
- Environment vars: `VIDEOTUNA_*` prefix (see `.env.example`)

## Project layout

```
videotuna/
  cli/              # Cyclopts CLI entry points (inference_app.py, train_app.py)
  flow/             # Two inference/training flows (wanvideo.py, diffusers_video.py)
  models/wan/       # Vendored Wan 2.1 code — do not modify freely (see docs/vendor-policy.md)
  training/
    flux_lora/      # Phase 1: Flux T2I LoRA (Accelerate + PEFT)
    wan_lora/       # Phase 2: Wan 2.1 LoRA config
  utils/            # wan_lora_bridge.py, device_utils.py, callbacks.py, attention.py
  base/             # Base classes (GenerationBase, Lightning trainer mixins)
  data/             # Dataset handling
configs/
  domain/           # Training configs (flux_t2i.json, wan_t2v_lora.yaml)
  inference/presets/# Inference presets
docs/
  decisions/        # ADRs (0001-dual-training-stacks.md, 0002-version-pins.md)
  runbooks/         # domain-adult-finetune.md, cloud-gpu-training.md
tests/              # pytest suite
```

## Key architectural decisions

1. **Two training stacks** ([ADR-001](../docs/decisions/0001-dual-training-stacks.md)):
   - Flux T2I → Accelerate (`videotuna/training/flux_lora/`)
   - Wan T2V/I2V → Lightning + DeepSpeed ZeRO-3 (`videotuna/flow/wanvideo.py`)
   - **Never unify** these stacks without superseding the ADR

2. **LoRA bridge**: `videotuna/utils/wan_lora_bridge.py` remaps Wan 2.1 Lightning keys → Wan 2.2 Diffusers keys. Remap coverage must stay >= 90%. Verify with `test_wan_lora_bridge.py`.

3. **Validation inference** uses `DiffusersVideoFlow` (unified Diffusers pipeline).

4. **Vendor policy** ([docs/vendor-policy.md](../docs/vendor-policy.md)): `videotuna/models/wan/` is vendored upstream. Check vendor policy before modifying.

## Testing

- **Framework**: pytest `^9.1` with markers: `gpu`, `rocm`, `cpu_smoke`
- **Coverage gate**: 33% line-coverage floor on `videotuna/training/` + `videotuna/utils/`
- Run tests: `poetry run test -q`
- Single file: `poetry run test tests/test_foo.py -q`
- Smoke tests: `poetry run test tests/test_import_smoke.py -q`
- Pre-finish: `poetry run lint && poetry run format-check && poetry run coverage-gate`

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

## Safety rules

- Never commit `.env`, checkpoints, `outputs/`, `results/`, weights, or secrets
- ROCm: always `VIDEOTUNA_ATTN_BACKEND=sdpa`; do not install flash-attn
- On CPU (CI): `VIDEOTUNA_ATTN_BACKEND=eager`, `VIDEOTUNA_TORCH_COMPILE=0`
