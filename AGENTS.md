# PrivTune — Agent Instructions

## Project overview

**PrivTune** (`privtune` in Poetry; Python import path `videotuna/`) is a private-domain LoRA training platform:

- **Phase 1:** Flux T2I LoRA (`videotuna/training/flux_lora/`, `train-flux-lora`)
- **Phase 2:** Wan 2.1 T2V LoRA (`videotuna/flow/wanvideo.py`, `train-wan2-1-t2v-lora`)
- **Phase 3:** Wan 2.2 Diffusers validation inference (`inference-wan2.2-t2v-720p`)

Canonical runbook: [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md)

Python 3.11+; Poetry default (`poetry run …`); optional uv.

## Role

Optimize for:

1. **Correct behavior** — training and smoke inference on CUDA, ROCm, and CPU config validation.
2. **Scoped diffs** — change only what the task requires.
3. **Portable device handling** — respect `videotuna/utils/device_utils.py` and `.env.example`.
4. **Safe boundaries** — never commit weights, datasets, `outputs/`, or secrets.

Cursor rules: [`.cursor/rules/privtune.mdc`](.cursor/rules/privtune.mdc)

## Agent workflow

1. `cd` into the repo root before running commands.
2. Prefer **Poetry** (`poetry run …`) unless the user explicitly uses uv.
3. Read [`docs/vendor-policy.md`](docs/vendor-policy.md) before touching vendored code.

## Install profiles

| Use case | Poetry | uv |
|----------|--------|-----|
| Inference NVIDIA | `poetry install -E cuda` | `uv sync` |
| Inference AMD ROCm | `poetry install -E rocm` then `poetry run install-rocm` | see [install-rocm.md](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu` then `poetry run install-cpu-torch` | see [install-cpu.md](docs/install-cpu.md) |
| + Training | `poetry install -E cuda --with training` | `uv sync --group training` |
| + Dev | `poetry install --with dev` | `uv sync --group dev` |

## Verification (required before finishing)

```bash
poetry run test tests/test_import_smoke.py -q
poetry run lint
```

| Change area | Additional tests |
|-------------|------------------|
| Domain configs | `test_domain_finetune_configs.py` |
| Flux trainer | `test_flux_lora_train_smoke.py` |
| Wan 2.2 presets | `test_wan_inference_presets.py` |
| diffusers_video | `test_diffusers_video_flow.py` |
| device/attention | `test_device_utils.py`, `test_attention_backend.py` |

## Commands

### Training

```bash
poetry run train-flux-lora --config_path configs/006_flux/domain_adult_t2i.json \
  --data_config_path configs/006_flux/domain_adult_t2i_data.json
poetry run train-wan2-1-t2v-lora --base configs/008_wanvideo/wan2_1_t2v_14B_lora_domain.yaml
poetry run install-deepspeed   # Wan LoRA
```

### Smoke inference

```bash
poetry run inference-flux-lora --lorackpt results/train/flux-domain-adult/checkpoint-2000
poetry run python scripts/inference_new.py --config configs/inference/presets/wan_domain_lora_smoke.yaml ...
poetry run inference-wan2.2-t2v-720p --config configs/inference/presets/balanced_wan2_2_720p.yaml
```

### Dev tooling

```bash
poetry run test -q
poetry run lint
poetry run format-check
poetry run benchmark-attn-backends --pipeline wan
```

## Environment variables

`VIDEOTUNA_*` prefix is retained for compatibility (no `PRIVTUNE_*` aliases in v0.2).

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | cuda / rocm / cpu override |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` | flash / sdpa / eager |
| `VIDEOTUNA_TORCH_COMPILE` | `0` | denoiser compile |
| `HF_TOKEN` | — | Gated HF models |

## Project layout

```
videotuna/
  flow/          # wanvideo.py, diffusers_video.py
  models/wan/    # Wan 2.1 training
  training/flux_lora/
  utils/
scripts/         # inference_new.py, train_new.py, train_flux_lora.py
configs/006_flux/, configs/008_wanvideo/, configs/inference/
cloud/vast/
docs/runbooks/
```

## Safety

- Never commit `.env`, checkpoints, `outputs/`, `results/`, or training data.
- ROCm: `VIDEOTUNA_ATTN_BACKEND=sdpa`; do not run `install-flash-attn`.
- QA = training callbacks + smoke inference (no VBench).

## Related docs

| Doc | Topic |
|-----|-------|
| [domain-adult-finetune.md](docs/runbooks/domain-adult-finetune.md) | Domain training runbook |
| [wan2.2-inference-profile.md](docs/runbooks/wan2.2-inference-profile.md) | Wan 2.2 rental GPU presets |
| [capability-matrix.md](docs/capability-matrix.md) | Supported model matrix |
| [checkpoints.md](docs/checkpoints.md) | Weight layout |
