# PrivTune

**PrivTune** is a private-domain LoRA training platform for still-image and short-video generation — Flux T2I style training, Wan 2.1 T2V LoRA training, and Wan 2.2 Diffusers validation inference.

The Python package directory remains `videotuna/` for compatibility; Poetry project name is `privtune`.

Canonical runbook: [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md)

## Pipeline

| Phase | Model | Data | Train | Validate |
|-------|-------|------|-------|----------|
| 1 — T2I | FLUX.1-dev LoRA | `data/t2i/domain/` | `poetry run train-flux-lora` | `poetry run inference-flux-lora` |
| 2 — T2V | Wan 2.1 T2V LoRA | `data/t2v/domain/` | `poetry run train-wan2-1-t2v-lora` | `inference_new` + `wan_domain_lora_smoke` |
| 3 — Production | Wan 2.2 Diffusers | trained LoRA ckpt | — | `poetry run inference-wan2.2-t2v-720p` |

QA is **training callbacks + LoRA smoke inference** — no VBench eval group.

## Removed legacy commands

Legacy inference and training commands for VideoCrafter, DynamiCrafter, Open-Sora, StepVideo, Hunyuan, CogVideoX, Mochi, LTX, and ModelScope V2V are no longer available. VBench evaluation (`eval/`, `poetry install --with eval`) is removed. Use the domain pipeline in [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md): `train-flux-lora`, `train-wan2-1-t2v-lora`, `inference-flux-lora`, and `inference-wan2.2-t2v-720p`.

## Get started

### Install

PrivTune supports **Poetry** (default) and **[uv](https://docs.astral.sh/uv/)**.

| Use case | Poetry | uv |
|----------|--------|-----|
| Inference NVIDIA (default) | `poetry install -E cuda` | `uv sync` |
| Inference AMD ROCm | `poetry install -E rocm` then `poetry run install-rocm` | see [install-rocm.md](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu` then `poetry run install-cpu-torch` | see [install-cpu.md](docs/install-cpu.md) |
| + Training (Flux + Wan LoRA) | `poetry install -E cuda --with training` | `uv sync --group training` |
| + Dev (pytest, ruff) | `poetry install --with dev` | `uv sync --group dev` |

```shell
conda create -n privtune python=3.11 -y
conda activate privtune
pip install poetry
poetry install -E cuda --with training
poetry run install-deepspeed   # required for Wan LoRA
```

See [`docs/vendor-policy.md`](docs/vendor-policy.md) for vendored upstream policy.

### Phase 1 — Flux T2I LoRA

```bash
poetry run train-flux-lora \
  --config_path configs/006_flux/domain_adult_t2i.json \
  --data_config_path configs/006_flux/domain_adult_t2i_data.json

poetry run inference-flux-lora \
  --lorackpt results/train/flux-domain-adult/checkpoint-2000 \
  --prompt "sks_style, portrait, soft lighting"
```

### Phase 2 — Wan 2.1 T2V LoRA

```bash
poetry run train-wan2-1-t2v-lora \
  --base configs/008_wanvideo/wan2_1_t2v_14B_lora_domain.yaml

poetry run python scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --ckpt_path checkpoints/wan/Wan2.1-T2V-14B \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --prompt "sks_style, slow camera push-in"
```

### Phase 3 — Wan 2.2 validation inference

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/balanced_wan2_2_720p.yaml \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --prompt "sks_style, cinematic lighting"
```

See [`docs/runbooks/wan2.2-inference-profile.md`](docs/runbooks/wan2.2-inference-profile.md).

### Cloud GPU training

Rented GPU provisioning (Vast.ai): [`docs/runbooks/cloud-gpu-training.md`](docs/runbooks/cloud-gpu-training.md)

### CPU dev (no weights)

```bash
poetry install -E cpu --with dev
poetry run install-cpu-torch
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_import_smoke.py -q
```

## Environment variables

`VIDEOTUNA_*` env vars are retained for compatibility (see [`.env.example`](.env.example)).

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_ATTN_BACKEND` | `auto`, `flash`, `sdpa`, `eager` — use `sdpa` on ROCm |
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, `cpu` |
| `HF_TOKEN` | Gated models (FLUX.1-dev) |

## Verification

```bash
poetry run lint
poetry run test tests/test_import_smoke.py -q
```

## Project layout

```
videotuna/
  flow/          # wanvideo (train), diffusers_video (Flux + Wan 2.2 infer)
  models/wan/    # Wan 2.1 native training stack
  training/      # flux_lora trainer
  utils/         # device, attention, inference CLI
scripts/         # inference_new.py, train_new.py, train_flux_lora.py
configs/         # 006_flux (domain T2I), 008_wanvideo (domain T2V)
cloud/vast/      # GPU provisioning scripts
docs/runbooks/   # domain-adult-finetune, wan2.2-inference-profile
```

## Related docs

| Doc | Topic |
|-----|-------|
| [domain-adult-finetune.md](docs/runbooks/domain-adult-finetune.md) | Full domain training runbook |
| [checkpoints.md](docs/checkpoints.md) | Weight download layout |
| [MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md) | FLUX.1 + Wan 2.1/2.2 pins |
| [capability-matrix.md](docs/capability-matrix.md) | Supported models matrix |

## License

See [LICENSE](./LICENSE).
