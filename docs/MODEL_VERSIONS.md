# Model versions

PrivTune supports three model families: two for **training** and one for **validation**.

| Model | Hub ID | Role |
|-------|--------|------|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | **Train** — Phase 1 T2I LoRA |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B` | **Train** — Phase 2 T2V LoRA |
| Wan 2.1 I2V 14B 480P | `Wan-AI/Wan2.1-I2V-14B-480P` | **Train** — Phase 2.5 I2V LoRA (optional) |
| Wan 2.2 T2V A14B Diffusers | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | **Validate** — Phase 2 production inference |
| Wan 2.2 I2V A14B Diffusers | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | **Validate** — Phase 2.5 I2V production inference |

## Training configs

| Phase | Config |
|-------|--------|
| Flux T2I | `configs/domain/flux_t2i.json` + `configs/domain/flux_t2i_data.json` |
| Wan T2V | `configs/domain/wan_t2v_lora.yaml` |
| Wan I2V (optional) | `configs/domain/wan_i2v_lora.yaml` |

Commands: `poetry run train-domain-t2i` / `poetry run train-domain-t2v` / `poetry run train-domain-i2v`

Wan training requires DeepSpeed ZeRO-3: `poetry run install-deepspeed`

## Smoke inference presets

| Phase | Preset | Command |
|-------|--------|---------|
| Flux LoRA | `configs/inference/presets/flux_domain_lora_smoke.yaml` | `poetry run inference-domain-t2i` |
| Wan 2.2 domain LoRA | `configs/inference/presets/wan_domain_lora_smoke_22.yaml` | `poetry run validate-domain-t2v --trained_ckpt <ckpt>` |
| Wan 2.2 domain I2V LoRA | `configs/inference/presets/wan_domain_i2v_smoke_22.yaml` | `poetry run validate-domain-i2v --trained_ckpt <ckpt> --prompt_dir <dir>` |
| Wan 2.2 (general) | `configs/inference/presets/balanced_wan2_2_720p.yaml` | `poetry run inference-wan2.2-t2v-720p` |
| Wan 2.1 LoRA (optional) | `configs/inference/presets/wan_domain_lora_smoke.yaml` | `inference_new` + `--trained_ckpt` |
| Wan 2.1 I2V LoRA (optional) | `configs/inference/presets/wan_domain_i2v_smoke.yaml` | `inference_new` + `--prompt_dir` |

LoRA bridge (Wan 2.1 native → 2.2 Diffusers): `videotuna/utils/wan_lora_bridge.py`

Offline export: `tools/convert_wan_lora_21_to_22.py`

## ML stack pins (Wan 2.2 LoRA bridge audit)

Validated against `tests/test_wan_lora_bridge.py` (remap coverage ≥ 90% on production-style fixture keys):

| Package | Constraint | Locked | Notes |
|---------|------------|--------|-------|
| diffusers | `^0.38.0` | 0.38.0 | Wan `WanTransformer3DModel` + `set_adapters(adapter_weights=…)` |
| peft | `^0.17.0` | 0.17.1 | Stays on 0.17.x — 0.18+ requires newer `transformers` than pinned |
| accelerate | `^1.14.0` | 1.14.0 | Transitive via peft / training |
| safetensors | `^0.8.0` | 0.8.0 | Required by diffusers 0.38 |

Matrix results (ephemeral `pip install --no-deps` where noted):

| Row | diffusers | peft | Result |
|-----|-----------|------|--------|
| A (baseline) | 0.36.0 | 0.17.1 | pass |
| B | 0.37.1 | 0.17.1 | pass |
| C | 0.38.0 | 0.17.1 | pass — **chosen combo** |
| D | 0.38.0 | 0.19.1 | fail — `peft` 0.19 needs `transformers` APIs not in `^4.48.0` |
| E | 0.36.0 | 0.19.1 | fail — same `transformers` / `peft` mismatch |

Debug inventory: `poetry run python tools/spike_wan_lora_bridge.py --synthetic /tmp/synthetic.ckpt`

## CI smoke (CPU config validation)

```bash
poetry run lint
poetry run format-check
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_wan_lora_bridge.py -q
poetry run test tests/test_wan_i2v_lora_bridge.py -q
poetry run test tests/test_wan_domain_lora_smoke_22_config.py -q
poetry run test tests/test_wan_domain_i2v_smoke_22_config.py -q
poetry run test tests/test_wan_i2v_dataset.py -q
poetry run test tests/test_wan_training_step.py -q
poetry run test tests/test_poetry_scripts.py -q
```

GPU inference smoke (optional, manual — skipped in CI without GPU):

```bash
# Base pipeline only (no LoRA)
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload

# Domain LoRA validation (requires trained denoiser ckpt + GPU)
poetry run validate-domain-t2v \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --num_inference_steps 4 \
  --config configs/inference/presets/wan_domain_lora_smoke_22_low_vram.yaml
```
