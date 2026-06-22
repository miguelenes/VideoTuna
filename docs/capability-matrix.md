# PrivTune capability matrix

Domain LoRA training and Wan 2.2 validation inference. For checkpoint downloads see [checkpoints.md](checkpoints.md).

## Training

| Phase | Model | Command | Config |
|-------|-------|---------|--------|
| T2I LoRA | FLUX.1-dev | `poetry run train-flux-lora` | `configs/006_flux/domain_adult_t2i.json` |
| T2V LoRA | Wan 2.1 T2V 14B | `poetry run train-wan2-1-t2v-lora` | `configs/008_wanvideo/wan2_1_t2v_14B_lora_domain.yaml` |

Requires `poetry install -E cuda --with training` and `poetry run install-deepspeed` for Wan.

## Smoke inference (QA)

| Phase | Command | Preset |
|-------|---------|--------|
| Flux LoRA | `poetry run inference-flux-lora` | `configs/inference/presets/flux_domain_lora_smoke.yaml` |
| Wan 2.1 LoRA (native) | `poetry run python scripts/inference_new.py ...` | `configs/inference/presets/wan_domain_lora_smoke.yaml` |
| Wan 2.2 validation | `poetry run inference-wan2.2-t2v-720p` | `configs/inference/presets/balanced_wan2_2_720p.yaml` |

Pass `--trained_ckpt` to Wan 2.2 inference to load Wan 2.1 native LoRA via the bridge module.

## Wan 2.2 memory presets (GPU)

| Preset | File | Est. VRAM |
|--------|------|-----------|
| Low VRAM | `presets/low_vram_wan2_2_720p.yaml` | 12–16 GB |
| Balanced | `presets/balanced_wan2_2_720p.yaml` | ~24 GB |
| Max speed | `presets/max_speed_wan2_2_720p.yaml` | 40–48 GB |
| CPU smoke | `presets/wan2_2_cpu_smoke.yaml` | RAM only |

## Attention backends

| Backend | NVIDIA | ROCm | CPU |
|---------|--------|------|-----|
| `auto` | flash → sdpa | sdpa | eager |
| `sdpa` | yes | recommended | yes |
| `eager` | yes | yes | required for `--cpu-smoke` |

```bash
export VIDEOTUNA_ATTN_BACKEND=sdpa   # ROCm
poetry run benchmark-attn-backends --resolutions 480
```

## CPU dev gates (no weights)

```bash
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_wan_lora_bridge.py -q
```

See [domain-adult-finetune.md](runbooks/domain-adult-finetune.md) and [wan2.2-inference-profile.md](runbooks/wan2.2-inference-profile.md).
