# Model versions

PrivTune supports three model families for domain LoRA training and validation.

| Family | Hub ID | Pipeline / stack | Integration | Role |
|--------|--------|------------------|-------------|------|
| Flux T2I | `black-forest-labs/FLUX.1-dev` | `FluxPipeline` | `DiffusersVideoFlow` | Phase 1 T2I LoRA train + smoke infer |
| Wan 2.1 T2V | `Wan-AI/Wan2.1-T2V-14B` | Native Wan modules | `wanvideo.py` | Phase 2 T2V LoRA train + native smoke |
| Wan 2.2 T2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanPipeline` | `DiffusersVideoFlow` | Phase 3 production validation |

## Flux notes

- Training config: `configs/006_flux/domain_adult_t2i.json`
- Smoke preset: `configs/inference/presets/flux_domain_lora_smoke.yaml`
- Command: `poetry run train-flux-lora` / `poetry run inference-flux-lora`

## Wan 2.1 notes

- Training config: `configs/008_wanvideo/wan2_1_t2v_14B_lora_domain.yaml`
- Requires DeepSpeed ZeRO-3: `poetry run install-deepspeed`
- Native smoke preset: `configs/inference/presets/wan_domain_lora_smoke.yaml`

## Wan 2.2 notes

- Base config: `configs/inference/wan2_2_t2v_a14b.yaml`
- Memory presets: `configs/inference/presets/low_vram_wan2_2_720p.yaml`, `balanced_wan2_2_720p.yaml`, `max_speed_wan2_2_720p.yaml`
- LoRA bridge: Wan 2.1 native `.ckpt` → Diffusers 2.2 via `videotuna/utils/wan_lora_bridge.py`
- Command: `poetry run inference-wan2.2-t2v-720p`

## CI smoke (CPU config validation)

```bash
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_wan_lora_bridge.py -q
```

GPU inference smoke (optional):

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload
```
