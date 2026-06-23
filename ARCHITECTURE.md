# PrivTune — Architecture

## Conceptual design

PrivTune is a **training-only LoRA platform** for two model families (Flux T2I, Wan T2V/I2V). Training and inference use separate stacks — training runs on native model code, validation inference runs on Diffusers.

The key architectural decision ([ADR-001](docs/decisions/0001-dual-training-stacks.md)): **two training stacks by design, not by technical debt.**

```
┌─────────────────────────────────────────────────────┐
│                    CLI Layer                         │
│  videotuna/cli/ (cyclopts)                          │
│  └─ train_app.py, inference_app.py, *options.py     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──── Phase 1: Flux T2I ────┐  ┌─ Phase 2: Wan ──┐│
│  │ training/flux_lora/        │  │ flow/wanvideo.py ││
│  │  (Accelerate + PEFT)       │  │ (Lightning+ZeRO) ││
│  │  configs/domain/flux_*.json│  │ vendored models/ ││
│  └────────────────────────────┘  └──────┬───────────┘│
│                                          │           │
│                              ┌───────────▼───────────┐│
│                              │ wan_lora_bridge.py     ││
│                              │ (Wan 2.1→2.2 key map) ││
│                              └───────────┬───────────┘│
│                                          │           │
│  ┌───────────────────────────────────────▼───────────┐│
│  │ Phase 3: Validation (DiffusersVideoFlow)          ││
│  │ flow/diffusers_video.py                           ││
│  │ configs/inference/presets/                        ││
│  └───────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                    Shared Layers                      │
│  settings.py (VIDEOTUNA_* env)                       │
│  utils/device_utils.py (compute backend)             │
│  base/ (GenerationBase, Lightning mixins)            │
│  data/ (datasets, transforms)                        │
└─────────────────────────────────────────────────────┘
```

## Major modules

### `videotuna/cli/` — CLI entry points
- **Framework:** Cyclopts ^3.0
- **Files:** `train_app.py`, `inference_app.py`, `train_options.py`, `inference_options.py`
- **Pattern:** Each `poetry run <cmd>` script in `pyproject.toml` points to a function in these files.
- **Options:** Declared as pydantic dataclasses in `*options.py`.

### `videotuna/flow/` — Execution flows (NEVER MIX)
| File | Class | Purpose | Stack |
|------|-------|---------|-------|
| `wanvideo.py` | `WanVideoModelFlow` | Wan 2.1 native training loop | PyTorch Lightning + DeepSpeed ZeRO-3 |
| `diffusers_video.py` | `DiffusersVideoFlow` | Unified inference (Flux + Wan 2.2) | Diffusers pipeline |

**Rule:** `wanvideo.py` is GPU-only (`FLOW_TIERS["gpu_required"]`). `diffusers_video.py` supports CPU smoke mode.

### `videotuna/training/flux_lora/` — Flux T2I LoRA trainer
- **Stack:** Accelerate + PEFT + Diffusers (first-party trainer)
- **Configs:** `configs/domain/flux_t2i.json` + `configs/domain/flux_t2i_data.json`
- **Key files:** `train.py` (loop), `dataset.py` (data loading + bucketing), `config.py` (schema), `checkpoint.py` (I/O), `model_utils.py` (PEFT setup)
- **Checkpoints:** `LoraModelCheckpoint` in `videotuna/utils/callbacks.py` strips non-LoRA weights.

### `videotuna/models/wan/` — Vendored Wan 2.1 stack
- **Status:** Vendored upstream (Apache 2.0); `docs/vendor-policy.md` governs modifications.
- **Contents:** Full Wan 2.1 native model (`modules/model.py`, `modules/attention.py`, VAE, T5), distributed helpers (`FSDP`, `sequence_parallel`), configs.
- **Entrypoints consumed by:** `train-domain-t2v`, `train-domain-i2v`, `scripts/train_new.py`, `scripts/inference_new.py`
- **Boundary:** First-party code must not import `easydict` (used only by vendor configs). Enforced by `test_vendor_import_boundary.py`.

### LoRA bridge: `videotuna/utils/wan_lora_bridge.py`
- **Purpose:** Remaps Wan 2.1 Lightning checkpoint keys (e.g., `blocks.N.self_attn.q`) to Wan 2.2 Diffusers keys (`attn1.to_q`) for validation inference.
- **Coverage:** Must be >= 90%. Verified by `test_wan_lora_bridge.py`.
- **Offline export:** `tools/convert_wan_lora_21_to_22.py`
- **Debugging:** `tools/spike_wan_lora_bridge.py`

### Settings: `videotuna/settings.py`
- **Class:** `PrivTuneSettings` (pydantic-settings)
- **Prefix:** `VIDEOTUNA_*` (retained for compatibility; no `PRIVTUNE_*` aliases)
- **Key fields:** `compute_backend` (auto/cuda/rocm/cpu), `attn_backend` (auto/flash/sdpa/eager), `torch_compile`, `cpu_mode`
- **Session support:** ContextVar-based `settings_session()` for scoped overrides.

### Device: `videotuna/utils/device_utils.py`
- **Single source of truth** for compute backend detection.
- **Functions:** `detect_compute_backend()` → `"cuda"` | `"rocm"` | `"cpu"`, `resolve_inference_device()`, `describe_compute_environment()`
- **Rule:** Never call `torch.cuda.is_available()` directly in flow code.

### Data: `videotuna/data/`
- **`datasets.py`:** CSV-based dataset loading, caption preprocessing
- **`datasets_utils.py`:** Image/video file handling utilities
- **`lightningdata.py`:** PyTorch Lightning DataModule for Wan training
- **`transforms.py`:** Video transforms for Wan data pipeline
- **Toy data:** `anno_files/toy_image_dataset.csv`, `anno_files/toy_video_dataset.csv`

## External dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.6.0 | Core deep learning framework |
| Diffusers | ^0.38 | Inference pipeline (Flux + Wan 2.2) |
| PEFT | ^0.17 | LoRA adapter management |
| Accelerate | ^1.14 | Flux training orchestration |
| DeepSpeed | 0.19.2 | Wan training ZeRO-3 offload |
| PyTorch Lightning | 2.4.0 | Wan training framework |
| Transformers | ^4.48 | Text encoders (T5, CLIP) |
| cyclopts | ^3.0 | CLI framework |
| pydantic-settings | ^2.8 | Environment config |
| safetensors | ^0.8 | Safe tensor serialization |
| opencv-python | 4.10.0 | Video/image processing |
| av (PyAV) | 12.3.0 | Video codec |

## Data flow

### Flux T2I training
```
Config (json) → Accelerate launches train.py
  → Dataset from CSV (bucketed image pairs)
  → PEFT LoRA on Flux transformer
  → Accelerate optimizer loop
  → LoraModelCheckpoint saves LoRA weights (safetensors)
  → ImageLogger logs sample generations
```

### Wan T2V/I2V training
```
Config (yaml) → GenerationBase.init_trainer()
  → LightningDataModule (video dataset from CSV)
  → DeepSpeed ZeRO-3 + native Wan 2.1 model
  → Lightning Trainer loop
  → WanCheckpoint / denoiser checkpoint export
```

### LoRA validation (Phase 3)
```
Trained .safetensors → wan_lora_bridge.py remaps keys
  → PEFT loads onto WanTransformer3DModel
  → DiffusersVideoFlow runs inference
  → Video saved to outputs/
```

## Config file layout

| Path | Format | Purpose |
|------|--------|---------|
| `configs/domain/flux_t2i.json` | JSON | Flux T2I training hyperparameters |
| `configs/domain/flux_t2i_data.json` | JSON | Flux training data paths |
| `configs/domain/wan_t2v_lora.yaml` | YAML | Wan T2V LoRA training config |
| `configs/domain/wan_i2v_lora.yaml` | YAML | Wan I2V LoRA training config |
| `configs/domain/flux_t2i_cloud_smoke.json` | JSON | Cloud smoke test Flux config |
| `configs/inference/presets/*.yaml` | YAML | Inference presets (smoke + production) |

See [ADR-001](docs/decisions/0001-dual-training-stacks.md) for rationale on the dual-stack design.
See `CONTRIBUTING.md` for conventions on config changes.
