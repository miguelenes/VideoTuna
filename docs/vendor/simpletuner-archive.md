# SimpleTuner snapshot archive

VideoTuna replaced the in-tree SimpleTuner snapshot with a first-party Flux LoRA trainer
(`videotuna/training/flux_lora/`) in 2025-06. This document records provenance before deletion
of `videotuna/third_party/flux/`.

| Field | Value |
|-------|-------|
| **Upstream** | https://github.com/bghira/SimpleTuner |
| **License** | Apache-2.0 |
| **VideoTuna import** | Pre-2025; last touched in git commit `1100b6a` |
| **Best-match upstream era** | SimpleTuner flat layout before the `simpletuner` pip package restructure |
| **Pinned upstream SHA** | Not verified byte-for-byte — snapshot was namespace-rewritten to `videotuna.third_party.flux` |

## VideoTuna-only patches (2 functional hooks)

| File | Change |
|------|--------|
| `training/model.py` | `LoraModelCheckpoint` from `videotuna.utils.callbacks` |
| `training/model.py` | `get_resize_crop_region_for_grid` from `videotuna.utils.common_utils` |

Additionally, 39 Python files had import paths rewritten to `videotuna.third_party.flux.*`.

## Replacement

| Before | After |
|--------|-------|
| `scripts/train_flux_lora.py` → SimpleTuner Model/ModelData | `videotuna.training.flux_lora.train` |
| `configs/006_flux/config.json` | Same config via compatibility shim |
| 71-file vendor tree | Deleted |

## Unsupported SimpleTuner features (not ported)

AWS/S3 backends, webhooks, text-embed disk cache, SD3/SDXL/SmolDiT, quantisation, LyCORIS, Compel.
