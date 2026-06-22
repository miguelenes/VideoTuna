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
| **Pinned upstream SHA** | `34b1fd729fd0fa86e6b085ba0f3dbc44ca8757dc` (2025-01-29; reference submodule at `videotuna/vendor/simpletuner/`) |
| **Byte-for-byte match** | No — VideoTuna snapshot was namespace-rewritten to `videotuna.third_party.flux` with 2 functional patches |

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
