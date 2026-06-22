# Vendor policy

VideoTuna vendors upstream model and training code when Diffusers/Hugging Face pipelines are insufficient or when legacy training paths must be preserved. This document defines **where** vendored code lives, **how** to update it, and **what** attribution is required.

## Directory convention

| Location | Purpose | Example |
|----------|---------|---------|
| `videotuna/models/<family>/` | Native model implementations used for inference and/or training | `wan/`, `opensora/`, `hunyuan/` |
| `videotuna/training/<task>/` | First-party training loops (Diffusers + PEFT + Accelerate) | `flux_lora/` |
| `videotuna/vendor/<upstream>/` | Third-party snapshots (git submodule preferred) | *(none today)* |
| `eval/vbench/third_party/<name>/` | Evaluation-only upstream deps (not imported by core package) | `RAFT/`, `ViCLIP/` |

**Rule:** New upstream code goes under `videotuna/vendor/<upstream>/` with a `VENDOR.md` at the tree root. Prefer first-party trainers under `videotuna/training/` when Diffusers covers the model.

## Required provenance (`VENDOR.md`)

Every vendored tree must include `VENDOR.md` (or `LICENSE` + `VENDOR.md`) with:

1. **Upstream repository URL**
2. **License** (SPDX identifier + link to upstream `LICENSE`)
3. **Pinned commit** (full SHA) at last sync
4. **Import date / VideoTuna PR** that introduced or last updated the snapshot
5. **VideoTuna entrypoints** that depend on the tree
6. **Update procedure** (submodule bump, manual diff, or replacement plan)

Archived snapshots: see [`docs/vendor/simpletuner-archive.md`](vendor/simpletuner-archive.md).

## Update process

1. Identify upstream release or commit to pin.
2. Record the SHA in `VENDOR.md` before merging.
3. Run import smoke tests for affected dependency groups (see `tests/test_import_smoke.py`).
4. Run the relevant Poetry script (`poetry run train-*` / `inference-*`) on a smoke config.
5. Note breaking config changes in `README.md` and `docs/MODEL_VERSIONS.md` if applicable.

Prefer **git submodule** or **pip/git dependency** over copying large trees. In-tree copies are allowed only when VideoTuna-specific patches are substantial.

## Dependency groups

| Group | Install command | Consumers |
|-------|-----------------|-----------|
| *(default / main)* | `poetry install` or `uv sync` | Diffusers inference, Wan, Hunyuan native, StepVideo, LVDM |
| `training` | `--with training` | Open-Sora (ColossalAI), Wan/Hunyuan/VC training, Flux LoRA, DeepSpeed |
| `eval` | `--with eval` | VBench metrics (`eval/vbench/`) |
| `dev` | `--with dev` | pytest, ruff, mypy |

## Poetry scripts → dependency groups

| Scripts | Groups required |
|---------|-----------------|
| `inference-*` (Diffusers, Wan, Hunyuan, CogVideo, Flux, Mochi, LTX, …) | default (inference) |
| `inference-opensora-v2`, `inference-opensora-v10-*` | default; ColossalAI only for distributed train paths |
| `train-wan2-*`, `train-hunyuan-*`, `train-cogvideox-*`, `train-videocrafter-*`, `train-dynamicrafter`, `train-opensorav10`, `train-flux-lora` | `training` |
| `install-deepspeed`, `install-flash-attn` | `training` / optional runtime |
| `test`, `lint`, `format*` | `dev` |
| VBench (`eval/scripts/`) | `eval` (+ inference for model outputs) |

## Inventory

| Path | Upstream | License | Entrypoints | Fate |
|------|----------|---------|-------------|------|
| `videotuna/training/flux_lora/` | VideoTuna first-party (replaced SimpleTuner) | N/A | `train-flux-lora` | **Keep** |
| `videotuna/models/wan/` | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Upstream terms | `inference-wan2.2-*`, `train-wan2-*` | **Keep** |
| `videotuna/models/opensora/` | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) | Mixed | `inference-opensora-*`, `train-opensorav10` | **Keep** |
| `videotuna/models/stepvideo/` | [stepfun-ai/Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V) | StepFun headers | `inference-stepvideo-*` | **Keep** |
| `videotuna/models/hunyuan/` | Tencent HunyuanVideo | Apache-2.0 (HF blocks) | `inference-hunyuan-*`, `train-hunyuan-*` | **Keep** |
| `videotuna/models/lvdm/` | [AILab-CVC/VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) + LVDM | Mixed | VC/DC/Open-Sora v1 train configs | **Keep** (frozen legacy) |
| `videotuna/models/cogvideo_hf/` | VideoTuna wrappers | N/A | `train-cogvideox-*`, Diffusers CogVideo | **Keep** |
| `videotuna/vendor/simpletuner/` | [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner) | Apache-2.0 | *(reference only)* | **Submodule** — pinned `34b1fd72`; see [`vendor/VENDOR.md`](../videotuna/vendor/VENDOR.md) |
| `videotuna/third_party/flux/` (SimpleTuner) | [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner) | Apache-2.0 | *(removed)* | **Deleted** — see archive doc |
| `eval/vbench/` + `eval/vbench/third_party/*` | [Vchitect/VBench](https://github.com/Vchitect/VBench) | VBench + sub-vendors | `eval/scripts/evaluation.py` | **Keep** |

## Flux LoRA training

**Current (2025-06):** First-party trainer at `videotuna/training/flux_lora/` (Diffusers + PEFT + Accelerate). Config compatibility shim for `configs/006_flux/`. Inference via `DiffusersVideoFlow` / `inference-flux-lora`.

**Unsupported vs legacy SimpleTuner:** S3 backends, text-embed disk cache, multi-dataset aspect bucketing, SD3/SDXL/SmolDiT, quantisation, LyCORIS.

## Removing vendored code

Before deleting any file:

1. Confirm zero references outside the vendor tree.
2. Confirm no Poetry script imports it.
3. Archive provenance in `docs/vendor/` and update this inventory.
