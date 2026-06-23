# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

**Canonical agent instructions:** [`AGENTS.md`](./AGENTS.md) — read that first. This file adds Claude-specific detail.

**Companion context files:** [`CONTEXT.md`](CONTEXT.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`DEVELOPMENT.md`](DEVELOPMENT.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`AGENT_NOTES.md`](AGENT_NOTES.md).

## Architecture deep-dive

### Two inference flows

`videotuna/flow/` contains the two execution paths that never mix:

- **`wanvideo.py` (`WanVideoModelFlow`)** — PyTorch Lightning / native Wan 2.1 stack. Used by `train-domain-t2v`, `train-domain-i2v`, and the Wan 2.1 smoke scripts. Requires GPU (`FLOW_TIERS["gpu_required"]`).
- **`diffusers_video.py` (`DiffusersVideoFlow`)** — Unified Diffusers pipeline supporting Flux T2I, Wan 2.2 T2V, and Wan 2.2 I2V. Used for all production inference and Flux training. Runs in CPU smoke mode for config validation.

### LoRA bridge: Wan 2.1 → 2.2

`videotuna/utils/wan_lora_bridge.py` is the critical path between training and validation. Wan 2.1 native Lightning LoRA checkpoints (`blocks.N.self_attn.q`) use different key names than Wan 2.2 Diffusers (`attn1.to_q`). The bridge remaps and loads the LoRA weights onto a `WanTransformer3DModel` via PEFT. Always verify remap coverage ≥ 90% when touching this file (`test_wan_lora_bridge.py`).

For offline export: `tools/convert_wan_lora_21_to_22.py`. For debugging: `tools/spike_wan_lora_bridge.py`.

### Phase 1: Flux LoRA trainer

`videotuna/training/flux_lora/` is a first-party trainer (Diffusers + PEFT + Accelerate). Configs live in `configs/domain/flux_t2i.json` + `configs/domain/flux_t2i_data.json`. `LoraModelCheckpoint` in `videotuna/utils/callbacks.py` strips all non-LoRA weights from checkpoints.

### Phase 2: Wan 2.1 native stack

`videotuna/models/wan/wan/` is vendored upstream Wan 2.1 code (see `docs/vendor-policy.md`). Do not modify freely — check the vendor policy before making changes. Training runs under DeepSpeed ZeRO-3 via `wanvideo.py`.

### Settings and device handling

All environment configuration goes through `videotuna/settings.py` (`PrivTuneSettings`, pydantic-settings). Environment variables use `VIDEOTUNA_*` prefix (retained from upstream for compatibility; no `PRIVTUNE_*` aliases exist).

`videotuna/utils/device_utils.py` is the single source of truth for compute backend detection (cuda/rocm/cpu). Always call `detect_compute_backend()` or `resolve_inference_device()` — do not check `torch.cuda.is_available()` directly in flow code.

### CLI layer

`videotuna/cli/inference_app.py` uses `cyclopts` to register all `inference-*` and `validate-*` Poetry scripts. Options are declared as dataclasses in `videotuna/cli/inference_options.py` (`InferenceRunOptions`, `StandardInferenceOptions`, `InferencePreset`). New inference entry points follow this pattern.

### Config layout

```
configs/domain/          # training configs (flux_t2i*.json, wan_t2v_lora.yaml, wan_i2v_lora.yaml)
configs/inference/
  presets/               # smoke + production presets (wan_domain_*, balanced_*, low_vram_*)
```

## Pre-merge checklist

Run these before finishing any change:

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

Additional tests by change area:

| Area | Tests |
|------|-------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py` |
| `diffusers_video` flow | `test_diffusers_video_flow.py` |
| Device / attention | `test_device_utils.py`, `test_attention_backend.py` |
| Inference CLI / memory | `test_inference_optimization.py` |
