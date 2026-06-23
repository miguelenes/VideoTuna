# ADR-002: Wan training stack version pins (DeepSpeed + PyTorch Lightning)

## Status

Accepted

## Date

2026-06-23

## Context

PrivTune's Wan 2.1 T2V/I2V LoRA training is the **oldest pinned training stack** in the repo. Flux T2I uses Hugging Face Accelerate on the shared `torch ^2.6` base; Wan uses **PyTorch Lightning + DeepSpeed ZeRO-3 CPU offload** because the 14B video model at 480×832×81 frames does not fit on ~38–44 GB GPUs without sharding (see [ADR-001](0001-dual-training-stacks.md)).

Current exact pins in `[tool.poetry.group.training]`:

| Package | Pin |
|---------|-----|
| `deepspeed` | **0.19.2** |
| `pytorch-lightning` | **2.4.0** |

An upgrade evaluation was requested to determine whether these pins should advance, with GPU smoke on the cloud training preset and explicit documentation if we stay pinned.

### Upstream survey (2026-06-23)

**DeepSpeed**

- **0.19.2 is the latest PyPI release** (2026-06-16). No 0.19.3 exists.
- ZeRO-3 + PEFT mixed-dtype failure ([#8072](https://github.com/deepspeedai/DeepSpeed/issues/8072)) has an open fix ([#8073](https://github.com/deepspeedai/DeepSpeed/pull/8073), not merged).
- PR [#8066](https://github.com/deepspeedai/DeepSpeed/pull/8066) in 0.19.2 correctly preserves fp32 buffers (e.g. RoPE `inv_freq`) but exposed the mixed-dtype allgather bug when PEFT keeps LoRA adapters in fp32 under bf16-mixed training.

**PyTorch Lightning**

| Version | Assessment |
|---------|------------|
| **2.4.0** (current) | Known-good with torch 2.6 + `deepspeed_stage_3_offload`; custom `VideoTunaModelCheckpoint` + `zero_to_fp32` export path |
| **2.5.5+** | Adds `exclude_frozen_parameters` on `DeepSpeedStrategy` ([#21060](https://github.com/Lightning-AI/pytorch-lightning/pull/21060)) — useful for LoRA checkpoint size but **not required** (PrivTune exports LoRA-only via custom callback) |
| **2.6.1** | Safe ceiling — last release before supply-chain compromise |
| **2.6.2 / 2.6.3** | **Compromised** ([GHSA-w37p-236h-pfx3](https://github.com/Lightning-AI/pytorch-lightning/security/advisories/GHSA-w37p-236h-pfx3), CVE-2026-44484) — must never be pinned |

### Evaluation evidence

**CPU gate (current pins):** `poetry run lint`, `poetry run format-check`, and `poetry run coverage-gate` passed on branch `cursor/deepspeed-pl-upgrade-eval-eaa8` (73 passed, 1 skipped). Resolved versions: `deepspeed 0.19.2`, `pytorch-lightning 2.4.0`, `torch 2.6.0+cu126`.

**GPU smoke:** Not executed in the evaluation agent environment (no NVIDIA GPU, no Wan 14B weights, no domain dataset). Manual procedure documented in [MODEL_VERSIONS.md](../MODEL_VERSIONS.md#gpu-training-smoke-manual). Re-run on cloud GPU before the next pin review.

**PL upgrade spike:** Skipped. No GPU to validate custom ZeRO-3 checkpoint export; low upside vs regression risk on `videotuna/utils/callbacks.py` `zero_to_fp32` path.

## Decision

**Keep `deepspeed==0.19.2` and `pytorch-lightning==2.4.0`.** Do not bump either package in this evaluation cycle.

1. **DeepSpeed** — already at latest PyPI; no newer release to adopt.
2. **PyTorch Lightning** — defer past 2.4.0. Optional future target is **≤ 2.6.1** only, after GPU smoke confirms checkpoint export and first-step training. **Never pin 2.6.2+** until upstream security posture is restored.
3. **Mitigation retained** — `autocast_adapter_dtype=False` in `videotuna/base/lora_training_mixin.py` and `videotuna/utils/wan_training.py` remains required until DeepSpeed ships #8073 in a post-0.19.2 release.

## Consequences

### No dependency changes

`pyproject.toml`, lockfiles, and `scripts/__init__.py` `install_deepspeed()` stay on current pins.

### Revisit triggers

- DeepSpeed **0.19.3+** releases with merged #8073 (may allow re-evaluating `autocast_adapter_dtype` policy).
- A concrete PrivTune pain point that PL 2.5.5+ `exclude_frozen_parameters` solves, validated by GPU smoke.
- PL security advisory lifted for versions above 2.6.1 (still cap at last known-good release).

### Governance

- Version table and breaking-change notes live in [MODEL_VERSIONS.md](../MODEL_VERSIONS.md); this ADR records the **evaluation outcome**, not duplicate pin tables.
- [ADR-001](0001-dual-training-stacks.md) remains authoritative for *why* Wan uses Lightning+DeepSpeed; this ADR is authoritative for *which versions* to pin.

## Related docs

- [ADR-001: Dual training stacks](0001-dual-training-stacks.md)
- [MODEL_VERSIONS.md](../MODEL_VERSIONS.md) — Wan training stack pins section
- [domain-adult-finetune.md](../runbooks/domain-adult-finetune.md) — VRAM and training runbook
- [cloud-gpu-training.md](../runbooks/cloud-gpu-training.md) — Vast smoke training
