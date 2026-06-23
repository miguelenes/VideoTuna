# CLI deprecations

Legacy Poetry script aliases and direct script paths are deprecated in **v0.2.0** and scheduled for removal in **v0.3.0**.

## Training

| Legacy | Canonical |
|--------|-----------|
| `train-flux-lora` | `train-domain-t2i` |
| `train-wan2-1-t2v-lora` | `train-domain-t2v` |
| `train-wan2-1-i2v-lora` | `train-domain-i2v` |

## Inference

| Legacy | Canonical |
|--------|-----------|
| `inference-flux-lora` | `inference-domain-t2i` |
| `python scripts/inference_new.py` | `poetry run inference-run` |
| (preset-specific) | `validate-domain-t2v`, `validate-domain-i2v`, `inference-wan2.2-t2v-720p`, etc. |

## Distributed native inference

`torchrun … scripts/inference_new.py` still works in v0.2.0 but emits a deprecation warning. Use the same flags with `inference-run` for single-process runs. Multi-GPU `torchrun` will migrate in v0.3.0.

## Migration example

```bash
# Before
poetry run train-flux-lora
poetry run python scripts/inference_new.py --config configs/inference/presets/wan_domain_lora_smoke.yaml

# After
poetry run train-domain-t2i
poetry run inference-run --config configs/inference/presets/wan_domain_lora_smoke.yaml
```

See [`docs/runbooks/domain-adult-finetune.md`](runbooks/domain-adult-finetune.md) for the canonical domain training workflow.
