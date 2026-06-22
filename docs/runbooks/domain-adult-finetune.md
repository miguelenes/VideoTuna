# Domain adult fine-tuning runbook (T2I + T2V)

Two-phase pipeline for domain-specific adult content: **Phase 1** Flux LoRA (still images), **Phase 2** Wan 2.1 T2V LoRA (short video clips).

All training data must be rights-cleared and consented. Never commit datasets, weights, or `outputs/` to git.

## Prerequisites

```bash
cd /path/to/PrivTune
poetry install -E cuda --with training   # or: poetry install -E rocm --with training
poetry run install-deepspeed             # required for Wan LoRA (ZeRO-3 offload)
huggingface-cli login                    # FLUX.1-dev is gated on Hugging Face
```

| Environment | Extra steps |
|-------------|-------------|
| AMD ROCm | `export VIDEOTUNA_ATTN_BACKEND=sdpa` — do not run `install-flash-attn` |
| CPU only | Config validation only — run training on a GPU machine (see [CPU stub](#cpu-stub-no-gpu)) |

## VRAM and time expectations

| Phase | Model | Peak VRAM | GPUs | Rough time | Limitation |
|-------|-------|-----------|------|------------|------------|
| 1 — T2I | Flux LoRA @ 512px | ~24–40 GB | 1 | 2000 steps ≈ hours on A100-class | Trains **FLUX.1-dev** |
| 2 — T2V | Wan 2.1 T2V LoRA @ 480×832×81 | ~38 GB | 1 + DeepSpeed | ~41 s/epoch on H800 | Trains **Wan 2.1**; production validation on Wan 2.2 (Phase 3) |

---

## Phase 1 — Flux T2I LoRA

### Dataset layout

Place images and sidecar captions under `data/t2i/domain/` (gitignored):

```
data/t2i/domain/
  0001.jpg
  0001.txt          # e.g. "sks_style, portrait, studio lighting"
  0002.jpg
  0002.txt
```

- Use a **consistent trigger token** (default: `sks_style`) in every `.txt` file.
- `caption_strategy: filename` pairs `0001.txt` with `0001.jpg`.
- Minimum ~10–30 images for a smoke run; 50–200+ recommended for production.

### Config files

| File | Purpose |
|------|---------|
| `configs/domain/flux_t2i.json` | Training hyperparameters |
| `configs/domain/flux_t2i_data.json` | Dataset backend (`data/t2i/domain`) |

### Download base weights

Weights auto-download on first train. For offline use:

```bash
mkdir -p checkpoints/flux
hf download black-forest-labs/FLUX.1-dev --local-dir checkpoints/flux/FLUX.1-dev
```

Then set `"--pretrained_model_name_or_path": "checkpoints/flux/FLUX.1-dev"` in `flux_t2i.json`.

### Train

```bash
poetry run train-domain-t2i
```

Legacy alias: `poetry run train-flux-lora` (same defaults).

Checkpoints: `results/train/flux-domain-adult/checkpoint-<step>/` (Diffusers LoRA format).

For a quick smoke on GPU, temporarily set `"--max_train_steps": 50` in the JSON.

### Inference smoke

```bash
poetry run inference-domain-t2i \
  --lorackpt results/train/flux-domain-adult/checkpoint-2000 \
  --prompt "sks_style, portrait, soft lighting"
```

---

## Phase 2 — Wan 2.1 T2V LoRA

### Dataset layout

```
data/t2v/domain/
  metadata.csv
  videos/
    clip001.mp4
    clip002.mp4
```

`metadata.csv`:

```csv
path,caption
data/t2v/domain/videos/clip001.mp4,"sks_style, slow pan, cinematic lighting"
data/t2v/domain/videos/clip002.mp4,"sks_style, close-up, warm lighting"
```

Clips should be **480×832**, **81 frames**. Re-encode if needed:

```bash
ffmpeg -i in.mp4 -vf scale=832:480 -r 16 -frames:v 81 data/t2v/domain/videos/clip001.mp4
```

### Config file

`configs/domain/wan_t2v_lora.yaml` — domain CSV path, 25-step checkpoint interval, 50 max epochs (raise for production).

### Download base weights

```bash
mkdir -p checkpoints/wan
hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/wan/Wan2.1-T2V-14B
```

### Train

```bash
poetry run train-domain-t2v
```

Legacy alias: `poetry run train-wan2-1-t2v-lora` (same defaults).

Checkpoint example:

`results/train/train_wan_domain_t2v_lora_<timestamp>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt`

### Inference smoke (interim — Wan 2.1 native)

Use this for Phase 2 validation until Wan 2.2 bridge work is complete (Prompt 4):

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --ckpt_path checkpoints/wan/Wan2.1-T2V-14B \
  --trained_ckpt results/train/train_wan_domain_t2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt "sks_style, slow camera push-in, soft lighting" \
  --height 480 --width 832 --frames 81 \
  --num_inference_steps 20 \
  --enable_model_cpu_offload
```

### Phase 3 — Wan 2.2 production validation (deferred)

Wan 2.2 Diffusers 720p validation is documented in [wan2.2-inference-profile.md](wan2.2-inference-profile.md). Not required for the training-only milestone.

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/balanced_wan2_2_720p.yaml \
  --trained_ckpt results/train/train_wan_domain_t2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt "sks_style, cinematic lighting" \
  --enable_model_cpu_offload
```

---

## CPU stub (no GPU)

When no CUDA/ROCm GPU is available locally:

1. Do **not** run training.
2. Validate configs:

```bash
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_poetry_scripts.py -q
```

3. Run the full train/infer commands above on a GPU machine with the same repo checkout and dataset paths.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM (Flux) | Lower `--resolution` to 384 in JSON; keep `gradient_checkpointing: true` |
| CUDA OOM (Wan) | Confirm DeepSpeed installed; reduce `num_frames` or resolution in YAML |
| ROCm flash-attn error | `export VIDEOTUNA_ATTN_BACKEND=sdpa` |
| HF gated model | `huggingface-cli login` and accept FLUX.1-dev license |
| Wan grey output at inference | Use `unconditional_guidance_scale: 12.0` during training preview (set in YAML `image_logger`) |

## Known limitations

- **FLUX.1 only:** Training uses FLUX.1-dev; see [`docs/MODEL_VERSIONS.md`](../MODEL_VERSIONS.md).
- **Wan 2.1 → 2.2:** LoRA trains on Wan 2.1 native; Wan 2.2 Diffusers validation uses `videotuna/utils/wan_lora_bridge.py` (Prompt 4).

## Related docs

- [`docs/runbooks/cloud-gpu-training.md`](cloud-gpu-training.md) — Vast.ai / rented GPU provisioning and Syncthing workflow
- [`docs/runbooks/wan2.2-inference-profile.md`](wan2.2-inference-profile.md) — Wan 2.2 production validation (Phase 3)
- [`docs/checkpoints.md`](../checkpoints.md)
- [`docs/datasets.md`](../datasets.md)
