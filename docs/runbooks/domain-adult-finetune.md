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
| 2 — T2V | Wan 2.1 T2V LoRA @ 480×832×81 | ~38 GB | 1 + DeepSpeed | ~41 s/epoch on H800 | Trains **Wan 2.1**; validates on **Wan 2.2** Diffusers 720p |

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

### Validation (Wan 2.2 Diffusers — primary)

After training, validate the native Lightning LoRA on Wan 2.2 Diffusers 720p:

```bash
export VIDEOTUNA_ATTN_BACKEND=auto   # NVIDIA; use sdpa on ROCm
poetry run validate-domain-t2v \
  --trained_ckpt results/train/train_wan_domain_t2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt_file inputs/t2v/domain_prompt.txt \
  --num_inference_steps 4
```

| VRAM | Preset override |
|------|-----------------|
| ~24 GB | default (`wan_domain_lora_smoke_22.yaml`) |
| 12–16 GB | `--config configs/inference/presets/wan_domain_lora_smoke_22_low_vram.yaml` |

Output: `results/t2v/wan-domain-lora-smoke-22/*.mp4` at **720×1280**, **81 frames**, **16 fps**.

For full-quality QA (20–50 steps), use [`balanced_wan2_2_720p.yaml`](../../configs/inference/presets/balanced_wan2_2_720p.yaml) — see [wan2.2-inference-profile.md](wan2.2-inference-profile.md).

**Optional:** export Diffusers safetensors for reuse without runtime PEFT injection:

```bash
poetry run python tools/convert_wan_lora_21_to_22.py \
  --input results/train/.../denoiser-000-000000025.ckpt \
  --output-dir results/lora/wan22-export/
```

**Bridge debug / spike:** `poetry run python tools/spike_wan_lora_bridge.py --synthetic /tmp/synthetic.ckpt`

<details>
<summary>Optional fast path — Wan 2.1 native smoke (480p)</summary>

Use only when debugging training checkpoints on the same base model (no 2.1→2.2 bridge):

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --trained_ckpt results/train/train_wan_domain_t2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt "sks_style, slow camera push-in, soft lighting" \
  --enable_model_cpu_offload
```

</details>

---

## CPU stub (no GPU)

When no CUDA/ROCm GPU is available locally:

1. Do **not** run training.
2. Validate configs:

```bash
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_wan_lora_bridge.py -q
poetry run test tests/test_wan_domain_lora_smoke_22_config.py -q
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
| Wan 2.2 validation OOM | Use `wan_domain_lora_smoke_22_low_vram.yaml` or `--enable_sequential_cpu_offload` |
| Bridge load warnings | Run `tools/spike_wan_lora_bridge.py --input <ckpt>` for key inventory |

## Known limitations

- **FLUX.1 only:** Training uses FLUX.1-dev; see [`docs/MODEL_VERSIONS.md`](../MODEL_VERSIONS.md).
- **Wan 2.1 → 2.2:** LoRA trains on Wan 2.1 native; Wan 2.2 validation uses `videotuna/utils/wan_lora_bridge.py` (loads onto both `transformer` and `transformer_2`).

## Related docs

- [`docs/runbooks/cloud-gpu-training.md`](cloud-gpu-training.md) — Vast.ai / rented GPU provisioning and Syncthing workflow
- [`docs/runbooks/wan2.2-inference-profile.md`](wan2.2-inference-profile.md) — Wan 2.2 VRAM tiers and benchmarks
- [`docs/checkpoints.md`](../checkpoints.md)
- [`docs/datasets.md`](../datasets.md)
