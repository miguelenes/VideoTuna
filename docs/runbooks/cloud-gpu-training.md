# Cloud GPU training runbook (Vast.ai / linux-desktop template)

Headless training on rented NVIDIA GPUs using the PrivTune `cloud/vast/` provisioning bundle. Primary workflow: **Flux LoRA T2I → Wan 2.1 T2V LoRA** (see [domain-adult-finetune.md](domain-adult-finetune.md)).

Never commit datasets, weights, API keys, or `results/` to git.

## A. Instance selection

| Workload | Peak VRAM | GPU examples | Notes |
|----------|-----------|--------------|-------|
| Flux LoRA @ 512px | ~24–40 GB | RTX 4090 24GB, A100 40GB | No DeepSpeed required |
| Wan 2.1 T2V LoRA @ 480×832×81 | ~38 GB | A100 40GB, H100 | Requires DeepSpeed ZeRO-3 offload |

**CUDA:** Template should ship NVIDIA driver + CUDA 12.x compatible with PrivTune's cu126 PyTorch wheels (`poetry install -E cuda --with training`).

**Disk:** Base weights are large (Wan 14B ≈ tens of GB). Use **≥200 GB** volume. Pre-download via manifest when `HF_TOKEN` is set.

## B. Launch checklist

1. Rent a **linux-desktop** (Selkies/VNC + SSH + Jupyter) template with a compatible CUDA tag.
2. Set template environment variables at rent time:

| Variable | Value |
|----------|-------|
| `WORKSPACE` | `/workspace` |
| `PROVISIONING_MANIFEST` | `https://raw.githubusercontent.com/miguelenes/VideoTuna/main/cloud/vast/provisioning.yaml` |
| `HF_TOKEN` | `hf_...` (required for FLUX.1-dev; accept license on Hugging Face first) |
| `WANDB_API_KEY` | optional |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` or `sdpa` if flash-attn not installed |

Fallback imperative provisioner:

```text
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/miguelenes/VideoTuna/main/cloud/vast/bootstrap.sh
```

3. **SSH in** (preferred) or use the Jupyter terminal on port 8080.
4. Wait for provisioning to finish:
   - Template marker: `/.provisioning_complete`
   - PrivTune marker: `/workspace/.videotuna_provisioned`
   - Logs: `/var/log/portal/provisioning.log` (if present)
5. Confirm smoke tests passed during bootstrap (`poetry run test tests/test_import_smoke.py`).
6. Sync datasets via Syncthing (port **8384**) — see [C. Data sync](#c-data-sync).
7. Start training:

```bash
cd /workspace/VideoTuna
source .env
export TRAIN_PROFILE=flux-lora
./cloud/vast/run-smoke-train.sh    # short GPU validation
./cloud/vast/run-train.sh          # full run
```

**Supervisor (survives SSH disconnect):**

```bash
cp cloud/vast/supervisor/videotuna-train.conf /etc/supervisor/conf.d/
supervisorctl reread && supervisorctl update
# Set TRAIN_PROFILE in .env first
supervisorctl start videotuna-train
supervisorctl status
```

## C. Data sync

Provisioner creates Syncthing-friendly paths and symlinks into the repo:

```
/workspace/data/t2i/domain/     ↔  local data/t2i/domain/
/workspace/data/t2v/domain/     ↔  local data/t2v/domain/
/workspace/results/             ↔  local results/  (pull back after training)
/workspace/checkpoints/         ↔  local checkpoints/  (optional, large)
```

Repo-relative configs (`data/t2i/domain`, `checkpoints/wan/...`) resolve via symlinks under `/workspace/VideoTuna/`.

**Flux layout:** paired `0001.jpg` + `0001.txt` with trigger token `sks_style`.

**Wan layout:** `metadata.csv` + `videos/*.mp4` at 480×832, 81 frames.

## D. Monitoring

```bash
tail -f /workspace/results/train.log
tail -f /workspace/results/train.err
supervisorctl status videotuna-train
nvidia-smi -l 5
```

Optional **Weights & Biases:** set `WANDB_API_KEY` and `WANDB_PROJECT` in `.env` (training dependency group includes `wandb`).

Smoke run logs: `/workspace/results/smoke-train.log`.

## E. Checkpoint recovery

| Phase | Checkpoint path |
|-------|-----------------|
| Flux LoRA | `results/train/flux-domain-adult/checkpoint-<step>/` |
| Flux smoke | `results/train/flux-cloud-smoke/checkpoint-<step>/` |
| Wan LoRA | `results/train/train_wan_domain_t2v_lora_<timestamp>/checkpoints/only_trained_model/denoiser-*.ckpt` |

**Before terminating the instance:** Syncthing `results/` (and optionally `checkpoints/`) back to your machine.

Resume Wan training:

```bash
export TRAIN_PROFILE=wan-t2v-lora
export RESUME_CKPT=/workspace/results/train/.../checkpoints/...
./cloud/vast/run-train.sh
```

## F. Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM (Flux) | Lower `--resolution` in JSON; keep `gradient_checkpointing: true` |
| CUDA OOM (Wan) | Confirm `poetry run install-deepspeed` succeeded; reduce frames/resolution in YAML |
| HF gated model | Set `HF_TOKEN`; `huggingface-cli login`; accept FLUX.1-dev license |
| flash-attn build fail | `export VIDEOTUNA_ATTN_BACKEND=sdpa` in `.env`; do not run `install-flash-attn` |
| DeepSpeed build fail | Check CUDA toolkit / nvcc; re-run `poetry run install-deepspeed` |
| Wan grey preview | Use `unconditional_guidance_scale: 12.0` in training YAML `image_logger` |
| Provisioning retry | Re-run `bash /workspace/VideoTuna/cloud/vast/bootstrap.sh` (idempotent) |

## G. Cost control

1. Run `./cloud/vast/run-smoke-train.sh` before any multi-hour job.
2. Stop the instance when finished — only persist `results/` and `checkpoints/` via Syncthing.
3. Use smoke configs: `configs/006_flux/cloud_smoke.json`, `configs/008_wanvideo/wan2_1_t2v_14B_lora_cloud_smoke.yaml`.

## Training profiles (`TRAIN_PROFILE`)

| Profile | Default config |
|---------|----------------|
| `flux-lora` | `configs/domain/flux_t2i.json` |
| `wan-t2v-lora` | `configs/domain/wan_t2v_lora.yaml` |

Override with `CONFIG_PATH` and `DATA_CONFIG_PATH` (Flux only) in `.env`.

## Related docs

- [domain-adult-finetune.md](domain-adult-finetune.md) — dataset layout, hyperparameters, inference smoke
- [../checkpoints.md](../checkpoints.md) — weight download layout
- [`cloud/vast/provisioning.yaml`](../../cloud/vast/provisioning.yaml) — manifest source
- [`AGENTS.md`](../../AGENTS.md) — local dev verification gates
