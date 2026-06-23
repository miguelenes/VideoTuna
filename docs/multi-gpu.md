# Multi-GPU inference and training on PrivTune

PrivTune supports multi-GPU paths for Wan 2.2 Diffusers validation, native Wan
distributed inference, and Wan/Flux LoRA training.

## Quick validation

Before launching any multi-GPU workload, run the validation CLI to check your
environment and print a safe launch command:

```shell
# Inference: device-map auto (Diffusers)
poetry run validate-multi-gpu inference --mode device_map --gpu-ids 0,1 --dry-run

# Inference: xfuser USP (native sequence parallel)
poetry run validate-multi-gpu inference --mode xfuser --gpu-ids 0,1,2,3 --ulysses-degree 2 --ring-degree 2 --dry-run

# Training: Wan Lightning
poetry run validate-multi-gpu training --mode wan_lightning --gpu-ids 0,1,2,3 --devices 0,1,2,3 --dry-run

# Training: Flux Accelerate
poetry run validate-multi-gpu training --mode flux_accelerate --gpu-ids 0,1,2,3 --num-processes 4 --dry-run
```

The validator checks:

| Check | device_map | xfuser | wan_lightning | flux_accelerate |
|-------|-----------|--------|---------------|-----------------|
| ROCm incompatible | allowed | **fatal** | **fatal** | allowed |
| CPU offload + multi-GPU | **fatal** | **fatal** | n/a | n/a |
| `ulysses × ring == nproc` | n/a | **fatal** | n/a | n/a |
| `< 2 GPUs visible` | warning | **fatal** | warning | warning |
| NCCL available | n/a | warning | warning | warning |
| CUDA torch | **fatal** | **fatal** | **fatal** | **fatal** |
| DeepSpeed installed | n/a | n/a | warning | n/a |
| `--num_processes > GPUs` | n/a | n/a | n/a | **fatal** |
| `--devices > GPUs` | n/a | n/a | **fatal** | n/a |
| Accelerate installed | **fatal** | n/a | n/a | **fatal** |

Troubleshoot known failure modes:

```shell
poetry run validate-multi-gpu diagnose hang
poetry run validate-multi-gpu diagnose oom
poetry run validate-multi-gpu diagnose xfuser_import_error
poetry run validate-multi-gpu diagnose xfuser_rocm
```

## Single-process multi-GPU (Diffusers)

For Wan 2.2 on a single host:

```shell
CUDA_VISIBLE_DEVICES=0,1 poetry run inference-wan2.2-t2v-720p --device-map auto --max-memory-per-gpu 22GiB
```

- Uses `accelerate` `infer_auto_device_map` to spread the transformer across GPUs.
- Requires `poetry install -E cuda` (accelerate is a core dependency).
- CPU offload and `device_map=auto` are **mutually exclusive** (the validator catches this).
- Customize per-GPU memory budget with `--max-memory-per-gpu` (default `22GiB`).

## Distributed sequence parallel (xfuser)

Native Wan flows support Ulysses + Ring attention via [xfuser](https://github.com/xdit-project/xDiT).

### Requirements

- NVIDIA CUDA only (blocked on ROCm — validated at launch).
- `ulysses_degree × ring_degree == WORLD_SIZE` (number of processes).
- No CPU offload when USP is enabled (validated at launch).
- NCCL-compatible driver and peers on the same node.

### Wan native

```shell
torchrun --nproc_per_node=4 scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --ulysses_degree 2 --ring_degree 2
```

Wan re-enables `dist.init_process_group` when `WORLD_SIZE > 1`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NCCL_DEBUG=INFO` | Debug collective hangs (included in generated xfuser commands) |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Sometimes stabilizes NCCL + flash attention (included in generated xfuser commands) |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs before `--device cuda:0` remapping |

## Failure modes

| Symptom | Likely cause |
|---------|----------------|
| Hang at init | `ulysses × ring ≠ nproc` or missing `torchrun` |
| OOM on rank > 0 | Model loaded on all ranks without broadcast (check flow logs) |
| xfuser import error | Install CUDA extra: `poetry install -E cuda` |
| xfuser on ROCm | Use single-GPU Wan 2.2 Diffusers with `VIDEOTUNA_ATTN_BACKEND=sdpa` |

## Training multi-GPU

- **Wan Lightning:** `--devices N` in `train-wan2-1-t2v-lora` / `scripts/train_new.py`.
  Validate and generate safe commands: `poetry run validate-multi-gpu training --mode wan_lightning --gpu-ids 0,1,2,3 --devices "0,1,2,3" --dry-run`
- **DeepSpeed:** `poetry run install-deepspeed` for ZeRO stage configs in domain YAMLs.
- **Flux Accelerate:** `--num_processes N` in `accelerate launch`. Currently hardcoded to 1;
  the validator generates the correct `accelerate launch` argv for multi-process:
  `poetry run validate-multi-gpu training --mode flux_accelerate --gpu-ids 0,1,2,3 --num-processes 4 --dry-run`

## Device selection with `CUDA_VISIBLE_DEVICES`

When GPUs are remapped, always use logical indices after remapping:

```shell
CUDA_VISIBLE_DEVICES=1 poetry run inference-wan2.2-t2v-720p --device cuda:0
```

`--device cuda:1` selects the second *visible* GPU.

## max_memory_per_gpu for device_map

When using `--device-map auto` with unequal GPUs, set `--max-memory-per-gpu` to
reflect the smallest GPU's VRAM:

```shell
CUDA_VISIBLE_DEVICES=0,1 poetry run inference-wan2.2-t2v-720p \
  --device-map auto --max-memory-per-gpu 22GiB
```

This value is passed to `accelerate.infer_auto_device_map(max_memory=...)` and
prevents OOM on smaller devices.
