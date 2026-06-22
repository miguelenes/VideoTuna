# Multi-GPU inference on VideoTuna

VideoTuna supports several multi-GPU paths. Pick the one that matches your model family.

## Single-process multi-GPU (Diffusers)

For large Diffusers pipelines (Wan A14B, Flux) on a single host:

```shell
CUDA_VISIBLE_DEVICES=0,1 poetry run inference-wan2.2-t2v-720p --device-map auto
```

- Uses `accelerate` `infer_auto_device_map` to spread the transformer across GPUs.
- Slower than native xfuser USP; no sequence parallel.
- Requires `poetry install -E cuda` (accelerate is a core dependency).

## Distributed sequence parallel (xfuser)

Native Hunyuan and Wan flows support Ulysses + Ring attention via [xfuser](https://github.com/xdit-project/xDiT).

### Requirements

- NVIDIA CUDA only (blocked on ROCm).
- `ulysses_degree × ring_degree == WORLD_SIZE` (number of processes).
- No CPU offload when USP is enabled.
- NCCL-compatible driver and peers on the same node.

### Hunyuan native

```shell
torchrun --nproc_per_node=4 scripts/inference_new.py \
  --config configs/007_hunyuanvideo/hunyuanvideo_t2v.yaml \
  --ulysses_degree 2 --ring_degree 2
```

Hunyuan initializes NCCL, sets `cuda:{local_rank}`, and broadcasts weights from rank 0.

### Wan native

```shell
torchrun --nproc_per_node=4 scripts/inference_new.py \
  --config configs/008_wanvideo/wanvideo_t2v_720p.yaml \
  --ulysses_degree 2 --ring_degree 2
```

Wan re-enables `dist.init_process_group` when `WORLD_SIZE > 1`.

### StepVideo tensor parallel

StepVideo uses proprietary CUDA `liboptimus` and xfuser tensor parallel (`tensor_parallel_degree` in config). **CUDA-only** — not available on ROCm.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NCCL_DEBUG=INFO` | Debug collective hangs |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Sometimes stabilizes NCCL + flash attention |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs before `--device cuda:0` remapping |

## Failure modes

| Symptom | Likely cause |
|---------|----------------|
| Hang at init | `ulysses × ring ≠ nproc` or missing `torchrun` |
| OOM on rank > 0 | Model loaded on all ranks without broadcast (check flow logs) |
| xfuser import error | Install CUDA extra: `poetry install -E cuda` |
| StepVideo on ROCm | Use Wan/Hunyuan Diffusers presets instead |

## Training multi-GPU

- **OpenSora:** NCCL via `videotuna/models/opensora/utils/train.py`
- **Lightning scripts:** `--devices N` in Poetry train entrypoints (`scripts/__init__.py`)
- **DeepSpeed:** optional `poetry run install-deepspeed` for ZeRO stage configs in training YAMLs

## Device selection with `CUDA_VISIBLE_DEVICES`

When GPUs are remapped, always use logical indices after remapping:

```shell
CUDA_VISIBLE_DEVICES=1 poetry run inference-hunyuan-t2v --device cuda:0
```

`--device cuda:1` selects the second *visible* GPU.
