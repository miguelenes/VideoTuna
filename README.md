<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/38efb5bc-723e-4012-aebd-f55723c593fb" alt="VideoTuna" style="width: 75%; min-width: 450px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna

![Version](https://img.shields.io/badge/version-0.1.0-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=VideoVerses.VideoTuna&left_color=green&right_color=red)  [![](https://dcbadge.limes.pink/api/server/AammaaR2?style=flat)](https://discord.gg/AammaaR2) <a href='https://github.com/user-attachments/assets/a48d57a3-4d89-482c-8181-e0bce4f750fd'><img src='https://badges.aleen42.com/src/wechat.svg'></a> [![Homepage](https://img.shields.io/badge/Homepage-VideoTuna-orange)](https://videoverses.github.io/videotuna/) [![GitHub](https://img.shields.io/github/stars/VideoVerses/VideoTuna?style=social)](https://github.com/VideoVerses/VideoTuna)


🤗🤗🤗 Videotuna is a useful codebase for text-to-video applications.  
🌟 VideoTuna is the first repo that integrates multiple AI video generation models including `text-to-video (T2V)`, `image-to-video (I2V)`, `text-to-image (T2I)`, and `video-to-video (V2V)` generation for model inference and finetuning (to the best of our knowledge).  
🌟 VideoTuna is the first repo that provides comprehensive pipelines in video generation, from fine-tuning to pre-training, continuous training, and post-training (alignment) (to the best of our knowledge).  



## 🔆 Features
![videotuna-pipeline-fig3](https://github.com/user-attachments/assets/625693d9-b5cf-4c00-8e84-20ea855c2445)
🌟 **All-in-one framework:** Inference and fine-tune various up-to-date pre-trained video generation models.  
🌟 **Continuous training:** Keep improving your model with new data.  
🌟 **Fine-tuning:** Adapt pre-trained models to specific domains.  
🌟 **Human preference alignment:** Leverage RLHF to align with human preferences.  
🌟 **Post-processing:** Enhance and rectify the videos with video-to-video enhancement model.  


## 🔆 Updates

- [2025-04-22] 🐟 Supported **inference** for `Wan2.1` and `Step Video` and **fine-tuning** for `HunyuanVideo T2V`, with a unified codebase architecture.
- [2025-02-03] 🐟 Supported automatic code formatting via [PR#27](https://github.com/VideoVerses/VideoTuna/pull/27). Thanks [@samidarko](https://github.com/samidarko)!
- [2025-02-01] 🐟 Migrated to [Poetry](https://python-poetry.org) for streamlined dependency and script management ([PR#25](https://github.com/VideoVerses/VideoTuna/pull/25)). Thanks [@samidarko](https://github.com/samidarko)!
- [2025-01-20] 🐟 Supported **fine-tuning** for `Flux-T2I`.
- [2025-01-01] 🐟 Released **training** for `VideoVAE+` in the [VideoVAEPlus repo](https://github.com/VideoVerses/VideoVAEPlus).
- [2025-01-01] 🐟 Supported **inference** for `Hunyuan Video` and `Mochi`.
- [2024-12-24] 🐟 Released `VideoVAE+`: a SOTA Video VAE model—now available in [this repo](https://github.com/VideoVerses/VideoVAEPlus)! Achieves better video reconstruction than NVIDIA’s [`Cosmos-Tokenizer`](https://github.com/NVIDIA/Cosmos-Tokenizer).
- [2024-12-01] 🐟 Supported **inference** for `CogVideoX-1.5-T2V&I2V` and `Video-to-Video Enhancement` from ModelScope.
- [2024-12-01] 🐟 Supported **fine-tuning** for `CogVideoX`.
- [2024-11-01] 🐟 🎉 Released **VideoTuna v0.1.0**!  
  Initial support includes inference for `VideoCrafter1-T2V&I2V`, `VideoCrafter2-T2V`, `DynamiCrafter-I2V`, `OpenSora-T2V`, `CogVideoX-1-2B-T2V`, `CogVideoX-1-T2V`, `Flux-T2I`, and training/fine-tuning of `VideoCrafter`, `DynamiCrafter`, and `Open-Sora`.

## 🔆 Get started

### 1.Prepare environment

VideoTuna supports **Poetry** (default) and **[uv](https://docs.astral.sh/uv/)**. The default install is the **inference stack** only; training (including Flux LoRA) uses the optional `training` group.

| Use case | Poetry | uv |
|----------|--------|-----|
| Inference NVIDIA (default) | `poetry install -E cuda` or `poetry install` | `uv sync` |
| Inference AMD ROCm | `poetry install -E rocm` then `poetry run install-rocm` | see [install-rocm.md](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu` then `poetry run install-cpu-torch` | see [install-cpu.md](docs/install-cpu.md) |
| + Training (Wan, Hunyuan, CogVideo, Flux LoRA, Open-Sora, …) | `poetry install -E cuda --with training` | `uv sync --group training` |
| + VBench eval | `poetry install --with eval` | `uv sync --group eval` |
| + Dev (pytest, ruff) | `poetry install --with dev` | `uv sync --group dev` |

See [`docs/vendor-policy.md`](docs/vendor-policy.md) for vendored upstream code and update procedures.

Optional reference submodule (not imported at runtime):

```bash
git submodule update --init videotuna/vendor/simpletuner
```

#### (1) If you use Linux and Conda (Recommend)
``` shell
conda create -n videotuna python=3.11 -y
conda activate videotuna
pip install poetry
poetry install -E cuda   # NVIDIA inference (default stack)
# poetry install --with training  # for fine-tuning (incl. Flux LoRA)
```
- ↑ It takes around 3 minitues.

**AMD ROCm (Linux x86_64)**

```shell
poetry install -E rocm
poetry run install-rocm
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

See [`docs/install-rocm.md`](docs/install-rocm.md) for model tiers, smoke tests, and troubleshooting.

**CPU-only development (Linux / no GPU)**

```shell
poetry install -E cpu --with dev
poetry run install-cpu-torch
poetry run verify-cpu-torch
poetry run pytest tests/ -m "not gpu and not cpu_smoke" -q
```

CPU smoke inference (CogVideoX 2B, tiny resolution — not for production):

```shell
export VIDEOTUNA_ATTN_BACKEND=eager
poetry run inference-cogvideo-t2v-diffusers \
  --config configs/inference/presets/cogvideox_2b_cpu_smoke.yaml \
  --cpu-smoke
```

See [`docs/install-cpu.md`](docs/install-cpu.md) for capability tiers, limitations, and how CPU inference differs from GPU+CPU offload.

**Limitations on CPU:** Wan/StepVideo/Hunyuan 720p, FP8, flash-attn, `torch.compile`, and training are not supported. 14B models at full resolution are impractical on CPU.

**Optional: Flash-attn installation (NVIDIA CUDA only)**

Hunyuan model uses it to reduce memory usage and speed up inference. If it is not installed, the model will run in normal mode. Install the `flash-attn` via:
``` shell
poetry run install-flash-attn 
```
- ↑ It takes 1 minitue.

### Performance tuning

VideoTuna routes attention through a unified backend selector in `videotuna/utils/attention.py`. Control it with environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, `cpu` | `auto` | Override backend detection; `cpu` forces CPU even when a GPU is visible |
| `VIDEOTUNA_CPU_MODE` | `off`, `smoke`, `force` | `off` | CPU inference mode (`smoke` = tiny runs; `force` = debug init). Prefer `--cpu-smoke` CLI flag |
| `VIDEOTUNA_ATTN_BACKEND` | `auto`, `flash`, `sdpa`, `eager` | `auto` | Attention implementation for Hunyuan, OpenSora, Flux, StepVideo, Wan, and diffusers pipelines |
| `VIDEOTUNA_ATTN_BACKEND_STRICT` | `0`, `1` | `0` | When `1`, fail if `flash` requested but flash-attn is missing (default: fall back to sdpa) |
| `VIDEOTUNA_TORCH_COMPILE` | `0`, `1` | `0` | Compile denoiser/transformer forward with `torch.compile` (not VAE or text encoders) |
| `VIDEOTUNA_TORCH_COMPILE_MODE` | `reduce-overhead`, `max-autotune` | `reduce-overhead` | `torch.compile` mode when compile is enabled |
| `VIDEOTUNA_METRICS_OWNER` | `script`, `flow` | `script` | Who writes `metrics.json` (`inference_new` vs per-flow) |

**`auto` resolution:** NVIDIA — `flash` (when `flash-attn` is installed) → `sdpa` → `eager` on CPU. AMD ROCm — `sdpa` → `eager` (flash is never auto-selected).

```shell
# Prefer flash-attn varlen (install optional dependency first)
poetry run install-flash-attn
export VIDEOTUNA_ATTN_BACKEND=flash

# PyTorch SDPA (no flash-attn build required)
export VIDEOTUNA_ATTN_BACKEND=sdpa

# Optional: compile denoiser after warm-up
export VIDEOTUNA_TORCH_COMPILE=1
```

Compare backends on a short CogVideoX diffusers smoke run (`steps=4`):

```shell
poetry run benchmark-attn-backends
poetry run benchmark-attn-backends --json-out results/bench_attn.json
poetry run verify-cuda-extras
```

**Device and VRAM CLI flags** (all `inference_new.py` entrypoints):

```shell
# CPU-only smoke (dev/CI)
poetry run inference-cogvideo-t2v-diffusers \
  --config configs/inference/presets/cogvideox_2b_cpu_smoke.yaml --cpu-smoke

# Select GPU (respects CUDA_VISIBLE_DEVICES remapping)
CUDA_VISIBLE_DEVICES=1 poetry run inference-hunyuan-t2v --device cuda:0

# Named memory presets
poetry run inference-wan2.2-t2v-720p --memory-preset low_vram
poetry run inference-hunyuan1.5-t2v --memory-preset balanced
poetry run inference-cogvideox1.5-t2v --memory-preset max_speed --compile

# Fail before model load when VRAM is insufficient
poetry run inference-hunyuan-t2v --min-vram-gb 48
```

Preset YAMLs live under [`configs/inference/presets/`](configs/inference/presets/). Multi-GPU: see [`docs/multi-gpu.md`](docs/multi-gpu.md).

Sequence parallel (`--ulysses-degree`, `--ring-degree` on Hunyuan/Wan) uses xfuser and is independent of `VIDEOTUNA_ATTN_BACKEND`. The first `torch.compile` iteration is slow; exclude it when timing inference.

**Optional: Video-to-video enhancement**
```
poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
- If this command ↑ get stucked, kill and re-run it will solve the issue.


#### (2) If you use Linux and Poetry (without Conda):
<details>
  <summary>Click to check instructions</summary>
  <br>

  Install Poetry: https://python-poetry.org/docs/#installation  
  Then:

  ``` shell
  poetry config virtualenvs.in-project true # optional but recommended, will ensure the virtual env is created in the project root
  poetry config virtualenvs.create true # enable this argument to ensure the virtual env is created in the project root
  poetry env use python3.11 # will create the virtual env, check with `ls -l .venv`.
  poetry env activate # optional because Poetry commands (e.g. `poetry install` or `poetry run <command>`) will always automatically load the virtual env.
  poetry install          # inference stack (default)
  # poetry install --with training  # fine-tuning (incl. Flux LoRA)
  # poetry install --with dev                             # pytest, ruff
  ```

  **uv (alternative):**

  ``` shell
  uv sync                 # inference stack
  uv sync --group training
  uv run poetry run inference-flux-dev --help  # or: uv run inference-flux-dev if synced
  ```

  **Optional: Flash-attn installation**

  Hunyuan model uses it to reduce memory usage and speed up inference. If it is not installed, the model will run in normal mode. Install the `flash-attn` via:
  ``` shell
  poetry run install-flash-attn
  ```
  
  **Optional: Video-to-video enhancement**
  ```
  poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
  ```
  - If this command ↑ get stucked, kill and re-run it will solve the issue.

</details>



#### (3) If you use MacOS
<details>
  <summary>Click to check instructions</summary>
  <br>

  On MacOS with Apple Silicon chip use [docker compose](https://docs.docker.com/compose/) because some dependencies are not supporting arm64 (e.g. `bitsandbytes`, `decord`, `xformers`).

  First build:

  ```shell
  docker compose build videotuna
  ```

  To preserve the project's files permissions set those env variables:

  ```shell
  export HOST_UID=$(id -u)
  export HOST_GID=$(id -g)
  ```

  Install dependencies:

  ```shell
  docker compose run --remove-orphans videotuna poetry env use /usr/local/bin/python
  docker compose run --remove-orphans videotuna poetry run python -m pip install --upgrade pip setuptools wheel
  docker compose run --remove-orphans videotuna poetry install
  docker compose run --remove-orphans videotuna poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
  ```

  Add a dependency:

  ```shell
  docker compose run --remove-orphans videotuna poetry add wheel
  ```

  Check dependencies:

  ```shell
  docker compose run --remove-orphans videotuna poetry run pip freeze
  ```

  Run Poetry commands:

  ```shell
  docker compose run --remove-orphans videotuna poetry run format
  ```

  Start a terminal:

  ```shell
  docker compose run -it --remove-orphans videotuna bash
  ```
</details>

### 2.Prepare checkpoints

- Please follow [docs/checkpoints.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/checkpoints.md) to download model checkpoints.  
- After downloading, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/blob/main/docs/checkpoints.md#checkpoint-orgnization-structure).

### 3.Inference state-of-the-art T2V/I2V/T2I models


Run the following commands to inference models:
It will automatically perform T2V/T2I based on prompts in `inputs/t2v/prompts.txt`, 
and I2V based on images and prompts in `inputs/i2v/576x1024`.  

**Diffusers models** (CogVideoX, Flux, Mochi, Wan 2.2, HunyuanVideo 1.5, LTX) use `scripts/inference_new.py` with presets under `configs/inference/`. Weights default to Hugging Face hub IDs; override with `--ckpt_path` for offline use. See [docs/MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md).

### Upgrade notes

| From | To | Migration |
|------|-----|-----------|
| CogVideoX 1.5 SAT | Diffusers 1.5 | `poetry run inference-cogvideox1.5-t2v` (81 frames, 16 fps, 768×1360) |
| CogVideoX 5b default | 1.5 default | Old IDs via `--ckpt_path` or `model_variant: 5b` in YAML |
| FLUX.1 aliases | FLUX.2 default | `inference-flux-dev` → FLUX.1; `inference-flux2-dev` → FLUX.2 |
| Wan 2.1 native | Wan 2.2 | Diffusers: `inference-wan2.2-t2v-720p`; native: `configs/008_wanvideo/wan2_2_*` |
| HunyuanVideo | HunyuanVideo 1.5 | `inference-hunyuan1.5-t2v`; native fp8 path not yet on 1.5 |
| Open-Sora v1 | Open-Sora 2.0 | `poetry run inference-opensora-v2` + `checkpoints/open-sora/v2` |

### CI smoke

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/cogvideox_t2v_2b.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload
poetry run pytest tests/test_inference_optimization.py tests/test_import_smoke.py -q
```

```bash
poetry run python scripts/inference_new.py --config configs/inference/cogvideox1.5_t2v_5b.yaml --num_inference_steps 4 --enable_model_cpu_offload
poetry run inference-flux2-dev --enable_model_cpu_offload --num_inference_steps 4
```

**T2V**
Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|T2V|HunyuanVideo|`poetry run inference-hunyuan-t2v`|129|720x1280|32min|60G|
|T2V|WanVideo|`poetry run inference-wanvideo-t2v-720p`|81|720x1280|32min|70G|
|T2V|StepVideo|`poetry run inference-stepvideo-t2v-544x992`|51|544x992|8min|61G|
|T2V|Mochi|`poetry run inference-mochi`|84|480x848|2min|26G (offload+tiling in preset)|
|T2V|CogVideoX1.5-5b|`poetry run inference-cogvideox1.5-t2v`|81|768x1360|~5min|24G (offload)|
|T2V|Wan 2.2 Diffusers|`poetry run inference-wan2.2-t2v-720p`|81|720x1280|TBD|offload preset|
|T2V|HunyuanVideo 1.5|`poetry run inference-hunyuan1.5-t2v`|121|720x1280|TBD|offload preset|
|T2V|LTX-Video|`poetry run inference-ltx-t2v`|121|512x768|TBD|16G+|
|T2V|CogVideoX-5b (legacy)|`poetry run python scripts/inference_new.py --config configs/inference/cogvideox_t2v_5b.yaml`|49|480x720|2min|3G|
|T2V|CogVideoX-2b (smoke)|`poetry run inference-cogvideo-t2v-diffusers`|49|480x720|2min|3G|
|T2V|Open-Sora 2.0|`poetry run inference-opensora-v2`|varies|256px|TBD|see docs|
|T2V|Open Sora V1.0|`poetry run inference-opensora-v10-16x256x256`|16|256x256|11s|24G|
|T2V|VideoCrafter-V2-320x512|`poetry run inference-vc2-t2v-320x512`|16|320x512|26s|11G|
|T2V|VideoCrafter-V1-576x1024|`poetry run inference-vc1-t2v-576x1024`|16|576x1024|2min|15G|

**Low-VRAM presets (≤24GB GPUs)** — metrics written to `metrics.json` beside outputs.

| Tier | Preset | Wan 2.2 720p (approx.) | Hunyuan 720p (approx.) |
|------|--------|------------------------|-------------------------|
| Full GPU | `max_speed` | ~40–48 GB | ~45 GB |
| Balanced | `balanced` | ~24 GB | ~24 GB |
| Low VRAM | `low_vram` | ~12–16 GB | ~16 GB |

*Approximate peaks; use `poetry run benchmark-attn-backends` or `--min-vram-gb` on your hardware.*

|Model|Command|Length|Resolution|Notes|
|:---------|:---------|:---------|:---------|:---------|
|T2V|HunyuanVideo (H800 baseline)|`poetry run inference-hunyuan-t2v`|129|720×1280|~32min, ~60GB peak VRAM on H800|
|T2V|HunyuanVideo (24GB preset)|`poetry run inference-hunyuan-t2v --memory-preset balanced`|129|720×1280|Or `--enable_sequential_cpu_offload --enable_vae_tiling --dtype bf16`|
|T2V|WanVideo (H800 baseline)|`poetry run inference-wanvideo-t2v-720p`|81|720×1280|~32min, ~70GB full GPU|
|T2V|WanVideo (24GB)|`poetry run inference-wanvideo-t2v-720p --memory-preset low_vram`|81|720×1280|~12–16 GB with sequential offload + fp16|

Shared inference flags (all `inference_new.py` models): `--device` / `--gpu-id`, `--min-vram-gb`, `--memory-preset low_vram|balanced|max_speed`, `--enable_vae_tiling`, `--enable_vae_slicing`, `--enable_model_cpu_offload`, `--enable_sequential_cpu_offload`, `--dtype bf16|fp16`, `--device-map auto` (Diffusers multi-GPU), `--fuse_qkv`, `--enable_attention_cache`, `--ulysses_degree`, `--ring_degree`, `--compile`, `--enable_fp8` (Hunyuan).

**Hardware:** Native Hunyuan/Wan/StepVideo 720p flows need a **GPU accelerator** (NVIDIA CUDA or AMD ROCm). Default install uses PyTorch+cu126 (`poetry install -E cuda`); AMD users: `poetry install -E rocm` + `poetry run install-rocm` — see [docs/install-rocm.md](docs/install-rocm.md). **Tier A** diffusers models (CogVideoX, Flux, Wan 2.2 Diffusers, Hunyuan 1.5) are the recommended ROCm path. StepVideo is **CUDA-only** (proprietary liboptimus). CPU-only dev: `poetry run pytest tests/test_inference_optimization.py`.

Legacy diffusers Hunyuan T2V (256×256 training workflow): `poetry run inference-hunyuan-t2v-diffusers`.

---


**I2V**


Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|I2V|WanVideo|`poetry run inference-wanvideo-i2v-720p `|81|720x1280|28min|77G|
|I2V|HunyuanVideo|`poetry run inference-hunyuan-i2v-720p`|129|720x1280|29min|43G|
|I2V|CogVideoX1.5-5B-I2V|`poetry run inference-cogvideox1.5-i2v`|81|768x1360|~5min|24G (offload)|
|I2V|Wan 2.2 Diffusers|`poetry run inference-wan2.2-i2v-720p`|81|720x1280|TBD|offload preset|
|I2V|HunyuanVideo 1.5|`poetry run inference-hunyuan1.5-i2v`|121|720x1280|TBD|offload preset|
|I2V|CogVideoX-5b-I2V (legacy)|`poetry run inference-cogvideo-i2v-diffusers`|49|480x720|5min|5G|
|I2V|DynamiCrafter|`poetry run inference-dc-i2v-576x1024`|16|576x1024|2min|53G|
|I2V|VideoCrafter-V1|`poetry run inference-vc1-i2v-320x512`|16|320x512|26s|11G|


---

**T2I**

Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|T2I|Flux2-dev (default)|`poetry run inference-flux2-dev`|1|768x1360|TBD|62G+ / offload|
|T2I|Flux2-klein-9b|`poetry run inference-flux2-klein-9b`|1|768x1360|~1s|29G|
|T2I|Flux1-dev (legacy)|`poetry run inference-flux-dev`|1|768x1360|4s|37G|
|T2I|Flux1-dev + offload|`poetry run inference-flux-dev --enable_vae_tiling --enable_sequential_cpu_offload`|1|768x1360|4.2min|2G|
|T2I|Flux1-schnell (legacy)|`poetry run inference-flux-schnell`|1|768x1360|1s|37G|
|T2I|Flux1-schnell + offload|`poetry run inference-flux-schnell --enable_vae_tiling --enable_sequential_cpu_offload`|1|768x1360|24s|2G|

### 4. Finetune T2V models
#### (1) Prepare dataset
Please follow the [docs/datasets.md](docs/datasets.md) to try provided toydataset or build your own datasets.

#### (2) Fine-tune
All  training commands were tested on H800 80G GPUs.  
**T2V**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|T2V|Wan Video|Lora Fine-tune|`poetry run train-wan2-1-t2v-lora`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|T2V|Wan Video|Full Fine-tune|`poetry run train-wan2-1-t2v-fullft`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|T2V|Hunyuan Video|Lora Fine-tune|`poetry run train-hunyuan-t2v-lora`|[docs/finetune_hunyuanvideo.md](docs/finetune_hunyuanvideo.md)|2|
|T2V|CogvideoX|Lora Fine-tune|`poetry run train-cogvideox-t2v-lora`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|1|
|T2V|CogvideoX|Full Fine-tune|`poetry run train-cogvideox-t2v-fullft`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|4|
|T2V|Open-Sora v1.0|Full Fine-tune|`poetry run train-opensorav10`|-|1|
|T2V|VideoCrafter|Lora Fine-tune|`poetry run train-videocrafter-lora`|[docs/finetune_videocrafter.md](docs/finetune_videocrafter.md)|1|
|T2V|VideoCrafter|Full Fine-tune|`poetry run train-videocrafter-v2`|[docs/finetune_videocrafter.md](docs/finetune_videocrafter.md)|1|

---

**I2V**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|I2V|Wan Video|Lora Fine-tune|`poetry run train-wan2-1-i2v-lora`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|I2V|Wan Video|Full Fine-tune|`poetry run train-wan2-1-i2v-fullft`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|I2V|CogvideoX|Lora Fine-tune|`poetry run train-cogvideox-i2v-lora`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|1|
|I2V|CogvideoX|Full Fine-tune|`poetry run train-cogvideox-i2v-fullft`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|4|

---

**T2I**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|T2I|Flux|Lora Fine-tune|`poetry run train-flux-lora`|[docs/finetune_flux.md](docs/finetune_flux.md)|1|


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance.
Please check [eval/README.md](docs/evaluation.md) for details.

<!-- ### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details. -->

## Contribute

## Git hooks

Git hooks are handled with [pre-commit](https://pre-commit.com) library.

### Hooks installation

Run the following command to install hooks on `commit`. They will check formatting, linting and types.

```shell
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

### Running the hooks without commiting

```shell
poetry run pre-commit run --all-files
```

## Acknowledgement
We thank the following repos for sharing their awesome models and codes!

* [Wan2.1](https://github.com/Wan-Video/Wan2.1): Wan: Open and Advanced Large-Scale Video Generative Models.
* [HunyuanVideo](https://github.com/Tencent/HunyuanVideo): A Systematic Framework For Large Video Generation Model.
* [Step-Video](https://github.com/stepfun-ai/Step-Video-T2V): A text-to-video pre-trained model with 30 billion parameters and the capability to generate videos up to 204 frames.
* [Mochi](https://www.genmo.ai/blog): A new SOTA in open-source video generation models
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): Overcoming Data Limitations for High-Quality Video Diffusion Models
* [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Open Diffusion Models for High-Quality Video Generation
* [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All
* [CogVideoX](https://github.com/THUDM/CogVideo): Text-to-Video Diffusion Models with An Expert Transformer
* [VADER](https://github.com/mihirp1998/VADER): Video Diffusion Alignment via Reward Gradients
* [VBench](https://github.com/Vchitect/VBench): Comprehensive Benchmark Suite for Video Generative Models
* [Flux](https://github.com/black-forest-labs/flux): Text-to-image models from Black Forest Labs.
* [SimpleTuner](https://github.com/bghira/SimpleTuner): Upstream inspiration for Flux LoRA configs (replaced by first-party trainer in VideoTuna).




## Some Resources
* [LLMs-Meet-MM-Generation](https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation): A paper collection of utilizing LLMs for multimodal generation (image, video, 3D and audio).
* [MMTrail](https://github.com/litwellchi/MMTrail): A multimodal trailer video dataset with language and music descriptions.
* [Seeing-and-Hearing](https://github.com/yzxing87/Seeing-and-Hearing): A versatile framework for Joint VA generation, V2A, A2V, and I2A.
* [Self-Cascade](https://github.com/GuoLanqing/Self-Cascade): A Self-Cascade model for higher-resolution image and video generation.
* [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter) and [HiPrompt](https://liuxinyv.github.io/HiPrompt/): Free method for higher-resolution image and video generation.
* [FreeTraj](https://github.com/arthur-qiu/FreeTraj) and [FreeNoise](https://github.com/AILab-CVC/FreeNoise): Free method for video trajectory control and longer-video generation.
* [Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji), [Follow-Your-Click](https://github.com/mayuelala/FollowYourClick), and [Follow-Your-Pose](https://follow-your-pose.github.io/): Follow family for controllable video generation.
* [Animate-A-Story](https://github.com/AILab-CVC/Animate-A-Story): A framework for storytelling video generation.
* [LVDM](https://github.com/YingqingHe/LVDM): Latent Video Diffusion Model for long video generation and text-to-video generation.



## 🍻 Contributors

<a href="https://github.com/VideoVerses/VideoTuna/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=VideoVerses/VideoTuna" />
</a>

## Upgrade notes

VideoTuna v0.1.0+ targets **Python 3.11**, **PyTorch 2.6 (CUDA 12.6)**, and **diffusers ≥ 0.35.2**. Key changes when upgrading from older installs:

| Area | Before | After |
|------|--------|-------|
| Python | 3.10 | **3.11** (`decord==0.6.0` has no reliable 3.12 wheels) |
| PyTorch / torchvision | 2.2.2 / 0.17.2 | **2.6.0+cu126 / 0.21.0+cu126** (via Poetry `pytorch-cu126` source) |
| diffusers / transformers | 0.32 / 4.46 | **≥ 0.35.2 / ≥ 4.48** |
| accelerate / peft | 0.33 / 0.12 | **≥ 1.2 / ≥ 0.17** |
| deepspeed / xformers | 0.16.5 / 0.0.25 | **0.19.x / 0.0.29.post3** |
| flash-attn (optional) | 2.7.3 + CUDA 12.1 | **2.7.4.post1 + CUDA 12.6** (`cxx11abiTRUE` wheel) |

**CUDA driver:** PyTorch `cu126` wheels require an NVIDIA driver compatible with CUDA 12.6+.

| Driver (min) | CUDA | PyTorch wheel | Notes |
|--------------|------|---------------|-------|
| ≥ 550.54 | 12.6 | `cu126` (default) | `poetry install -E cuda` |
| ≥ 545.x | 12.4 | `cu124` (optional) | Swap torch source to `pytorch-cu124`; see extras `cuda124` |
| ≥ 525.x | 12.1 | legacy | Not supported in v0.1.0 default lockfile |

**GPU architecture (`TORCH_CUDA_ARCH_LIST`) when building CUDA extensions:**

| Family | Example GPUs | `TORCH_CUDA_ARCH_LIST` |
|--------|--------------|------------------------|
| Turing | T4, RTX 20xx | `7.5` |
| Ampere | A100, RTX 30xx | `8.0;8.6` |
| Ada | RTX 4090, L40 | `8.9` |
| Hopper | H100, H800 | `9.0` |

Verify optional NVIDIA packages: `poetry run verify-cuda-extras` (add `--expect-flash` on GPU CI).

**Poetry install on Linux:** `torch`, `torchvision`, and `xformers` resolve from the explicit `pytorch-cu126` index; NVIDIA CUDA runtime packages and `triton` are listed in `pyproject.toml` so `poetry install` is self-contained on Linux x86_64.

**Diffusers API:** prefer `dtype=` over deprecated `torch_dtype=` in `from_pretrained()` calls (both still work in diffusers 0.35).

**Optional install helpers** (Conda + NVIDIA GPU recommended):

```shell
poetry run install-flash-attn   # flash-attn 2.7.4.post1, CUDA 12.6
poetry run install-deepspeed    # deepspeed 0.19.2, CUDA 12.6
```

**Useful environment variables:**

- `TOKENIZERS_PARALLELISM=false` — set automatically by training scripts; avoids HF tokenizer fork warnings.
- `CUDA_HOME` — required for building flash-attn or DeepSpeed ops from source.
- `TORCH_CUDA_ARCH_LIST` — GPU architectures when compiling CUDA extensions (e.g. `8.0;8.6;9.0`).
- `DS_BUILD_CPU_ADAM=1` — enables CPU Adam op when building DeepSpeed (set by `install-deepspeed`).
- `DS_BUILD_OPS=0` — skip optional DeepSpeed CUDA op builds for faster install.

**OpenSora / ColossalAI:** `colossalai` remains pinned at **0.3.6** because newer releases declare incompatible `diffusers`/`transformers` pins. OpenSora training still uses ColossalAI; other backends use the upgraded HF stack.

## 📋 License
Please follow [CC-BY-NC-ND](./LICENSE). If you want a license authorization, please contact the project leads Yingqing He (yhebm@connect.ust.hk) and Yazhou Xing (yxingag@connect.ust.hk).

## 😊 Citation

```bibtex
@software{videotuna,
  author = {Yingqing He and Yazhou Xing and Zhefan Rao and Haoyu Wu and Zhaoyang Liu and Jingye Chen and Pengjun Fang and Jiajun Li and Liya Ji and Runtao Liu and Xiaowei Chi and Yang Fei and Guocheng Shao and Yue Ma and Qifeng Chen},
  title = {VideoTuna: A Powerful Toolkit for Video Generation with Model Fine-Tuning and Post-Training},
  month = {Nov},
  year = {2024},
  url = {https://github.com/VideoVerses/VideoTuna}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VideoVerses/VideoTuna&type=Date)](https://star-history.com/#VideoVerses/VideoTuna&Date)
