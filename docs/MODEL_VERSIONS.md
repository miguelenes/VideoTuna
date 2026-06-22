# Model versions

Master reference for VideoTuna inference model families: Hugging Face IDs, Diffusers pipeline classes, integration path, and status.

| Family | Old hub / checkpoint ID | New default hub ID | Pipeline class | Integration | Status |
|--------|-------------------------|-------------------|----------------|-------------|--------|
| CogVideoX T2V | `THUDM/CogVideoX-5b` | `THUDM/CogVideoX1.5-5B` | `CogVideoXPipeline` | `DiffusersVideoFlow` | **upgraded** |
| CogVideoX T2V (smoke) | — | `THUDM/CogVideoX-2b` | `CogVideoXPipeline` | `configs/inference/cogvideox_t2v_2b.yaml` | **current** (CI gate) |
| CogVideoX T2V (legacy) | `THUDM/CogVideoX-5b` | — | `CogVideoXPipeline` | `--model_variant 5b` / `cogvideox_t2v_5b.yaml` | **legacy** |
| CogVideoX I2V | `THUDM/CogVideoX-5b-I2V` | `THUDM/CogVideoX1.5-5B-I2V` | `CogVideoXImageToVideoPipeline` | `DiffusersVideoFlow` | **upgraded** |
| CogVideoX V2V | `THUDM/CogVideoX-5b` | `THUDM/CogVideoX1.5-5B` | `CogVideoXVideoToVideoPipeline` | `cogvideox1.5_v2v_5b.yaml` | **upgraded** |
| CogVideoX 1.5 SAT | local `CogVideoX1.5-5B-SAT` | — | SAT custom | `inference-cogvideox-15-5b-*` | **legacy** (deprecated) |
| Flux T2I | `FLUX.1-dev` / `FLUX.1-schnell` | `black-forest-labs/FLUX.2-dev` | `Flux2Pipeline` | `flux_dev.yaml` | **upgraded** |
| Flux T2I (legacy) | `FLUX.1-*` | — | `FluxPipeline` | `flux1_dev.yaml`, `inference-flux-dev` | **legacy** |
| Flux T2I (fast) | — | `FLUX.2-klein-9B` | `Flux2Pipeline` | `flux2_klein_9b.yaml` | **current** |
| Mochi T2V | `genmo/mochi-1-preview` | *(unchanged)* | `MochiPipeline` | `mochi_t2v.yaml` | **current** |
| Hunyuan T2V | `tencent/HunyuanVideo` | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | `HunyuanVideo15Pipeline` | `DiffusersVideoFlow` | **upgraded** |
| Hunyuan I2V | `tencent/HunyuanVideo-I2V` | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | `HunyuanVideo15ImageToVideoPipeline` | `DiffusersVideoFlow` | **upgraded** |
| Hunyuan (native) | `tencent/HunyuanVideo` | — | custom | `hunyuanvideo.py` | **legacy** (train / fp8) |
| Wan T2V/I2V | `Wan-AI/Wan2.1-*` | `Wan-AI/Wan2.2-*-Diffusers` | `WanPipeline` / `WanImageToVideoPipeline` | `DiffusersVideoFlow` | **upgraded** |
| Wan (native) | `Wan-AI/Wan2.1-*` | `Wan-AI/Wan2.2-T2V-A14B` | vendored Wan2.2 | `wanvideo.py` | **upgraded** |
| Wan lightweight | `Wan2.1-T2V-1.3B` | — | vendored | `t2v-1.3B` task | **legacy** (optional) |
| Open-Sora | v1.0 `.pth` | `hpcai-tech/Open-Sora-v2` | ColossalAI native | `inference-opensora-v2` | **upgraded** (partial) |
| Open-Sora v1 | STDiT v1/v2/v3 | — | native | `inference-opensora-v10-*` | **legacy** |
| StepVideo T2V | `stepfun-ai/stepvideo-t2v` | *(no newer public T2V)* | native | `stepvideo.py` | **current** |
| VideoCrafter 1/2 | `.ckpt` on HF | — | native | `videocrafter.py` | **legacy / frozen** |
| DynamiCrafter I2V | `Doubiiu/DynamiCrafter_1024` | *(no newer HF release)* | native | `inference-dc-i2v-*` | **legacy / frozen** |
| ModelScope V2V | ModelScope API | — | ModelScope | `inference-v2v-ms` | **legacy** (low priority) |
| LTX-Video | — | `Lightricks/LTX-Video` | `LTXPipeline` | `ltx_video.yaml` | **current** (optional lightweight) |

## CogVideoX 1.5 notes

- **Frames:** 81 (or 161); rule 16N+1 for 1.5.
- **FPS:** 16 for export (`savefps: 16`).
- **Resolution:** min(W,H)=768; e.g. 768×1360.
- **Scheduler:** DPM (2b still uses DDIM via hub ID / variant).
- **Legacy SAT:** `poetry run inference-cogvideox-15-5b-t2v` prints a deprecation warning; prefer `poetry run inference-cogvideox1.5-t2v`.

## HunyuanVideo 1.5 notes

- **Frames:** 121 @ 24 fps (720p presets).
- **FP8:** native `--enable_fp8` path **blocked** for 1.5 until official maps exist; use Diffusers offload + VAE tiling.

## CI smoke (canonical)

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/cogvideox_t2v_2b.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload
poetry run pytest tests/test_inference_optimization.py tests/test_import_smoke.py -q
```

Requires NVIDIA CUDA for the inference smoke; pytest gate runs on CPU.
