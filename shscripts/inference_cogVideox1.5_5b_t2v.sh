#!/usr/bin/env bash
# CogVideoX 1.5 T2V via Diffusers (replaces legacy SAT inference_cogVideo_sat_refactor.py).
poetry run inference-cogvideox1.5-t2v \
  --num_inference_steps 4 \
  --enable_model_cpu_offload \
  "$@"
