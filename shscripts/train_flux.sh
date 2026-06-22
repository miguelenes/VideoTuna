#!/usr/bin/env bash
# Flux LoRA fine-tuning via first-party Diffusers trainer (replaces legacy SimpleTuner train_flux.py).
export TOKENIZERS_PARALLELISM=false

poetry run train-flux-lora \
  --config_path configs/006_flux/config.json \
  --data_config_path configs/006_flux/multidatabackend.json \
  "$@"
