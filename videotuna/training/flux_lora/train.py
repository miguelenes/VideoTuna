"""Accelerate training loop for Flux LoRA fine-tuning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from videotuna.training.flux_lora.checkpoint import save_lora_checkpoint
from videotuna.training.flux_lora.config import (
    FluxLoraTrainConfig,
    load_train_config,
    stamp_output_dir,
)
from videotuna.training.flux_lora.dataset import FluxLoraImageDataset
from videotuna.training.flux_lora.model_utils import load_flux_training_models
from videotuna.utils.logging_config import bound_logger, resolve_device_label

logger = bound_logger(phase="t2i", flow="flux_lora")


def create_flux_accelerator(
    output_dir: Path,
    *,
    mixed_precision: str,
    gradient_accumulation_steps: int = 1,
) -> Accelerator:
    """Build an Accelerate instance with local TensorBoard experiment tracking."""
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "tensorboard"),
    )
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )


def _flux_tracker_config(config: FluxLoraTrainConfig) -> dict[str, Any]:
    return {
        "lora_rank": config.lora_rank,
        "learning_rate": config.learning_rate,
        "max_train_steps": config.max_train_steps,
        "resolution": config.resolution,
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
    }


def _prepare_batch_latents(vae, pixel_values, weight_dtype):
    pixel_values = pixel_values.to(dtype=weight_dtype)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    batch_size, num_channels, height, width = latents.shape
    packed = FluxPipeline._pack_latents(
        latents, batch_size, num_channels, height, width
    )
    return packed, height, width


def _compute_loss(
    pipeline: Any,
    transformer,
    batch,
    weight_dtype,
    accelerator,
) -> torch.Tensor:
    pixel_values = batch["pixel_values"]
    captions = batch["caption"]
    if isinstance(captions, str):
        captions = [captions]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=captions,
            prompt_2=captions,
            device=accelerator.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        model_input, latent_height, latent_width = _prepare_batch_latents(
            pipeline.vae, pixel_values, weight_dtype
        )
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        u = torch.rand(bsz, device=accelerator.device)
        sigmas = u
        timesteps = (sigmas * pipeline.scheduler.config.num_train_timesteps).long()
        sigmas = sigmas.view(-1, 1, 1)
        noisy_input = (1.0 - sigmas) * model_input + sigmas * noise
        target = noise - model_input

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            bsz,
            latent_height // 2,
            latent_width // 2,
            accelerator.device,
            weight_dtype,
        )

    guidance = torch.tensor([1.0], device=accelerator.device, dtype=weight_dtype)
    guidance = guidance.expand(model_input.shape[0])

    model_pred = transformer(
        hidden_states=noisy_input,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def train(config: FluxLoraTrainConfig, data_config) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = create_flux_accelerator(
        output_dir,
        mixed_precision=config.mixed_precision,
    )
    log = logger.bind(device=resolve_device_label(accelerator.device))
    if accelerator.is_main_process:
        log.info("Training Flux LoRA → {}", output_dir)

    dataset = FluxLoraImageDataset(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    components = load_flux_training_models(
        config.pretrained_model_name_or_path,
        lora_rank=config.lora_rank,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    weight_dtype = components["weight_dtype"]
    transformer = components["transformer"]

    pipeline = FluxPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=components["vae"],
        text_encoder=components["text_encoder_one"],
        text_encoder_2=components["text_encoder_two"],
        tokenizer=components["tokenizer_one"],
        tokenizer_2=components["tokenizer_two"],
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    pipeline.text_encoder_2.to(accelerator.device)

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.train_batch_size)
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    pipeline.transformer = accelerator.unwrap_model(transformer)
    accelerator.init_trackers("flux-domain-lora", config=_flux_tracker_config(config))

    progress = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_main_process,
        desc="Flux LoRA",
    )
    global_step = 0
    epoch = 0
    while global_step < max_train_steps:
        epoch += 1
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                loss = _compute_loss(
                    pipeline,
                    transformer,
                    batch,
                    weight_dtype,
                    accelerator,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                accelerator.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                if (
                    global_step % config.checkpointing_steps == 0
                    or global_step == max_train_steps
                ):
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(transformer)
                        ckpt = save_lora_checkpoint(unwrapped, output_dir, global_step)
                        log.info("Saved LoRA checkpoint to {}", ckpt)

                if global_step >= max_train_steps:
                    break

    accelerator.end_training()
    if accelerator.is_main_process:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(
                {
                    "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
                    "lora_rank": config.lora_rank,
                    "max_train_steps": config.max_train_steps,
                    "resolution": config.resolution,
                },
                f,
                indent=2,
            )
        log.info("Training finished. Output: {}", output_dir)


def run_training(
    config_path: str, data_config_path: str, stamp_output: bool = True
) -> None:
    train_cfg, data_cfg = load_train_config(config_path, data_config_path)
    if stamp_output:
        train_cfg.output_dir = stamp_output_dir(train_cfg.output_dir)
        with open(config_path) as f:
            raw = json.load(f)
        raw["--output_dir"] = train_cfg.output_dir
        with open(config_path, "w") as f:
            json.dump(raw, f, indent=4)
    train(train_cfg, data_cfg)
