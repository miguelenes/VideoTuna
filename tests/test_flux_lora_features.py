"""Unit tests for Flux LoRA bucketing, checkpoints, cache, and strict config."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

from videotuna.training.flux_lora.bucketing import (
    bucket_dimensions,
    bucket_dimensions_for_image,
    meets_minimum_size,
    round_aspect_ratio,
    target_pixel_area,
)
from videotuna.training.flux_lora.checkpoint import (
    find_latest_checkpoint,
    has_accelerate_state,
    has_lora_weights,
    load_lora_checkpoint,
    prune_checkpoints,
)
from videotuna.training.flux_lora.config import FluxLoraTrainConfig, load_train_config
from videotuna.training.flux_lora.text_embed_cache import build_or_load_cache
from videotuna.training.flux_lora.train import (
    _resolve_resume_checkpoint,
    _run_training_loop,
    _run_validation,
    create_flux_accelerator,
    run_training,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


def test_round_aspect_ratio():
    assert round_aspect_ratio(1920, 1080, 2) == 1.78


def test_target_pixel_area():
    assert target_pixel_area(512) == 512 * 512


def test_bucket_dimensions_align_to_64():
    width, height = bucket_dimensions(1.78, target_pixel_area(512))
    assert width % 64 == 0
    assert height % 64 == 0


def test_bucket_dimensions_square():
    width, height = bucket_dimensions_for_image(1000, 1000, 512, "pixel_area", 2)
    assert width == height


def test_meets_minimum_size_disabled():
    assert meets_minimum_size(64, 64, 0, "pixel_area")


def test_meets_minimum_size_pixel_area():
    assert meets_minimum_size(512, 512, 512, "pixel_area")
    assert not meets_minimum_size(256, 256, 512, "pixel_area")


def test_flux_domain_config_loads_all_fields():
    train_cfg, data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    assert train_cfg.gradient_checkpointing is True
    assert train_cfg.disable_tf32 is True
    assert train_cfg.disable_benchmark is False
    assert train_cfg.validation_steps == 40
    assert train_cfg.validation_num_inference_steps == 10
    assert train_cfg.checkpoints_total_limit == 20
    assert train_cfg.resume_from_checkpoint == "latest"
    assert train_cfg.gradient_accumulation_steps == 1
    assert train_cfg.write_batch_size == 1
    assert train_cfg.aspect_bucket_rounding == 2
    assert train_cfg.resolution_type == "pixel_area"
    assert data_cfg.text_embeds is not None
    assert data_cfg.text_embeds.cache_dir == "cache/text/flux/domain-adult"


def test_unknown_config_key_raises(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--unknown_key":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_train_config(config_path, data_path)


def test_invalid_lora_type_raises(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--lora_type":"dora"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lora_type"):
        load_train_config(config_path, data_path)


def test_invalid_num_train_epochs_raises(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--num_train_epochs":5}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_train_epochs"):
        load_train_config(config_path, data_path)


def test_find_latest_checkpoint(tmp_path):
    for step in (10, 250, 50):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        (d / "optimizer.bin").write_bytes(b"")
        (d / "adapter_config.json").write_bytes(b"{}")
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-250"


def test_prune_checkpoints(tmp_path):
    for step in (10, 50, 100, 250):
        (tmp_path / f"checkpoint-{step}").mkdir()
    prune_checkpoints(tmp_path, 2)
    remaining = sorted(path.name for path in tmp_path.iterdir() if path.is_dir())
    assert remaining == ["checkpoint-100", "checkpoint-250"]


def test_build_or_load_cache_batches_encode_prompt(tmp_path):
    cache_dir = tmp_path / "cache"
    captions = ["caption-a", "caption-b", "caption-c"]
    calls: list[int] = []

    def fake_encode_prompt(*, prompt, **_kwargs):
        calls.append(len(prompt))
        batch = len(prompt)
        return (
            torch.zeros(batch, 2, 4),
            torch.zeros(batch, 3),
            torch.zeros(batch, 2, 2),
        )

    pipeline = mock.MagicMock()
    pipeline.encode_prompt.side_effect = fake_encode_prompt

    lookup = build_or_load_cache(
        pipeline,
        captions,
        cache_dir,
        write_batch_size=2,
        device=torch.device("cpu"),
    )
    assert set(lookup) == set(captions)
    assert calls == [2, 1]
    assert len(list(cache_dir.glob("*.pt"))) == 3

    pipeline.encode_prompt.reset_mock()
    second_lookup = build_or_load_cache(
        pipeline,
        captions,
        cache_dir,
        write_batch_size=2,
        device=torch.device("cpu"),
    )
    assert second_lookup.keys() == lookup.keys()
    pipeline.encode_prompt.assert_not_called()


def _minimal_flux_data_backend(tmp_path: Path) -> Path:
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    return data_path


def test_gradient_checkpointing_false_parses_from_json(tmp_path):
    data_path = _minimal_flux_data_backend(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--gradient_checkpointing":"false"}',
        encoding="utf-8",
    )
    train_cfg, _ = load_train_config(config_path, data_path)
    assert train_cfg.gradient_checkpointing is False


def test_load_flux_training_models_respects_gradient_checkpointing_false():
    pytest.importorskip("peft")
    from videotuna.training.flux_lora import model_utils

    mock_peft_model = mock.MagicMock()
    mock_peft_model.enable_gradient_checkpointing = mock.MagicMock()
    mock_peft_model.disable_gradient_checkpointing = mock.MagicMock()
    mock_component = mock.MagicMock()

    with (
        mock.patch.object(model_utils, "CLIPTokenizer") as tokenizer_cls,
        mock.patch.object(model_utils, "T5TokenizerFast") as tokenizer_two_cls,
        mock.patch.object(model_utils, "CLIPTextModel") as text_encoder_one_cls,
        mock.patch.object(model_utils, "T5EncoderModel") as text_encoder_two_cls,
        mock.patch.object(model_utils, "AutoencoderKL") as vae_cls,
        mock.patch.object(model_utils, "FluxTransformer2DModel") as transformer_cls,
        mock.patch.object(model_utils, "get_peft_model", return_value=mock_peft_model),
    ):
        tokenizer_cls.from_pretrained.return_value = mock_component
        tokenizer_two_cls.from_pretrained.return_value = mock_component
        text_encoder_one_cls.from_pretrained.return_value = mock_component
        text_encoder_two_cls.from_pretrained.return_value = mock_component
        vae_cls.from_pretrained.return_value = mock_component
        transformer_cls.from_pretrained.return_value = mock_component

        model_utils.load_flux_training_models(
            "black-forest-labs/FLUX.1-dev",
            lora_rank=4,
            gradient_checkpointing=False,
        )

    mock_peft_model.enable_gradient_checkpointing.assert_not_called()
    mock_peft_model.disable_gradient_checkpointing.assert_called_once()


def test_run_validation_writes_preview(tmp_path):
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        validation_prompt="sks_style, portrait",
        validation_steps=40,
        validation_resolution="512x512",
        validation_num_inference_steps=4,
    )
    mock_image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    mock_pipeline = mock.MagicMock()
    mock_pipeline.return_value = mock.MagicMock(images=[mock_image])
    mock_pipeline.transformer = mock.MagicMock()

    mock_writer = mock.MagicMock()
    mock_tracker = mock.MagicMock(writer=mock_writer)
    mock_accelerator = mock.MagicMock(
        is_main_process=True,
        device=torch.device("cpu"),
        trackers=[mock_tracker],
    )
    mock_log = mock.MagicMock()

    _run_validation(
        mock_pipeline,
        config,
        tmp_path,
        global_step=40,
        accelerator=mock_accelerator,
        weight_dtype=torch.bfloat16,
        log=mock_log,
        metrics_backend="tensorboard",
    )

    image_path = tmp_path / "validation" / "step-000040.png"
    assert image_path.is_file()
    mock_writer.add_image.assert_called_once()
    mock_pipeline.transformer.train.assert_called_once()


def test_run_validation_skips_when_step_not_divides(tmp_path):
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        validation_prompt="sks_style, portrait",
        validation_steps=40,
    )
    mock_pipeline = mock.MagicMock()
    mock_accelerator = mock.MagicMock(
        is_main_process=True,
        device=torch.device("cpu"),
        trackers=[],
    )

    _run_validation(
        mock_pipeline,
        config,
        tmp_path,
        global_step=39,
        accelerator=mock_accelerator,
        weight_dtype=torch.bfloat16,
        log=mock.MagicMock(),
        metrics_backend="tensorboard",
    )

    mock_pipeline.assert_not_called()
    assert not (tmp_path / "validation").exists()


def test_resolve_resume_checkpoint_latest(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-50" / "optimizer.bin").write_bytes(b"")
    (tmp_path / "checkpoint-50" / "adapter_config.json").write_bytes(b"{}")
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    resolved = _resolve_resume_checkpoint(config, tmp_path)
    assert resolved is not None
    assert resolved.name == "checkpoint-50"


def test_resolve_resume_checkpoint_relative_path(tmp_path):
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-100" / "optimizer.bin").write_bytes(b"")
    (tmp_path / "checkpoint-100" / "adapter_config.json").write_bytes(b"{}")
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="checkpoint-100",
    )
    resolved = _resolve_resume_checkpoint(config, tmp_path)
    assert resolved is not None
    assert resolved.name == "checkpoint-100"


def test_resolve_resume_checkpoint_missing_returns_none(tmp_path):
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    assert _resolve_resume_checkpoint(config, tmp_path) is None


def test_invalid_resume_from_checkpoint_raises(tmp_path):
    data_path = _minimal_flux_data_backend(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--resume_from_checkpoint":""}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        load_train_config(config_path, data_path)


def test_gradient_accumulation_steps_parses_from_json(tmp_path):
    data_path = _minimal_flux_data_backend(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--gradient_accumulation_steps":4}',
        encoding="utf-8",
    )
    train_cfg, _ = load_train_config(config_path, data_path)
    assert train_cfg.gradient_accumulation_steps == 4


def test_invalid_gradient_accumulation_steps_raises(tmp_path):
    data_path = _minimal_flux_data_backend(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10,'
        '"--gradient_accumulation_steps":0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        load_train_config(config_path, data_path)


def test_create_flux_accelerator_passes_gradient_accumulation_steps(tmp_path):
    target = "videotuna.training.flux_lora.train.Accelerator"
    with mock.patch(target) as accelerator_cls:
        create_flux_accelerator(
            tmp_path,
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        )
    accelerator_cls.assert_called_once()
    assert accelerator_cls.call_args.kwargs["gradient_accumulation_steps"] == 4


def test_run_training_skips_stamp_when_resuming(tmp_path):
    data_path = _minimal_flux_data_backend(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    train_cfg = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path / "run"),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    data_cfg = mock.MagicMock()
    with (
        mock.patch(
            "videotuna.training.flux_lora.train.load_train_config",
            return_value=(train_cfg, data_cfg),
        ),
        mock.patch("videotuna.training.flux_lora.train.stamp_output_dir") as stamp,
        mock.patch("videotuna.training.flux_lora.train.train") as train_fn,
    ):
        run_training(str(config_path), str(data_path))
    stamp.assert_not_called()
    train_fn.assert_called_once_with(train_cfg, data_cfg)


def test_train_raises_when_resume_checkpoint_missing(tmp_path):
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    data_config = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    with (
        mock.patch(
            "videotuna.training.flux_lora.train.load_flux_training_models",
            return_value={
                "weight_dtype": torch.bfloat16,
                "transformer": mock_transformer,
                "vae": mock.MagicMock(),
                "text_encoder_one": mock.MagicMock(),
                "text_encoder_two": mock.MagicMock(),
                "tokenizer_one": mock.MagicMock(),
                "tokenizer_two": mock.MagicMock(),
            },
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.create_flux_accelerator",
            return_value=mock.MagicMock(
                device=torch.device("cpu"),
                is_main_process=True,
            ),
        ),
    ):
        with pytest.raises(ValueError, match="No checkpoint found"):
            train(config, data_config)


def test_has_accelerate_state_false_when_empty(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    assert has_accelerate_state(ckpt) is False


def test_has_accelerate_state_true_with_optimizer_bin(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "optimizer.bin").write_bytes(b"")
    assert has_accelerate_state(ckpt) is True


def test_has_accelerate_state_true_with_scheduler_pt(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "scheduler.pt").write_bytes(b"")
    assert has_accelerate_state(ckpt) is True


def test_has_accelerate_state_true_with_random_states(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "random_states_0.pkl").write_bytes(b"")
    assert has_accelerate_state(ckpt) is True


def test_has_accelerate_state_false_for_nonexistent_dir(tmp_path):
    assert has_accelerate_state(tmp_path / "nope") is False


def test_resolve_resume_checkpoint_returns_none_without_accelerate_state(tmp_path):
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    assert _resolve_resume_checkpoint(config, tmp_path) is None


def test_resolve_resume_checkpoint_returns_path_with_accelerate_state(tmp_path):
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "optimizer.bin").write_bytes(b"")
    (ckpt / "scheduler.pt").write_bytes(b"")
    (ckpt / "adapter_config.json").write_bytes(b"{}")
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
    )
    resolved = _resolve_resume_checkpoint(config, tmp_path)
    assert resolved is not None
    assert resolved.name == "checkpoint-50"


def test_save_state_called_at_checkpoint_step(tmp_path):
    """accelerator.save_state is called after save_lora_checkpoint."""
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        max_train_steps=1,
        checkpointing_steps=1,
        checkpoints_total_limit=None,
    )
    mock_accelerator = mock.MagicMock()
    mock_accelerator.is_main_process = True
    mock_accelerator.sync_gradients = True
    mock_accelerator.accumulate.return_value.__enter__ = mock.MagicMock()
    mock_accelerator.accumulate.return_value.__exit__ = mock.MagicMock(
        return_value=False
    )
    mock_accelerator.device = torch.device("cpu")
    mock_accelerator.unwrap_model.return_value = mock.MagicMock()

    mock_pipeline = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_dataloader = mock.MagicMock()
    mock_optimizer = mock.MagicMock()
    mock_lr_scheduler = mock.MagicMock()

    batch = {"pixel_values": torch.zeros(1, 3, 64, 64), "caption": ["test"]}
    mock_dataloader.__iter__ = mock.MagicMock(return_value=iter([batch]))

    with (
        mock.patch(
            "videotuna.training.flux_lora.train._compute_loss",
            return_value=torch.tensor(1.0, requires_grad=True),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_validation",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.save_lora_checkpoint",
            return_value=tmp_path / "checkpoint-1",
        ) as save_ckpt,
        mock.patch(
            "videotuna.training.flux_lora.train.prune_checkpoints",
        ),
    ):
        _run_training_loop(
            config=config,
            output_dir=tmp_path,
            pipeline=mock_pipeline,
            transformer=mock_transformer,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            accelerator=mock_accelerator,
            weight_dtype=torch.bfloat16,
            global_step=0,
            max_train_steps=1,
            log=mock.MagicMock(),
            metrics_backend="tensorboard",
        )

    save_ckpt.assert_called_once()
    mock_accelerator.save_state.assert_called_once_with(str(tmp_path / "checkpoint-1"))


def test_load_state_called_on_resume(tmp_path):
    """accelerator.load_state is called when resume_path is not None."""
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "optimizer.bin").write_bytes(b"")
    (ckpt / "scheduler.pt").write_bytes(b"")
    (ckpt / "adapter_config.json").write_bytes(b"{}")

    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
        max_train_steps=100,
    )
    data_config = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_accelerator = mock.MagicMock(
        device=torch.device("cpu"),
        is_main_process=True,
    )
    mock_accelerator.prepare.return_value = (
        mock_transformer,
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )

    with (
        mock.patch(
            "videotuna.training.flux_lora.train.load_flux_training_models",
            return_value={
                "weight_dtype": torch.bfloat16,
                "transformer": mock_transformer,
                "vae": mock.MagicMock(),
                "text_encoder_one": mock.MagicMock(),
                "text_encoder_two": mock.MagicMock(),
                "tokenizer_one": mock.MagicMock(),
                "tokenizer_two": mock.MagicMock(),
            },
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.create_flux_accelerator",
            return_value=mock_accelerator,
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.load_lora_checkpoint",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FluxPipeline.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FlowMatchEulerDiscreteScheduler.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._build_embed_lookup",
            return_value={},
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FluxLoraImageDataset",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._build_dataloader",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._create_optimizer",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.get_scheduler",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_training_loop",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.checkpoint_step",
            return_value=50,
        ),
    ):
        train(config, data_config)

    mock_accelerator.load_state.assert_called_once_with(str(ckpt))


def test_no_manual_lr_scheduler_step_on_resume(tmp_path):
    """The resume path must not call lr_scheduler.step() in a loop."""
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "optimizer.bin").write_bytes(b"")
    (ckpt / "scheduler.pt").write_bytes(b"")
    (ckpt / "adapter_config.json").write_bytes(b"{}")

    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="latest",
        max_train_steps=100,
    )
    data_config = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_lr_scheduler = mock.MagicMock()
    mock_accelerator = mock.MagicMock(
        device=torch.device("cpu"),
        is_main_process=True,
    )
    mock_accelerator.prepare.return_value = (
        mock_transformer,
        mock.MagicMock(),
        mock.MagicMock(),
        mock_lr_scheduler,
    )

    with (
        mock.patch(
            "videotuna.training.flux_lora.train.load_flux_training_models",
            return_value={
                "weight_dtype": torch.bfloat16,
                "transformer": mock_transformer,
                "vae": mock.MagicMock(),
                "text_encoder_one": mock.MagicMock(),
                "text_encoder_two": mock.MagicMock(),
                "tokenizer_one": mock.MagicMock(),
                "tokenizer_two": mock.MagicMock(),
            },
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.create_flux_accelerator",
            return_value=mock_accelerator,
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.load_lora_checkpoint",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FluxPipeline.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FlowMatchEulerDiscreteScheduler.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._build_embed_lookup",
            return_value={},
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.FluxLoraImageDataset",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._build_dataloader",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._create_optimizer",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.get_scheduler",
            return_value=mock_lr_scheduler,
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_training_loop",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.checkpoint_step",
            return_value=50,
        ),
    ):
        train(config, data_config)

    mock_lr_scheduler.step.assert_not_called()


def test_load_lora_checkpoint_roundtrip():
    pytest.importorskip("peft")
    mock_transformer = mock.MagicMock()
    lora_state = {"layer.lora_A.weight": torch.zeros(4, 8)}
    with (
        mock.patch(
            "videotuna.training.flux_lora.checkpoint.FluxPipeline.lora_state_dict",
            return_value=lora_state,
        ),
        mock.patch(
            "videotuna.training.flux_lora.checkpoint.set_peft_model_state_dict",
        ) as set_state,
    ):
        load_lora_checkpoint(mock_transformer, "/tmp/checkpoint-100")
    set_state.assert_called_once_with(mock_transformer, lora_state)


def test_clip_grad_norm_called_with_configured_max_norm(tmp_path):
    """accelerator.clip_grad_norm_ is called when max_grad_norm is set."""
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        max_train_steps=1,
        max_grad_norm=1.0,
        checkpointing_steps=1,
        checkpoints_total_limit=None,
    )
    mock_accelerator = mock.MagicMock()
    mock_accelerator.is_main_process = True
    mock_accelerator.sync_gradients = True
    mock_accelerator.accumulate.return_value.__enter__ = mock.MagicMock()
    mock_accelerator.accumulate.return_value.__exit__ = mock.MagicMock(
        return_value=False
    )
    mock_accelerator.device = torch.device("cpu")
    mock_accelerator.clip_grad_norm_.return_value = 0.5
    mock_accelerator.unwrap_model.return_value = mock.MagicMock()

    mock_pipeline = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_transformer.parameters.return_value = [torch.randn(2, 2)]
    mock_dataloader = mock.MagicMock()
    mock_optimizer = mock.MagicMock()
    mock_lr_scheduler = mock.MagicMock()

    batch = {"pixel_values": torch.zeros(1, 3, 64, 64), "caption": ["test"]}
    mock_dataloader.__iter__ = mock.MagicMock(return_value=iter([batch]))

    with (
        mock.patch(
            "videotuna.training.flux_lora.train._compute_loss",
            return_value=torch.tensor(1.0, requires_grad=True),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_validation",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.save_lora_checkpoint",
            return_value=tmp_path / "checkpoint-1",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.prune_checkpoints",
        ),
    ):
        _run_training_loop(
            config=config,
            output_dir=tmp_path,
            pipeline=mock_pipeline,
            transformer=mock_transformer,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            accelerator=mock_accelerator,
            weight_dtype=torch.bfloat16,
            global_step=0,
            max_train_steps=1,
            log=mock.MagicMock(),
            metrics_backend="tensorboard",
        )

    # clip_grad_norm_ should be called with transformer.parameters() and max_norm=1.0
    mock_accelerator.clip_grad_norm_.assert_called_once()
    call_args = mock_accelerator.clip_grad_norm_.call_args
    assert call_args[1]["max_norm"] == 1.0

    # Verify grad_norm is logged
    log_call_args = mock_accelerator.log.call_args_list
    assert any("train/grad_norm" in call[0][0] for call in log_call_args)


def test_clip_grad_norm_disabled_when_none(tmp_path):
    """accelerator.clip_grad_norm_ is not called when max_grad_norm is None."""
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        max_train_steps=1,
        max_grad_norm=None,
        checkpointing_steps=1,
        checkpoints_total_limit=None,
    )
    mock_accelerator = mock.MagicMock()
    mock_accelerator.is_main_process = True
    mock_accelerator.sync_gradients = True
    mock_accelerator.accumulate.return_value.__enter__ = mock.MagicMock()
    mock_accelerator.accumulate.return_value.__exit__ = mock.MagicMock(
        return_value=False
    )
    mock_accelerator.device = torch.device("cpu")
    mock_accelerator.unwrap_model.return_value = mock.MagicMock()

    mock_pipeline = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_dataloader = mock.MagicMock()
    mock_optimizer = mock.MagicMock()
    mock_lr_scheduler = mock.MagicMock()

    batch = {"pixel_values": torch.zeros(1, 3, 64, 64), "caption": ["test"]}
    mock_dataloader.__iter__ = mock.MagicMock(return_value=iter([batch]))

    with (
        mock.patch(
            "videotuna.training.flux_lora.train._compute_loss",
            return_value=torch.tensor(1.0, requires_grad=True),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_validation",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.save_lora_checkpoint",
            return_value=tmp_path / "checkpoint-1",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.prune_checkpoints",
        ),
    ):
        _run_training_loop(
            config=config,
            output_dir=tmp_path,
            pipeline=mock_pipeline,
            transformer=mock_transformer,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            accelerator=mock_accelerator,
            weight_dtype=torch.bfloat16,
            global_step=0,
            max_train_steps=1,
            log=mock.MagicMock(),
            metrics_backend="tensorboard",
        )

    # clip_grad_norm_ should not be called
    mock_accelerator.clip_grad_norm_.assert_not_called()

    # Verify grad_norm is not logged
    log_call_args = mock_accelerator.log.call_args_list
    assert not any("train/grad_norm" in str(call[0][0]) for call in log_call_args)


def test_has_lora_weights_true(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_bytes(b"{}")
    assert has_lora_weights(ckpt) is True


def test_has_lora_weights_false_on_empty_dir(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    assert has_lora_weights(ckpt) is False


def test_has_lora_weights_false_on_nonexistent(tmp_path):
    assert has_lora_weights(tmp_path / "nope") is False


def test_find_latest_skips_incomplete_dirs(tmp_path):
    """Only checkpoint-50 has accelerate state AND LoRA weights."""
    for step in (10, 100):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        (d / "optimizer.bin").write_bytes(b"")
    d50 = tmp_path / "checkpoint-50"
    d50.mkdir()
    (d50 / "optimizer.bin").write_bytes(b"")
    (d50 / "adapter_config.json").write_bytes(b"{}")
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-50"


def test_find_latest_returns_none_all_incomplete(tmp_path):
    for step in (10, 50):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        (d / "optimizer.bin").write_bytes(b"")
    assert find_latest_checkpoint(tmp_path) is None


def test_resolve_resume_rejects_state_only_checkpoint(tmp_path):
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-50" / "optimizer.bin").write_bytes(b"")
    (tmp_path / "checkpoint-50" / "scheduler.pt").write_bytes(b"")
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="checkpoint-50",
    )
    with pytest.raises(ValueError, match="missing.*LoRA weight"):
        _resolve_resume_checkpoint(config, tmp_path)


def test_resolve_resume_rejects_lora_only_checkpoint(tmp_path):
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-50" / "adapter_config.json").write_bytes(b"{}")
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        resume_from_checkpoint="checkpoint-50",
    )
    with pytest.raises(ValueError, match="missing.*optimizer/scheduler"):
        _resolve_resume_checkpoint(config, tmp_path)


def test_save_state_called_on_non_main_process(tmp_path):
    """accelerator.save_state must run even when is_main_process=False."""
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir=str(tmp_path),
        instance_data_dir="data",
        max_train_steps=1,
        checkpointing_steps=1,
        checkpoints_total_limit=None,
    )
    mock_accelerator = mock.MagicMock()
    mock_accelerator.is_main_process = False
    mock_accelerator.sync_gradients = True
    mock_accelerator.accumulate.return_value.__enter__ = mock.MagicMock()
    mock_accelerator.accumulate.return_value.__exit__ = mock.MagicMock(
        return_value=False
    )
    mock_accelerator.device = torch.device("cpu")

    mock_pipeline = mock.MagicMock()
    mock_transformer = mock.MagicMock()
    mock_dataloader = mock.MagicMock()
    mock_optimizer = mock.MagicMock()
    mock_lr_scheduler = mock.MagicMock()

    batch = {"pixel_values": torch.zeros(1, 3, 64, 64), "caption": ["test"]}
    mock_dataloader.__iter__ = mock.MagicMock(return_value=iter([batch]))

    with (
        mock.patch(
            "videotuna.training.flux_lora.train._compute_loss",
            return_value=torch.tensor(1.0, requires_grad=True),
        ),
        mock.patch(
            "videotuna.training.flux_lora.train._run_validation",
        ),
        mock.patch(
            "videotuna.training.flux_lora.train.save_lora_checkpoint",
        ) as save_ckpt,
        mock.patch(
            "videotuna.training.flux_lora.train.prune_checkpoints",
        ),
    ):
        _run_training_loop(
            config=config,
            output_dir=tmp_path,
            pipeline=mock_pipeline,
            transformer=mock_transformer,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            accelerator=mock_accelerator,
            weight_dtype=torch.bfloat16,
            global_step=0,
            max_train_steps=1,
            log=mock.MagicMock(),
            metrics_backend="tensorboard",
        )

    mock_accelerator.save_state.assert_called_once()
    save_ckpt.assert_not_called()
