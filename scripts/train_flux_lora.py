"""Train Flux LoRA adapters using the first-party Diffusers trainer."""

import argparse
import logging

from videotuna.settings import get_settings
from videotuna.training.flux_lora.train import run_training

logger = logging.getLogger("FluxLoraTrainer")
logger.setLevel(get_settings().log_level)


def main(args: argparse.Namespace) -> None:
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as exc:
        logger.warning("Could not set multiprocessing start method to 'fork': %s", exc)
    run_training(args.config_path, args.data_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Flux LoRA (Diffusers + PEFT)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Training config JSON"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        required=True,
        help="Path to multidatabackend JSON",
    )
    main(parser.parse_args())
