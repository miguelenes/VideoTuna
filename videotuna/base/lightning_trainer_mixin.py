import os
import signal
from typing import Any, Dict, cast

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from videotuna.base.component_loader import Component
from videotuna.utils.common_utils import get_dist_info, instantiate_from_config
from videotuna.utils.train_utils import (
    get_trainer_callbacks,
    get_trainer_logger,
    get_trainer_strategy,
)


class LightningTrainerMixin:
    """Mixin providing Lightning trainer setup, optimizer, and LR-scheduler config."""

    def configure_lr_config(self, lr_config: Dict[str, Any], bs: int, num_rank: int):
        base_lr = lr_config["base_learning_rate"]
        if lr_config.get("scale_lr", True):
            lr_config["learning_rate"] = num_rank * bs * base_lr
        else:
            lr_config["learning_rate"] = base_lr
        self.lr_config = lr_config

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the generative model.

        :return: A list containing the optimizer and optionally a list
            containing the learning rate scheduler.
        """
        assert self.lr_config is not None
        lr_config = self.lr_config
        lr = lr_config["learning_rate"]
        params = [p for p in self.parameters() if p.requires_grad]
        logger.info(f"@Training [{len(params)}] Full Paramters.")

        ## optimizer
        if self.trainer.strategy.__class__.__name__ == "DeepSpeedStrategy":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(params, lr=lr)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if lr_config.get("lr_scheduler_config", None):
            logger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_lr_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer

    def configure_lr_schedulers(self, optimizer):
        """
        Configures the learning rate scheduler based on the provided configuration.

        :param optimizer: The optimizer for which the scheduler is being configured.
        :return: A dictionary containing the scheduler, interval, and frequency.
        """
        assert self.lr_config is not None
        lr_scheduler_config = self.lr_config["lr_scheduler_config"]
        assert "target" in lr_scheduler_config
        scheduler_name = lr_scheduler_config["target"].split(".")[-1]
        interval = lr_scheduler_config["interval"]
        frequency = lr_scheduler_config["frequency"]
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(lr_scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                "interval": interval,
                "frequency": frequency,
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(lr_scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer, T_max=decay_steps, last_epoch=last_step
                ),
                "interval": interval,
                "frequency": frequency,
            }
        else:
            raise NotImplementedError
        return lr_scheduler

    def init_trainer(self, train_config: DictConfig):
        # 1. basic info setup
        local_rank, global_rank, num_rank = get_dist_info()

        debug = train_config["debug"]
        workdir = train_config["workdir"]
        ckptdir = train_config["ckptdir"]
        lightning_config: DictConfig = train_config.get("lightning")
        trainer_config: DictConfig = lightning_config.get("trainer")
        self.first_stage_key = train_config.first_stage_key
        self.cond_stage_key = train_config.cond_stage_key
        self.logdir = workdir

        # 2. lr
        lr_config: DictConfig = train_config.get("lr_config")
        bs = train_config["data"]["params"]["batch_size"]
        self.lr_config = cast(
            dict[str, Any], OmegaConf.to_container(lr_config, resolve=True)
        )
        self.configure_lr_config(self.lr_config, bs=bs, num_rank=num_rank)

        # 3. dataset
        logger.info("***** Configuring Data *****")
        data = instantiate_from_config(train_config["data"])
        self.data = data
        assert data is not None
        data.setup()
        for k in data.datasets:
            logger.info(
                f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
            )

        ## 4. lightning trainer config
        logger.info(f"trainer_config: {trainer_config}")
        num_nodes = trainer_config["num_nodes"]
        ngpu_per_node = trainer_config["devices"]
        logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")
        logger.info("***** Configuring Trainer *****")

        # 4.1 trainer gpu
        if "accelerator" not in trainer_config:
            trainer_config["accelerator"] = "gpu"

        ## 4.2 logger
        trainer_kwargs: dict[str, Any] = {}
        trainer_kwargs["num_sanity_val_steps"] = 0
        logger_cfg = get_trainer_logger(lightning_config, workdir, debug)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        logger_obj = trainer_kwargs["logger"]
        if hasattr(logger_obj, "save_dir"):
            logger.info(f"logger save_dir: {logger_obj.save_dir}")

        ## 4.3 callback
        callbacks_cfg = cast(
            dict[str, Any], get_trainer_callbacks(lightning_config, workdir, ckptdir)
        )
        callbacks_cfg["image_logger"]["params"]["save_dir"] = workdir
        if "training_metrics" in callbacks_cfg:
            callbacks_cfg["training_metrics"]["params"]["save_dir"] = workdir
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]

        ## 4.4 strategy
        strategy_cfg = get_trainer_strategy(lightning_config)
        trainer_kwargs["strategy"] = (
            strategy_cfg
            if isinstance(strategy_cfg, str)
            else instantiate_from_config(
                cast(dict[str, Any], OmegaConf.to_container(strategy_cfg))
            )
        )
        trainer_kwargs["sync_batchnorm"] = False

        ## 4.5 create Trainer
        logger.info(f"trainer_kwargs: {trainer_kwargs}")
        enable_profiler = lightning_config.get("enable_profiler", False)
        profiler = None
        if enable_profiler:
            from pytorch_lightning.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(emit_nvtx=True)
        trainer_config_dict = cast(
            dict[str, Any], OmegaConf.to_container(trainer_config, resolve=True)
        )
        trainer = Trainer(**trainer_config_dict, **trainer_kwargs, profiler=profiler)
        self.trainer = trainer

        ## 5. allow user
        def melk(*args, **kwargs):
            ## run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        ## since loaded weight will ovrride params, make sure it is been handled
        if trainer.strategy.__class__.__name__ == "DeepSpeedStrategy":
            logger.info(
                "Make parameter contiguous in case deepseed does not allow "
                "non contigouous data"
            )
            for param in self.parameters():
                param.data = param.data.contiguous()
        self.set_trainable_components([Component.DENOISER.value])
