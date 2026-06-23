from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from videotuna.base.checkpoint_mixin import CheckpointMixin
from videotuna.base.component_loader import Component, ComponentLoader, LoadingMethod
from videotuna.base.inference_base import InferenceBase
from videotuna.base.lightning_trainer_mixin import LightningTrainerMixin
from videotuna.base.lora_training_mixin import LoraTrainingMixin
from videotuna.base.train_base import TrainBase
from videotuna.base.vram_mixin import VramMixin

__all__ = ["GenerationBase", "Component", "LoadingMethod", "ComponentLoader"]


class GenerationBase(
    LightningTrainerMixin,
    VramMixin,
    TrainBase,
    InferenceBase,
    ComponentLoader,
    LoraTrainingMixin,
    CheckpointMixin,
):
    denoiser: nn.Module | None = None
    first_stage_model: nn.Module | None = None
    cond_stage_model: nn.Module | None = None
    cond_stage_2_model: nn.Module | None = None
    scheduler: Any | None = None
    lr_config: dict[str, Any] | None = None
    data: Any | None = None
    pipeline: Any | None = None

    """
    The GenerationFlow class is a generative model class that inherits from both
    TrainBase and InferenceBase.
    It manages the instantiation of different stages of a generative process,
    including a denoiser and a scheduler.
    It also configures optimizers and learning rate schedulers for training.

    The main components of the model are:
        - `first_stage`: a VAE model that encodes the input video into a latent
          space and decodes it back to the original video.
        - `cond_stage`: a conditional model that takes the latent space and the
          conditioning text as input and generates the output video.
        - `denoiser`: a denoiser model that takes the noisy output of the
          `cond_stage` and tries to remove the noise.
        - `scheduler`: a scheduler that controls denosing and sampling.
    """

    def __init__(
        self,
        first_stage_config: Optional[Dict[str, Any]] = None,
        cond_stage_config: Optional[Dict[str, Any]] = None,
        denoiser_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        trainable_components: Union[str, List[str]] = [],
        pipeline_only: bool = False,
    ):
        """
        Initializes the GenerationFlow class with configurations for different
        stages and components.

        :param first_stage_config: Dictionary containing configuration for the
            first stage model.
        :param cond_stage_config: Dictionary containing configuration for the
            conditional stage model.
        :param cond_stage_2_config: Dictionary containing configuration for the
            conditional stage model 2, can be none.
        :param denoiser_config: Dictionary containing configuration for the
            denoiser model.
        :param scheduler_config: Dictionary containing configuration for the
            diffusion scheduler.
        :param trainable_components: The components of the model that should be
            trainable.
        :param pipeline_only: When True, skip stage instantiation (Diffusers
            pipeline flows).
        """
        super().__init__()

        # instantiate the modules
        self.components: list[str] = []
        self.pipeline_only = pipeline_only
        if pipeline_only:
            self.use_lora = False
            self.pipeline = None
            return
        # 1. denoiser
        self.instantiate_denoiser(denoiser_config)

        # 2. first stage
        self.instantiate_first_stage(first_stage_config)

        # 3. cond stage
        self.instantiate_cond_stage(cond_stage_config)

        # 4. cond stage 2
        self.instantiate_cond_stage_2(cond_stage_2_config)

        # 5. lora: will set is_lora and lora_params
        self.instantiate_lora(lora_config)

        # 6. scheduler
        self.instantiate_scheduler(scheduler_config)

        # config
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.cond_stage_2_config = cond_stage_2_config
        self.denoiser_config = denoiser_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config

        # set trainable components
        # be aware: loaded weight will overide requrie_grad attribute etc
        # make sure call it again after loading weight
        self.set_trainable_components(trainable_components)
