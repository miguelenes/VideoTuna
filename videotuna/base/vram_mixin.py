from typing import List

from loguru import logger

from videotuna.utils.device_utils import (
    empty_accelerator_cache,
    resolve_inference_device,
)


class VramMixin:
    def enable_vram_management(self):
        logger.info("enable_vram_management: default moving to cuda")
        self.cuda()

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names: List[str] = [], device=None):
        if device is None:
            device = str(resolve_inference_device())
        skip_components = ["scheduler"]
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            logger.info("cpu offload is closed, skipping")
            return
        # offload the unneeded models to cpu
        for model_name in self.components:
            if model_name in skip_components:
                logger.info(f"{model_name} no need cpu offload, skipping")
                continue

            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        logger.info(f"{model_name} cpu offloading using offload method")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        logger.info(f"{model_name} cpu offloading using to cpu method")
                        model.cpu()

        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    logger.info(f"{model_name} onloading using onload method")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    logger.info(f"{model_name} onloading using to device method")
                    model.to(device)
        # fresh the accelerator cache
        empty_accelerator_cache()
