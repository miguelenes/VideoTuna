"""Explicit factory functions for domain flow construction.

These replace the magic ``instantiate_from_config`` dispatch for domain-only
configs.  Legacy configs (``configs/008_wanvideo/``) may still use
``instantiate_from_config`` as a fallback.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

from videotuna.base.generation_base import GenerationBase

FlowType = Literal["diffusers", "wan"]


class DiffusersFlowParams(BaseModel):
    """Validated parameters for building a DiffusersVideoFlow."""

    model_config = ConfigDict(extra="allow")

    model_family: str
    mode: str
    pipeline_only: bool = True
    model_variant: str | None = None
    pretrained_model_name_or_path: str | None = None
    lora_rank: int = 128
    lora_weight_name: str = "pytorch_lora_weights.safetensors"
    fuse_qkv: bool = False
    enable_attention_cache: bool = False


class WanFlowParams(BaseModel):
    """Validated parameters for building a WanVideoModelFlow.

    Sub-component configs (denoiser_config, first_stage_config, etc.) are
    passed through as dicts — their internal ``target:`` strings are resolved
    by ComponentLoader at construction time.
    """

    model_config = ConfigDict(extra="allow")

    task: str
    ckpt_path: str
    offload_model: bool | None = None
    ulysses_size: int = 1
    ring_size: int = 1
    t5_fsdp: bool = False
    t5_cpu: bool = False
    dit_fsdp: bool = False
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: str | None = None
    prompt_extend_target_lang: str = "zh"
    seed: int = -1
    gradient_checkpointing: bool = True

    # Sub-component configs (validated by their own loaders)
    denoiser_config: dict[str, Any] | None = None
    first_stage_config: dict[str, Any] | None = None
    cond_stage_config: dict[str, Any] | None = None
    cond_stage_2_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None
    lora_config: dict[str, Any] | None = None


def build_diffusers_flow(cfg: DictConfig) -> GenerationBase:
    """Construct a DiffusersVideoFlow from a resolved config.

    ``cfg`` is the ``flow`` section of a domain YAML (with ``flow_type: diffusers``).
    The ``params`` sub-key holds the constructor kwargs validated by
    :class:`DiffusersFlowParams`.
    """
    from videotuna.flow.diffusers_video import DiffusersVideoFlow

    params_raw = OmegaConf.to_container(
        cfg.get("params", OmegaConf.create()), resolve=True
    )
    validated = DiffusersFlowParams.model_validate(params_raw)
    kwargs = validated.model_dump(exclude_none=False)
    logger.info(
        "build_diffusers_flow: family={} mode={}",
        kwargs["model_family"],
        kwargs["mode"],
    )
    return DiffusersVideoFlow(**kwargs)


def build_wan_flow(cfg: DictConfig) -> GenerationBase:
    """Construct a WanVideoModelFlow from a resolved config.

    ``cfg`` is the ``flow`` section of a domain YAML (with ``flow_type: wan``).
    The ``params`` sub-key holds the constructor kwargs validated by
    :class:`WanFlowParams`.
    """
    from videotuna.flow.wanvideo import WanVideoModelFlow

    params_raw = OmegaConf.to_container(
        cfg.get("params", OmegaConf.create()), resolve=True
    )
    validated = WanFlowParams.model_validate(params_raw)
    kwargs = validated.model_dump(exclude_none=False)
    logger.info("build_wan_flow: task={}", kwargs["task"])
    return WanVideoModelFlow(**kwargs)


_FLOW_FACTORIES: dict[str, Any] = {
    "diffusers": build_diffusers_flow,
    "wan": build_wan_flow,
}


def build_flow(flow_config: DictConfig) -> GenerationBase:
    """Top-level dispatcher: build a flow from config.

    Checks for a ``flow_type`` key (domain configs) and dispatches to the
    appropriate factory.  If no ``flow_type`` is present but a ``target`` key
    exists, falls back to ``instantiate_from_config`` for legacy compatibility.

    Raises ``ValueError`` if neither ``flow_type`` nor ``target`` is present.
    """
    flow_type = flow_config.get("flow_type", None)

    if flow_type is not None:
        factory = _FLOW_FACTORIES.get(flow_type)
        if factory is None:
            raise ValueError(
                f"Unknown flow_type={flow_type!r}. "
                f"Valid types: {list(_FLOW_FACTORIES.keys())}"
            )
        return cast(GenerationBase, factory(flow_config))

    # Legacy fallback — config has ``target:`` string
    if "target" in flow_config:
        from videotuna.utils.common_utils import instantiate_from_config

        logger.debug(
            "build_flow: no flow_type found, falling back to instantiate_from_config "
            "(target={})",
            flow_config["target"],
        )
        return cast(GenerationBase, instantiate_from_config(flow_config, resolve=True))

    raise ValueError(
        "Flow config must have either 'flow_type' (domain) or 'target' (legacy)."
    )
