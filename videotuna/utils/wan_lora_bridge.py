"""Bridge Wan 2.1 native Lightning LoRA checkpoints onto Wan 2.2 Diffusers pipelines."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

# Native Wan 2.1 PEFT targets (training config in wan_t2v_lora.yaml).
WAN_NATIVE_LORA_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]

# Diffusers WanTransformer3DModel PEFT targets (attn1 self-attn, attn2 cross-attn, FFN).
WAN_DIFFUSERS_LORA_TARGETS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]

DEFAULT_LORA_RANK = 16
HIGH_NOISE_ADAPTER = "domain_high"
LOW_NOISE_ADAPTER = "domain_low"
MIN_REMAP_COVERAGE = 0.9


@dataclass
class WanBridgeConfig:
    """Configuration for the Wan 2.1 → 2.2 LoRA bridge.

    Controls coverage thresholds, fallback remap behavior, and strictness.
    Use ``from_env()`` to pick up ``VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE``.
    """

    min_coverage: float = MIN_REMAP_COVERAGE
    min_coverage_export: float | None = None
    allow_fallback_remap: bool = False

    @property
    def effective_export_coverage(self) -> float:
        return (
            self.min_coverage_export
            if self.min_coverage_export is not None
            else self.min_coverage
        )

    @classmethod
    def from_env(cls) -> WanBridgeConfig:
        """Build from ``VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE`` env var if set."""
        raw = os.environ.get("VIDEOTUNA_WAN_BRIDGE_MIN_COVERAGE")
        if raw is not None:
            return cls(min_coverage=float(raw))
        return cls()


_SELF_ATTN_RE = re.compile(
    r"^blocks\.(\d+)\.self_attn\.(q|k|v|o)\.(lora_[AB]\.weight)$"
)
_CROSS_ATTN_RE = re.compile(
    r"^blocks\.(\d+)\.cross_attn\.(q|k|v|o)\.(lora_[AB]\.weight)$"
)
_FFN0_RE = re.compile(r"^blocks\.(\d+)\.ffn\.0\.(lora_[AB]\.weight)$")
_FFN2_RE = re.compile(r"^blocks\.(\d+)\.ffn\.2\.(lora_[AB]\.weight)$")
# Legacy / test shorthand: blocks.N.attn.q (no self_/cross_ prefix).
_LEGACY_ATTN_RE = re.compile(r"^blocks\.(\d+)\.attn\.(q|k|v|o)\.(lora_[AB]\.weight)$")

_PATTERN_LABELS: Dict[re.Pattern, str] = {
    _SELF_ATTN_RE: "self_attn",
    _CROSS_ATTN_RE: "cross_attn",
    _FFN0_RE: "ffn.0",
    _FFN2_RE: "ffn.2",
    _LEGACY_ATTN_RE: "legacy_attn",
}


@dataclass
class WanLoraLoadReport:
    """Structured result from loading native LoRA onto a Diffusers transformer."""

    expert: str
    rank: int
    source_keys: int
    remapped_keys: int
    loaded_lora_params: int
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    unmapped_keys: List[str] = field(default_factory=list)
    renamed_keys: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def remap_ratio(self) -> float:
        if self.source_keys == 0:
            return 0.0
        return self.remapped_keys / self.source_keys

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expert": self.expert,
            "rank": self.rank,
            "source_keys": self.source_keys,
            "remapped_keys": self.remapped_keys,
            "loaded_lora_params": self.loaded_lora_params,
            "remap_ratio": round(self.remap_ratio, 4),
            "missing_keys": len(self.missing_keys),
            "unexpected_keys": len(self.unexpected_keys),
            "unmapped_keys": len(self.unmapped_keys),
            "renamed_keys": len(self.renamed_keys),
        }


@dataclass
class KeyDiffEntry:
    """Per-key remap status entry from the bridge."""

    native_key: str
    diffusers_key: str | None = None
    status: str = "unmapped"  # "remapped" | "unmapped" | "fallback"
    pattern: str | None = None
    expert: str = "both"  # high-noise / low-noise — both until expert split


@dataclass
class ParityReport:
    """Comparison between runtime bridge remap and offline export remap."""

    keys_match: bool
    runtime_key_count: int
    export_key_count: int
    only_in_export: List[str] = field(default_factory=list)


def is_native_wan_lora_ckpt(path: str | Path) -> bool:
    """Return True when path looks like a Wan native Lightning LoRA .ckpt."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix not in (".ckpt", ".pt", ".pth"):
        return False
    try:
        state = load_native_wan_lora_state_dict(p)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(state) and any("lora" in k.lower() for k in state)


def load_native_wan_lora_state_dict(ckpt_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load and normalize LoRA tensors from a Wan training checkpoint."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint type in {ckpt_path}")

    lora_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if "lora" not in key.lower():
            continue
        normalized = key
        for prefix in ("denoiser.", "model.", "module.", "base_model.model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        lora_state[normalized] = tensor
    if not lora_state:
        raise ValueError(f"No LoRA keys found in {ckpt_path}")
    return lora_state


def _infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight") or ".lora_A." in key:
            return int(tensor.shape[0])
        if key.endswith(".lora_B.weight") or ".lora_B." in key:
            return int(tensor.shape[1])
    return DEFAULT_LORA_RANK


def _remap_single_native_key(key: str) -> str:
    """Map one native Wan 2.1 LoRA key to Diffusers WanTransformer3DModel naming."""
    m = _SELF_ATTN_RE.match(key)
    if m:
        idx, proj, suffix = m.groups()
        diff_proj = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}[proj]
        return f"blocks.{idx}.attn1.{diff_proj}.{suffix}"

    m = _CROSS_ATTN_RE.match(key)
    if m:
        idx, proj, suffix = m.groups()
        diff_proj = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}[proj]
        return f"blocks.{idx}.attn2.{diff_proj}.{suffix}"

    m = _LEGACY_ATTN_RE.match(key)
    if m:
        idx, proj, suffix = m.groups()
        diff_proj = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}[proj]
        return f"blocks.{idx}.attn1.{diff_proj}.{suffix}"

    m = _FFN0_RE.match(key)
    if m:
        return f"blocks.{m.group(1)}.ffn.net.0.proj.{m.group(2)}"

    m = _FFN2_RE.match(key)
    if m:
        return f"blocks.{m.group(1)}.ffn.net.2.{m.group(2)}"

    # Back-compat: blocks.* -> transformer_blocks.* (older bridge attempt).
    if key.startswith("blocks."):
        return "transformer_blocks." + key[len("blocks.") :]
    return key


def _remap_native_to_diffusers_keys(
    native_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap native Wan 2.1 block names to Diffusers WanTransformer3DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in native_state.items():
        remapped[_remap_single_native_key(key)] = tensor
    return remapped


def _matches_known_remap_pattern(key: str) -> bool:
    """Return True when the key matches one of the supported remap patterns."""
    return any(
        pattern.match(key)
        for pattern in (
            _SELF_ATTN_RE,
            _CROSS_ATTN_RE,
            _FFN0_RE,
            _FFN2_RE,
            _LEGACY_ATTN_RE,
        )
    )


def _match_label(key: str) -> str | None:
    """Return the human-readable pattern label for a key, or None."""
    for pattern, label in _PATTERN_LABELS.items():
        if pattern.match(key):
            return label
    return None


def _remap_state_with_meta(
    native_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[Tuple[str, str]]]:
    """Remap keys and return (remapped_state, unmapped_keys, renamed_pairs)."""
    remapped: Dict[str, torch.Tensor] = {}
    unmapped: List[str] = []
    renamed: List[Tuple[str, str]] = []
    for key, tensor in native_state.items():
        if not _matches_known_remap_pattern(key):
            unmapped.append(key)
            continue
        new_key = _remap_single_native_key(key)
        renamed.append((key, new_key))
        remapped[new_key] = tensor
    return remapped, unmapped, renamed


def compute_remap_coverage(
    native_state: Dict[str, torch.Tensor],
) -> Tuple[int, int, float]:
    """Return transformed key count, total keys, and coverage ratio."""
    if not native_state:
        return 0, 0, 0.0
    remapped, unmapped, _ = _remap_state_with_meta(native_state)
    transformed = len(native_state) - len(unmapped)
    total = len(native_state)
    return transformed, total, transformed / total


def validate_remap_coverage(
    native_state: Dict[str, torch.Tensor],
    *,
    min_coverage: float = MIN_REMAP_COVERAGE,
    context: str = "",
    config: WanBridgeConfig | None = None,
) -> Tuple[int, int, float, List[str]]:
    """Validate remap coverage and return (remapped_count, total, ratio, unmapped).

    Raises RuntimeError when coverage is below ``min_coverage`` (or
    ``config.min_coverage`` if ``config`` is provided).
    """
    if config is not None:
        min_coverage = config.min_coverage
    if not native_state:
        raise ValueError("No LoRA tensors to validate; checkpoint may be empty")
    remapped, unmapped, _ = _remap_state_with_meta(native_state)
    remapped_count = len(native_state) - len(unmapped)
    total = len(native_state)
    coverage = remapped_count / total
    if coverage < min_coverage:
        prefix = f"{context}: " if context else ""
        sample = unmapped[:10]
        raise RuntimeError(
            f"{prefix}Wan LoRA bridge remap coverage {coverage:.1%} is below the "
            f"required {min_coverage:.0%} threshold. "
            f"{len(unmapped)} of {total} keys were not remapped "
            f"(first {len(sample)} unmapped): {sample}. "
            "Run tools/spike_wan_lora_bridge.py for a full key inventory."
        )
    return remapped_count, total, coverage, unmapped


def analyze_native_wan_lora_ckpt(ckpt_path: str | Path) -> Dict[str, Any]:
    """Inventory native checkpoint keys and remapped Diffusers targets."""
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    remapped, unmapped, renamed = _remap_state_with_meta(native_state)
    remapped_count = len(native_state) - len(unmapped)
    return {
        "path": str(ckpt_path),
        "rank": _infer_lora_rank(native_state),
        "native_key_count": len(native_state),
        "remapped_key_count": len(remapped),
        "remap_coverage": (
            round(remapped_count / len(native_state), 4) if native_state else 0.0
        ),
        "unmapped_keys": unmapped,
        "renamed_keys": renamed,
        "sample_native": sorted(native_state.keys())[:5],
        "sample_remapped": sorted(remapped.keys())[:5],
    }


def build_bridge_key_map(
    native_state: Dict[str, torch.Tensor],
    *,
    config: WanBridgeConfig | None = None,
) -> List[KeyDiffEntry]:
    """Build a per-key remap status table for the entire state dict.

    Every native key gets one entry showing its remap status, target
    Diffusers key, and the matched pattern label.
    """
    cfg = config or WanBridgeConfig()
    entries: List[KeyDiffEntry] = []
    for key in native_state:
        if _matches_known_remap_pattern(key):
            new_key = _remap_single_native_key(key)
            entries.append(
                KeyDiffEntry(
                    native_key=key,
                    diffusers_key=new_key,
                    status="remapped",
                    pattern=_match_label(key),
                )
            )
        elif cfg.allow_fallback_remap and key.startswith("blocks."):
            entries.append(
                KeyDiffEntry(
                    native_key=key,
                    diffusers_key="transformer_blocks." + key[len("blocks.") :],
                    status="fallback",
                )
            )
        else:
            entries.append(KeyDiffEntry(native_key=key, status="unmapped"))
    return entries


def verify_runtime_export_parity(
    ckpt_path: str | Path,
    *,
    config: WanBridgeConfig | None = None,
) -> ParityReport:
    """Compare runtime bridge remap with offline export remap.

    The runtime path uses ``_remap_state_with_meta`` which drops keys
    not matching known patterns.  The export path uses
    ``_remap_native_to_diffusers_keys`` which applies the fallback
    ``blocks.* → transformer_blocks.*`` to all keys.  This function
    flags any divergence between the two.
    """
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    runtime_remapped, _, _ = _remap_state_with_meta(native_state)
    export_remapped = _remap_native_to_diffusers_keys(native_state)

    runtime_keys = set(runtime_remapped)
    export_keys = set(export_remapped)

    only_in_export = sorted(export_keys - runtime_keys)
    keys_match = runtime_keys == export_keys

    return ParityReport(
        keys_match=keys_match,
        runtime_key_count=len(runtime_keys),
        export_key_count=len(export_keys),
        only_in_export=only_in_export,
    )


def _peft_prefix_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k if k.startswith("base_model.model.") else f"base_model.model.{k}": v
        for k, v in state.items()
    }


def _count_lora_params(module: Any) -> int:
    return sum(1 for name, _ in module.named_parameters() if "lora" in name.lower())


def _apply_lora_to_transformer(
    transformer: Any,
    remapped_state: Dict[str, torch.Tensor],
    *,
    rank: int,
    adapter_name: str,
    expert_label: str,
    source_keys: int,
    remapped_keys: int,
    unmapped_keys: List[str],
    renamed_keys: List[Tuple[str, str]],
) -> Tuple[Any, WanLoraLoadReport]:
    """Inject PEFT LoRA adapters and load remapped weights onto one transformer."""
    if not hasattr(transformer, "peft_config") or not transformer.peft_config:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights=True,
            target_modules=WAN_DIFFUSERS_LORA_TARGETS,
        )
        transformer = get_peft_model(
            transformer, lora_config, adapter_name=adapter_name
        )

    peft_state = _peft_prefix_keys(remapped_state)
    result = set_peft_model_state_dict(
        transformer,
        peft_state,
        adapter_name=adapter_name,
    )
    missing: List[str] = []
    unexpected: List[str] = []
    if hasattr(result, "missing_keys"):
        missing = [k for k in result.missing_keys if "lora" in k.lower()]
        unexpected = list(result.unexpected_keys)

    loaded = _count_lora_params(transformer)
    report = WanLoraLoadReport(
        expert=expert_label,
        rank=rank,
        source_keys=source_keys,
        remapped_keys=remapped_keys,
        loaded_lora_params=loaded,
        missing_keys=missing,
        unexpected_keys=unexpected,
        unmapped_keys=unmapped_keys,
        renamed_keys=renamed_keys,
    )
    if unexpected:
        logger.warning(
            "Wan LoRA bridge [{}]: {} unexpected keys (first 5): {}",
            expert_label,
            len(unexpected),
            unexpected[:5],
        )
    if missing:
        logger.warning(
            "Wan LoRA bridge [{}]: {} missing LoRA keys (first 5): {}",
            expert_label,
            len(missing),
            missing[:5],
        )
    logger.info("Wan LoRA bridge [{}]: {}", expert_label, report.as_dict())
    return transformer, report


def apply_native_wan_lora_to_pipeline(
    pipeline: Any,
    ckpt_path: str | Path,
    *,
    lora_scale: float = 1.0,
    lora_scale_2: Optional[float] = None,
    mode: str = "t2v",
    bridge_config: WanBridgeConfig | None = None,
) -> List[WanLoraLoadReport]:
    """
    Attach Wan 2.1 native LoRA weights to a Wan 2.2 Diffusers pipeline.

    Loads the same adapter onto ``transformer`` (high-noise) and ``transformer_2``
    (low-noise) when both are present, matching Diffusers community practice.

    Args:
        pipeline: A Wan 2.2 Diffusers pipeline (T2V or I2V).
        ckpt_path: Native Wan 2.1 Lightning LoRA checkpoint.
        lora_scale: Scale for the high-noise (transformer) adapter.
        lora_scale_2: Scale for the low-noise (transformer_2) adapter.
        mode: "t2v" or "i2v"; affects logging/report labels only.
        bridge_config: Optional bridge configuration (defaults to env vars).
    """
    config = bridge_config or WanBridgeConfig.from_env()
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    rank = _infer_lora_rank(native_state)
    remapped, unmapped, renamed = _remap_state_with_meta(native_state)
    remapped_keys, source_keys, _, _ = validate_remap_coverage(
        native_state,
        config=config,
        context=f"Wan {mode.upper()} LoRA bridge",
    )
    scale_2 = lora_scale if lora_scale_2 is None else lora_scale_2

    reports: List[WanLoraLoadReport] = []
    adapters: List[str] = []
    scales: List[float] = []

    expert_prefix = f"{mode}_" if mode != "t2v" else ""

    pipeline.transformer, report_high = _apply_lora_to_transformer(
        pipeline.transformer,
        remapped,
        rank=rank,
        adapter_name=HIGH_NOISE_ADAPTER,
        expert_label=f"{expert_prefix}transformer",
        source_keys=source_keys,
        remapped_keys=remapped_keys,
        unmapped_keys=unmapped,
        renamed_keys=renamed,
    )
    reports.append(report_high)
    adapters.append(HIGH_NOISE_ADAPTER)
    scales.append(lora_scale)

    transformer_2 = getattr(pipeline, "transformer_2", None)
    if transformer_2 is not None:
        pipeline.transformer_2, report_low = _apply_lora_to_transformer(
            transformer_2,
            remapped,
            rank=rank,
            adapter_name=LOW_NOISE_ADAPTER,
            expert_label=f"{expert_prefix}transformer_2",
            source_keys=source_keys,
            remapped_keys=remapped_keys,
            unmapped_keys=unmapped,
            renamed_keys=renamed,
        )
        reports.append(report_low)
        adapters.append(LOW_NOISE_ADAPTER)
        scales.append(scale_2)

    total_loaded = sum(r.loaded_lora_params for r in reports)
    if total_loaded == 0:
        raise RuntimeError(
            f"Wan LoRA bridge failed to load any parameters from {ckpt_path}. "
            "Run tools/spike_wan_lora_bridge.py for a key inventory."
        )

    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(adapters, adapter_weights=scales)
    elif hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=lora_scale)

    logger.info(
        "Wan %s LoRA bridge: rank=%d experts=%s total_lora_params=%d scales=%s",
        mode.upper(),
        rank,
        [r.expert for r in reports],
        total_loaded,
        scales,
    )
    return reports


def apply_native_wan_lora_to_i2v_pipeline(
    pipeline: Any,
    ckpt_path: str | Path,
    *,
    lora_scale: float = 1.0,
    lora_scale_2: Optional[float] = None,
    bridge_config: WanBridgeConfig | None = None,
) -> List[WanLoraLoadReport]:
    """
    Attach Wan 2.1 native I2V LoRA weights to a Wan 2.2 I2V Diffusers pipeline.

    Uses the same block-level key remap as T2V; both transformer experts receive
    identical adapter weights when ``transformer_2`` is present. The mode label
    is set to "i2v" for reporting and validation context.
    """
    return apply_native_wan_lora_to_pipeline(
        pipeline,
        ckpt_path,
        lora_scale=lora_scale,
        lora_scale_2=lora_scale_2,
        mode="i2v",
        bridge_config=bridge_config,
    )


def export_diffusers_lora_state_dicts(
    ckpt_path: str | Path,
    *,
    mode: str = "t2v",
    bridge_config: WanBridgeConfig | None = None,
    include_key_diff: bool = False,
) -> Dict[str, Any]:
    """
    Export remapped LoRA tensors for Diffusers ``load_lora_weights``.

    Validates remap coverage before export and raises RuntimeError if it is
    below the configured threshold (default ``MIN_REMAP_COVERAGE``).

    Returns a dict with ``high_noise`` and ``low_noise`` tensor dicts,
    plus ``_parity`` metadata.  When ``include_key_diff=True`` a
    ``_key_diff`` entry with full per-key status is also included.

    Both expert dicts hold the **same** remapped tensors; Wan 2.2
    loads low-noise via ``load_into_transformer_2``.
    """
    config = bridge_config or WanBridgeConfig.from_env()
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    validate_remap_coverage(
        native_state, config=config, context=f"Wan {mode.upper()} LoRA export"
    )
    remapped = _remap_native_to_diffusers_keys(native_state)
    exports: Dict[str, Any] = {
        "high_noise": remapped,
        "low_noise": dict(remapped),
    }

    parity = verify_runtime_export_parity(ckpt_path, config=config)
    exports["_parity"] = {
        "keys_match": parity.keys_match,
        "runtime_key_count": parity.runtime_key_count,
        "export_key_count": parity.export_key_count,
        "only_in_export": parity.only_in_export,
    }
    if include_key_diff:
        exports["_key_diff"] = [
            {
                "native_key": e.native_key,
                "diffusers_key": e.diffusers_key,
                "status": e.status,
                "pattern": e.pattern,
            }
            for e in build_bridge_key_map(native_state, config=config)
        ]

    return exports


def strip_peft_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove PEFT prefixes for safetensors export."""
    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        k = key
        for prefix in ("base_model.model.", "base_model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = tensor
    return cleaned
