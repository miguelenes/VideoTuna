"""
Unified attention backend selection for VideoTuna model families.

Environment variables:
    VIDEOTUNA_ATTN_BACKEND: auto | flash | sdpa | eager  (default: auto)
    VIDEOTUNA_TORCH_COMPILE: 0 | 1  (default: 0)
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

AttnBackend = Literal["flash", "sdpa", "eager"]
AttnLayout = Literal["bsnd", "bhsd"]

_ATTN_BACKEND_ENV = "VIDEOTUNA_ATTN_BACKEND"
_TORCH_COMPILE_ENV = "VIDEOTUNA_TORCH_COMPILE"

_FLASH_ATTN_FUNC = None
_FLASH_ATTN_VARLEN_FUNC = None
_FLASH_ATTN_3_VARLEN_FUNC = None
_FLASH_ATTN_AVAILABLE = False

try:
    from flash_attn import flash_attn_func as _FLASH_ATTN_FUNC
    from flash_attn import flash_attn_varlen_func as _FLASH_ATTN_VARLEN_FUNC

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

try:
    from flash_attn_interface import flash_attn_varlen_func as _FLASH_ATTN_3_VARLEN_FUNC
except ImportError:
    pass


def is_flash_attn_available() -> bool:
    return _FLASH_ATTN_AVAILABLE


def _resolve_auto_backend() -> AttnBackend:
    if _FLASH_ATTN_AVAILABLE and torch.cuda.is_available():
        return "flash"
    if torch.cuda.is_available():
        return "sdpa"
    return "eager"


def get_attn_backend() -> AttnBackend:
    """Resolve the active attention backend from env or auto-detection."""
    requested = os.environ.get(_ATTN_BACKEND_ENV, "auto").strip().lower()
    if requested == "auto":
        return _resolve_auto_backend()
    if requested in ("flash", "sdpa", "eager"):
        if requested == "flash" and not _FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "VIDEOTUNA_ATTN_BACKEND=flash requires flash-attn. "
                "Install with: poetry run install-flash-attn"
            )
        if requested == "sdpa" and not torch.cuda.is_available():
            return "eager"
        return requested  # type: ignore[return-value]
    raise ValueError(
        f"Invalid {_ATTN_BACKEND_ENV}={requested!r}. "
        "Expected auto, flash, sdpa, or eager."
    )


def _to_bhsd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: AttnLayout
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == "bhsd":
        return q, k, v
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _from_bhsd(x: torch.Tensor, layout: AttnLayout) -> torch.Tensor:
    if layout == "bhsd":
        return x
    return x.transpose(1, 2)


@contextmanager
def _sdpa_context():
    """Prefer flash/mem-efficient SDPA kernels on CUDA when available."""
    if not torch.cuda.is_available():
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            yield
    except (ImportError, AttributeError):
        yield


def attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: Optional[float] = None,
    layout: AttnLayout = "bsnd",
) -> torch.Tensor:
    q, k, v = _to_bhsd(q, k, v, layout)
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    b, _, s, _ = q.shape
    s1 = k.size(2)
    attn_bias = torch.zeros(b, q.size(1), s, s1, dtype=q.dtype, device=q.device)
    if causal:
        assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
        temp_mask = torch.ones(
            b, q.size(1), s, s, dtype=torch.bool, device=q.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_bias = attn_bias + attn_mask

    dtype = q.dtype
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn + attn_bias
    attn = attn.softmax(dim=-1).to(dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    out = attn @ v
    return _from_bhsd(out, layout)


def attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: Optional[float] = None,
    layout: AttnLayout = "bsnd",
    backend: Optional[AttnBackend] = None,
) -> torch.Tensor:
    """Dense attention with unified backend selection."""
    backend = backend or get_attn_backend()

    if backend == "flash":
        if layout == "bhsd":
            q_f, k_f, v_f = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q_f, k_f, v_f = q, k, v
        assert _FLASH_ATTN_FUNC is not None
        return _FLASH_ATTN_FUNC(
            q_f,
            k_f,
            v_f,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=causal,
        )

    if backend == "sdpa":
        q_s, k_s, v_s = _to_bhsd(q, k, v, layout)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_s.dtype)
        with _sdpa_context():
            out = F.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=scale,
            )
        return _from_bhsd(out, layout)

    return attention_eager(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        causal=causal,
        scale=scale,
        layout=layout,
    )


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    batch_size: Optional[int] = None,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    prefer_flash3: bool = True,
    backend: Optional[AttnBackend] = None,
) -> torch.Tensor:
    """Variable-length packed attention (flash varlen or dense fallback)."""
    backend = backend or get_attn_backend()

    if backend == "flash":
        if prefer_flash3 and _FLASH_ATTN_3_VARLEN_FUNC is not None:
            out = _FLASH_ATTN_3_VARLEN_FUNC(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
            if isinstance(out, tuple):
                out = out[0]
        else:
            assert _FLASH_ATTN_VARLEN_FUNC is not None
            out = _FLASH_ATTN_VARLEN_FUNC(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        if batch_size is not None:
            return out.view(batch_size, max_seqlen_q, out.shape[-2], out.shape[-1])
        return out

    if batch_size is None:
        raise ValueError("batch_size is required for non-flash varlen fallback")

    # Reshape packed varlen tensors back to padded batch for sdpa/eager.
    total_q = q.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[2]
    q_pad = q.view(batch_size, max_seqlen_q, n_heads, head_dim)
    k_pad = k.view(batch_size, max_seqlen_kv, n_heads, head_dim)
    v_pad = v.view(batch_size, max_seqlen_kv, n_heads, head_dim)
    return attention_dense(
        q_pad,
        k_pad,
        v_pad,
        dropout_p=dropout_p,
        causal=causal,
        scale=softmax_scale,
        layout="bsnd",
        backend=backend,
    )


_DIFFUSERS_BACKEND_MAP = {
    "flash": "flash",
    "sdpa": "native",
    "eager": "_native_math",
}


def apply_diffusers_attention_backend(model) -> None:
    """Map VIDEOTUNA_ATTN_BACKEND to diffusers set_attention_backend."""
    backend = get_attn_backend()
    diffusers_backend = _DIFFUSERS_BACKEND_MAP[backend]

    if hasattr(model, "set_attention_backend"):
        try:
            model.set_attention_backend(diffusers_backend)
            return
        except ValueError:
            if backend == "flash":
                model.set_attention_backend("native")
                return
            raise

    os.environ["DIFFUSERS_ATTN_BACKEND"] = diffusers_backend


def maybe_compile_denoiser(module: nn.Module) -> nn.Module:
    """Optionally compile a denoiser module when VIDEOTUNA_TORCH_COMPILE=1."""
    if os.environ.get(_TORCH_COMPILE_ENV, "0") != "1":
        return module
    if not torch.cuda.is_available():
        return module
    return torch.compile(module, mode="reduce-overhead", fullgraph=True)
