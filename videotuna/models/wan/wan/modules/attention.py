# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

import torch

from videotuna.utils.attention import attention_varlen, get_attn_backend

__all__ = [
    "flash_attention",
    "attention",
]

FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False

try:
    import flash_attn_interface  # noqa: F401

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    import flash_attn  # noqa: F401

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    pass


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead."
        )

    prefer_flash3 = (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE
    cu_seqlens_q = (
        torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )
    cu_seqlens_k = (
        torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(k.device, non_blocking=True)
    )

    x = attention_varlen(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=lq,
        max_seqlen_kv=lk,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
        batch_size=b,
        window_size=window_size,
        deterministic=deterministic,
        prefer_flash3=prefer_flash3,
    )
    if x.ndim == 3:
        x = x.unflatten(0, (b, lq))
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    backend = get_attn_backend()
    if backend != "flash" and (q_lens is not None or k_lens is not None):
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
        )

    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
