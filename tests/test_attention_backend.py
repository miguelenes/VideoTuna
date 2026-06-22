import os

import pytest
import torch

from videotuna.utils.attention import (
    attention_dense,
    attention_eager,
    get_attn_backend,
    is_flash_attn_available,
)


@pytest.mark.parametrize("layout", ["bsnd", "bhsd"])
def test_eager_matches_sdpa_on_cpu(layout):
    torch.manual_seed(0)
    b, s, h, d = 2, 8, 4, 16
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    v = torch.randn(b, s, h, d)

    os.environ["VIDEOTUNA_ATTN_BACKEND"] = "eager"
    out_eager = attention_dense(q, k, v, layout=layout)

    os.environ["VIDEOTUNA_ATTN_BACKEND"] = "sdpa"
    out_sdpa = attention_dense(q, k, v, layout=layout)

    assert out_eager.shape == out_sdpa.shape
    torch.testing.assert_close(out_eager, out_sdpa, rtol=1e-2, atol=1e-2)


def test_attention_eager_scale():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = attention_eager(q, k, v, layout="bhsd", scale=0.125)
    assert out.shape == q.shape


def test_get_attn_backend_auto_cpu(monkeypatch):
    monkeypatch.delenv("VIDEOTUNA_ATTN_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_attn_backend() == "eager"


def test_get_attn_backend_explicit_eager(monkeypatch):
    monkeypatch.setenv("VIDEOTUNA_ATTN_BACKEND", "eager")
    assert get_attn_backend() == "eager"


def test_get_attn_backend_flash_requires_package(monkeypatch):
    monkeypatch.setenv("VIDEOTUNA_ATTN_BACKEND", "flash")
    if is_flash_attn_available():
        assert get_attn_backend() == "flash"
    else:
        with pytest.raises(RuntimeError, match="flash-attn"):
            get_attn_backend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_attn_backend_auto_cuda():
    os.environ.pop("VIDEOTUNA_ATTN_BACKEND", None)
    backend = get_attn_backend()
    assert backend in ("flash", "sdpa")
