"""Verify flex_attention actually runs in eager and inductor modes on NPU."""

from __future__ import annotations

import os

import torch
from torch.nn.attention.flex_attention import flex_attention


def _noop_score_mod(score, b, h, m, n):
    return score


# ── test 1: eager mode ──────────────────────────────────────────────────


def test_eager_flex_attention():
    import torch_npu  # noqa: F401
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    output = flex_attention(q, k, v, score_mod=_noop_score_mod)
    assert output.shape == (B, H, S, D), f"eager: unexpected shape {output.shape}"
    print("  eager flex_attention OK")


# ── test 2: inductor mode ───────────────────────────────────────────────


def test_inductor_flex_attention():
    import torch_npu  # noqa: F401

    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    @torch.compile
    def compiled_fn(query, key, value):
        return flex_attention(query, key, value)

    output = compiled_fn(q, k, v)
    assert output.shape == (B, H, S, D), f"inductor: unexpected shape {output.shape}"
    print("  inductor flex_attention OK")


if __name__ == "__main__":
    test_eager_flex_attention()
    test_inductor_flex_attention()
    print("PASS: NPU FlexAttention eager + inductor execution.")
