"""Correctness tests for packed_sdpa_attention (numerical + reshape-layout).

Run: ``pytest tests/test_packed_sdpa.py`` (needs torch, ideally CUDA for the speed check).
The speed/quality numbers are in benchmarks/ and README; these tests only assert the
attention math is a faithful block-diagonal replacement for per-image SDPA and that the
output head layout is the one MoonViT's ``wo`` projection expects.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from locateanything_optimization import packed_sdpa_attention


def _ref_block(qb, kb, vb, nh, hd):
    """Per-image reference: run SDPA on a single block, lay out heads contiguous per position."""
    out = F.scaled_dot_product_attention(
        qb.transpose(0, 1).unsqueeze(0),
        kb.transpose(0, 1).unsqueeze(0),
        vb.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
    )
    return out.permute(0, 2, 1, 3).reshape(qb.shape[0], nh * hd)


def test_uniform_blocks_match_per_block_sdpa():
    torch.manual_seed(0)
    nh, hd, S, N = 2, 4, 3, 2
    L = S * N
    q = torch.randn(L, nh, hd)
    k = torch.randn(L, nh, hd)
    v = torch.randn(L, nh, hd)
    cu = torch.tensor([0, S, 2 * S], dtype=torch.int32)

    packed = packed_sdpa_attention(q, k, v, cu, cu)
    ref = torch.cat(
        [_ref_block(q[i * S : (i + 1) * S], k[i * S : (i + 1) * S], v[i * S : (i + 1) * S], nh, hd)
         for i in range(N)],
        dim=0,
    )
    assert packed.shape == ref.shape
    assert torch.allclose(packed, ref, atol=1e-6)


def test_ragged_blocks_match_per_block_sdpa():
    torch.manual_seed(1)
    nh, hd = 2, 3
    sizes = [2, 4, 1]
    cu = torch.tensor([0] + [sum(sizes[: i + 1]) for i in range(len(sizes))], dtype=torch.int32)
    L = int(cu[-1])
    q = torch.randn(L, nh, hd)
    k = torch.randn(L, nh, hd)
    v = torch.randn(L, nh, hd)

    packed = packed_sdpa_attention(q, k, v, cu, cu)
    ref = torch.cat(
        [_ref_block(q[int(cu[i]) : int(cu[i + 1])], k[int(cu[i]) : int(cu[i + 1])],
                    v[int(cu[i]) : int(cu[i + 1])], nh, hd) for i in range(len(sizes))],
        dim=0,
    )
    assert torch.allclose(packed, ref, atol=1e-6)


def test_single_block_path():
    torch.manual_seed(2)
    nh, hd, S = 2, 3, 4
    q = torch.randn(S, nh, hd)
    k = torch.randn(S, nh, hd)
    v = torch.randn(S, nh, hd)
    packed = packed_sdpa_attention(q, k, v, None, None)
    ref = _ref_block(q, k, v, nh, hd)
    assert torch.allclose(packed, ref, atol=1e-6)


def test_output_head_layout_is_position_contiguous():
    """Regression guard for the reshape bug: heads must be contiguous WITHIN each position,
    not grouped as [head0:all positions][head1:all positions]. A transpose(0,1).reshape
    mistake would fail this."""
    torch.manual_seed(3)
    nh, hd, S = 3, 2, 4
    q = torch.randn(S, nh, hd)
    k = torch.randn(S, nh, hd)
    v = torch.randn(S, nh, hd)
    out = packed_sdpa_attention(q, k, v, None, None)  # (S, nh*hd)
    # position p's head h is at out[p, h*hd : (h+1)*hd]; rebuild and compare to the per-head SDPA
    attn = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), dropout_p=0.0,
    )  # (1, nh, S, hd)
    for h in range(nh):
        assert torch.allclose(out[:, h * hd : (h + 1) * hd], attn[0, h], atol=1e-6)
