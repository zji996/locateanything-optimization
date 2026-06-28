"""Packed batched-SDPA attention for LocateAnything-3B's MoonViT vision encoder.

MoonViT packs N images into one packed ``[L, D]`` sequence (L = sum of per-image token
counts) and, in its stock ``sdpa_attention``, builds a dense ``[1, L, L]`` boolean mask so
each image attends only within itself (block-diagonal). That boolean mask is unsupported by
the fused flash / mem-efficient SDPA kernels, so PyTorch falls back to the slow **math**
backend -- which materializes the full attention matrix. The result is ~1278 ms/img on an
RTX 4000 Ada, and it does not improve with batch (the vendored batched engine encodes images
one at a time when the flash-attn wheel is absent, to avoid the O(N^2) packed mask).

This module replaces that call with a **true batched SDPA**: reshape the packed ``q/k/v`` to
``[N, heads, S, head_dim]`` (each image is its own batch element) and run one fused
``scaled_dot_product_attention`` with no mask. Block-diagonal by construction -> numerically
the same attention as per-image (modulo bf16 accumulation-order drift ~0.016 max-abs per op,
within model tolerance), and ~10x faster because SDPA now picks the fused flash kernel.

Output layout matters: ``out`` is ``permute(0,2,1,3)`` before ``reshape`` so heads are
contiguous within each position (what MoonViT's ``wo`` projection consumes). An earlier
experiment used ``transpose(0,1).reshape``, which scrambles head ordering into
``[head0:all positions][head1:all positions]`` and corrupts long decodes (the crowd-count
13->1 regression). The regression tests pin the correct layout.
"""

from __future__ import annotations

import importlib
from typing import Callable

import torch
import torch.nn.functional as F


def packed_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-diagonal attention over a packed sequence, via true-batched fused SDPA.

    Args:
        q, k, v: packed ``(L, heads, head_dim)`` tensors (MoonViT's native packed layout).
        q_cu_seqlens / k_cu_seqlens: cumulative per-image block boundaries (``[0, S, 2S, ...]``).
            ``None`` or a length-2 tensor means a single block (one image).

    Returns:
        ``(L, heads * head_dim)`` -- per-position head groups contiguous, ready for ``wo``.

    For uniform image sizes (the common case in a batch) the whole batch is reshaped to
    ``(N, heads, S, head_dim)`` and run in one SDPA call. Ragged sizes fall back to a
    per-block SDPA loop (still fused per block, still no dense mask).
    """
    L = q.shape[0]
    nh, hd = q.shape[1], q.shape[2]
    if q_cu_seqlens is None or q_cu_seqlens.numel() <= 2:
        # single block: out (1, heads, L, head_dim) -> permute to (1, L, heads, head_dim) -> flat
        q4 = q.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q4, k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), dropout_p=0.0
        )
        return out.permute(0, 2, 1, 3).reshape(L, nh * hd)
    diffs = q_cu_seqlens[1:] - q_cu_seqlens[:-1]
    dmin, dmax = int(diffs.min().item()), int(diffs.max().item())
    if dmin == dmax and dmin > 0:
        N = int(q_cu_seqlens.numel()) - 1
        S = dmin
        q4 = q.reshape(N, S, nh, hd).permute(0, 2, 1, 3)
        k4 = k.reshape(N, S, nh, hd).permute(0, 2, 1, 3)
        v4 = v.reshape(N, S, nh, hd).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0)
        return out.permute(0, 2, 1, 3).reshape(L, nh * hd)
    # ragged: per-block fused SDPA loop
    outs = []
    for i in range(int(q_cu_seqlens.numel()) - 1):
        s, e = int(q_cu_seqlens[i]), int(q_cu_seqlens[i + 1])
        qi = q[s:e].transpose(0, 1).unsqueeze(0)
        ki = k[s:e].transpose(0, 1).unsqueeze(0)
        vi = v[s:e].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        outs.append(out.permute(0, 2, 1, 3).reshape(e - s, nh * hd))
    return torch.cat(outs, dim=0)


def install_packed_sdpa(model: torch.nn.Module) -> Callable[[], None]:
    """Patch a loaded LocateAnything-3B model's MoonViT to use :func:`packed_sdpa_attention`.

    Swaps ``VL_VISION_ATTENTION_FUNCTIONS["sdpa"]`` in the model's dynamic module so every
    MoonViT attention block runs the packed fused-SDPA path instead of the stock dense-mask
    math path. Safe to call on a model loaded with ``attn_implementation="sdpa"`` (the
    LocateAnything default). Calling twice is a no-op (the second call re-patches the already-
    patched function). Returns a ``restore()`` callable that reverts to the original attention.

    Requires the model's ``vision_model.encoder.blocks[*].attn_implementation`` to be
    ``"sdpa"`` (the LocateAnything-3B default); flash_attention_2 blocks are left untouched
    (they already use flash). After patching, encode N images by packing their
    ``pixel_values`` / ``grid_hws`` into one ``model.extract_feature`` call.
    """
    vm = model.vision_model
    mod = importlib.import_module(type(vm).__module__)
    table = getattr(mod, "VL_VISION_ATTENTION_FUNCTIONS", None)
    if table is None or "sdpa" not in table:
        raise ValueError(
            f"{mod.__name__} has no VL_VISION_ATTENTION_FUNCTIONS['sdpa']; "
            "is this a LocateAnything-3B MoonViT module?"
        )
    original = table["sdpa"]
    table["sdpa"] = packed_sdpa_attention

    def restore() -> None:
        table["sdpa"] = original

    return restore
