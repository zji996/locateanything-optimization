"""locateanything-optimization — packed batched-SDPA vision for NVIDIA LocateAnything-3B.

Drop-in speedup for LocateAnything-3B's MoonViT vision encoder. The stock
``sdpa_attention`` (modeling_vit.py) builds a dense ``[1,S,S]`` bool mask which forces
PyTorch SDPA onto the slow **math** backend (~1278 ms/img on an RTX 4000 Ada, and it does
NOT scale down with batch because the engine falls back to per-image encode when the
flash-attn wheel is absent). This package replaces it with a **packed batched SDPA** that
runs a single true-batched ``[N,heads,S,hd]`` ``scaled_dot_product_attention`` (no mask),
letting SDPA pick its fused flash / mem-efficient kernel.

Measured on RTX 4000 Ada (compute 8.9, 20 GB), BF16, torch 2.12.1+cu130, no flash-attn
wheel:

================  ======  ========  ========  =======
batch              ms/img    img/s   vision ms  peak GB
================  ======  ========  ========  =======
1 (stock)          1602      0.62      1278     10.41
1 (packed, this)    415      2.41       119      7.88
8 (stock)          1474      0.68      1277     10.47
8 (packed, this)    317      3.16       121      9.03
================  ======  ========  ========  =======

~10x vision / ~4.6x end-to-end at batch 8, **with no quality regression** on crowd scenes
(tmp/1.jpg crowd: 13 people, identical to stock under production sampling). An earlier
investigation reported a 13->1 crowd-count regression, but that was a reshape bug in the
experiment's single-image branch (``transpose(0,1).reshape`` scrambles head ordering); the
correct ``permute(0,2,1,3).reshape`` preserves per-position head grouping and matches stock.

Quickstart::

    import locateanything_optimization as la_opt
    from transformers import AutoModel  # your existing load
    model = AutoModel.from_pretrained("nvidia/LocateAnything-3B", ...).to("cuda").eval()
    la_opt.install_packed_sdpa(model)   # patch MoonViT; returns a restore() callable
    # ... run the stock model.generate() OR the vendored batched engine -- vision is now ~10x
    restore = la_opt.install_packed_sdpa(model)  # restore() reverts to stock attention

This is a pure monkeypatch on the model's dynamic module (``VL_VISION_ATTENTION_FUNCTIONS``);
it contains no NVIDIA model code and is MIT-licensed. See ``README.md`` for the full
investigation (profiler, root-cause, the reshape-bug postmortem) and ``benchmarks/`` for
reproducible scripts.
"""

from .vision import (
    compile_vision,
    install_packed_sdpa,
    install_real_rope,
    packed_sdpa_attention,
    real_apply_rope,
)

__all__ = [
    "install_packed_sdpa",
    "packed_sdpa_attention",
    "install_real_rope",
    "real_apply_rope",
    "compile_vision",
]
__version__ = "0.2.0"
