# locateanything-optimization

**~4.6× faster end-to-end inference for [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B), with no quality regression.**

A drop-in monkeypatch that speeds up LocateAnything-3B's MoonViT vision encoder by replacing its slow dense-mask SDPA with a packed batched fused-SDPA. No flash-attn wheel required, no model code redistributed, no fine-tuning.

```python
import locateanything_optimization as la_opt
from transformers import AutoModel

model = AutoModel.from_pretrained("nvidia/LocateAnything-3B", dtype="bfloat16",
                                  trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()
la_opt.install_packed_sdpa(model)   # one line; vision encode is now ~10× faster
# ... run model.generate() or any batched engine exactly as before
```

## Why it's faster

MoonViT packs N images into one packed `[L, D]` sequence and, in its stock `sdpa_attention`, builds a **dense `[1, L, L]` boolean mask** so each image attends only within itself (block-diagonal). That boolean mask is unsupported by the fused flash / mem-efficient SDPA kernels, so PyTorch falls back to the slow **math** backend — which materializes the full attention matrix. Result: ~1278 ms/img on an RTX 4000 Ada, and it does **not** improve with batch (the stock batched engine encodes images one at a time when the flash-attn wheel is absent, to avoid the O(N²) packed mask).

This package replaces that call with a **true batched SDPA**: reshape the packed `q/k/v` to `[N, heads, S, head_dim]` (each image is its own batch element) and run one fused `scaled_dot_product_attention` with no mask. Block-diagonal by construction → numerically the same attention as per-image, and ~10× faster because SDPA now picks the fused flash kernel.

## Results

RTX 4000 Ada (compute 8.9, 20 GB), BF16, torch 2.12.1+cu130, **no flash-attn wheel**, greedy detect on synthetic 1024×768 single-target images:

| batch | stock ms/img | packed ms/img | stock img/s | packed img/s | vision ms (stock → packed) | peak VRAM |
|------:|------------:|--------------:|------------:|-------------:|---------------------------:|----------:|
| 1     | 1602        | **415**       | 0.62        | **2.41**     | 1278 → **119**             | 10.4 → 7.9 GB |
| 4     | 1480        | **322**       | 0.68        | **3.11**     | 1277 → **123**             | 10.4 → 8.4 GB |
| 8     | 1474        | **317**       | 0.68        | **3.16**     | 1277 → **121**             | 10.5 → 9.0 GB |
| 16    | —           | **311**       | —           | **3.22**     | — → **121**                | 10.3 GB |

Vision encode (the MoonViT forward) drops from 87% of wall time to ~38%. Reproduce with `benchmarks/throughput.py`.

**Quality**: identical to stock on crowd scenes. `tmp/1.jpg` (a crowd photo) gives **13 people** under production sampling (temp 0.7, top_p 0.9) both before and after patching — see the root-cause notes below for why an earlier investigation saw a regression.

## The investigation (and the bug that wasn't)

The first A/B reported a **13 → 1 crowd-count regression** from this path, concluding "speed and correctness can't coexist". A systematic root-cause pass disproved it:

1. **`rootcause2`**: forcing math backend on both `3D (nh,S,hd)` and `4D (1,nh,S,hd)` SDPA inputs gives `max_abs_diff = 0.0` — the attention op is numerically identical across shapes. So the regression wasn't in the attention math.
2. **The actual cause**: the experiment's single-image branch used `out.transpose(0,1).reshape(L, nh*hd)`, which lays out heads as `[head0:all positions][head1:all positions]` instead of `[position:all heads]`. That scrambled MoonViT's `wo` projection input and corrupted long decodes. The correct `permute(0,2,1,3).reshape` (used here, and pinned by `tests/test_packed_sdpa.py`) preserves per-position head grouping and matches stock.

The regression tests (`test_output_head_layout_is_position_contiguous`) exist specifically to catch this class of mistake.

## Install

```bash
pip install -e .          # from this repo
# or, once published:
pip install locateanything-optimization
```

Requires `torch >= 2.1` with CUDA (the fused SDPA kernels need a GPU). Works with or without the `flash-attn` wheel; if present, MoonViT already uses `flash_attention_2` and this patch is a no-op on those blocks.

## Requirements

- LocateAnything-3B loaded with `attn_implementation="sdpa"` (its default).
- A CUDA GPU with compute capability ≥ 8.0 (flash/mem-efficient SDPA backends).

## License

MIT. This package contains only the patch code (a monkeypatch on the model's dynamic module); the NVIDIA LocateAnything-3B model weights and code remain under their own license and are **not** redistributed here.
