#!/usr/bin/env python3
"""Reproduce the packed-SDPA vs stock throughput table.

Loads LocateAnything-3B once, then runs the stock (dense-mask math) vision path and the
packed fused-SDPA path back-to-back at several batch sizes, reporting ms/img, img/s,
per-stage (vision/prefill/decode) and peak VRAM. Requires the vendored batched engine OR
the stock model; the vision patch works on either.

Usage (inside a worker-like env with the model cached offline):
    LA3B_MODEL=/path/to/LocateAnything-3B python benchmarks/throughput.py
    LA3B_MODEL=... LA_BATCHES=1,4,8,16 python benchmarks/throughput.py
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# Prefer the vendored batched engine if present (gives the full generate_batch + per-stage
# stats); fall back to raw AutoModel + this package's patch otherwise.
try:
    sys.path.insert(0, os.environ.get("LA_BATCH_ENGINE_PATH", ""))
    import locateanything_batch as lab  # type: ignore
    _HAVE_BATCH_ENGINE = True
except Exception:
    _HAVE_BATCH_ENGINE = False

import locateanything_optimization as la_opt

MAX_NEW = int(os.environ.get("LA_MAX_NEW_TOKENS", "128"))
NIMG = int(os.environ.get("LA_NIMG", "32"))
ROUNDS = int(os.environ.get("LA_ROUNDS", "4"))
BATCHES = [int(b) for b in os.environ.get("LA_BATCHES", "1,4,8,16").split(",")]

_W, _H, _BG = 1024, 768, (210, 210, 210)
_SPECS = [
    ((220, 40, 40), (120, 140, 470, 520)),
    ((40, 160, 60), (560, 100, 880, 360)),
    ((40, 90, 220), (320, 380, 700, 690)),
    ((240, 140, 20), (700, 420, 960, 700)),
    ((150, 40, 190), (140, 90, 380, 470)),
    ((20, 170, 190), (430, 220, 760, 540)),
    ((210, 40, 140), (600, 60, 900, 300)),
    ((180, 170, 30), (200, 300, 560, 660)),
]


def make_images():
    from PIL import Image, ImageDraw

    imgs = []
    for color, box in _SPECS:
        im = Image.new("RGB", (_W, _H), _BG)
        ImageDraw.Draw(im).rectangle(box, fill=color)
        imgs.append(im)
    return imgs


def measure(pairs, batch, rounds=ROUNDS):
    warm = pairs[:batch] if batch <= len(pairs) else pairs
    lab.generate_batch(warm, temperature=0.0, max_new_tokens=MAX_NEW, return_stats=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last = {}
    for _ in range(rounds):
        for i in range(0, len(pairs), batch):
            _o, last = lab.generate_batch(pairs[i : i + batch], temperature=0.0, max_new_tokens=MAX_NEW, return_stats=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = len(pairs) * rounds
    return {
        "ms_per_image": round(dt / n * 1e3, 1),
        "images_per_s": round(n / dt, 3),
        "vision_ms": round(float(last.get("vision_encode_ms", 0)) / batch, 1),
        "prefill_ms": round(float(last.get("llm_prefill_ms", 0)) / batch, 1),
        "decode_ms": round(float(last.get("mtp_decode_ms", 0)) / batch, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    if not _HAVE_BATCH_ENGINE:
        raise SystemExit("benchmarks/throughput.py needs the locateanything_batch engine on LA_BATCH_ENGINE_PATH")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    lab.engine._PROMPT = "Detect the "
    _tok, _proc, model = lab.load()
    print(f"[tp] device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    imgs = make_images()
    pairs = [(imgs[i % len(imgs)], "rectangle") for i in range(NIMG)]

    result = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_images": NIMG, "rounds": ROUNDS}

    # stock path (dense-mask math): patch is NOT installed
    print("[tp] === stock (dense-mask math) ===", flush=True)
    result["stock"] = {}
    for b in BATCHES:
        r = measure(pairs, b)
        result["stock"][f"batch_{b}"] = r
        print(f"[tp] STOCK b={b}: {r['ms_per_image']} ms/img  {r['images_per_s']} img/s  "
              f"vision={r['vision_ms']} peak={r['peak_vram_gb']}GB", flush=True)

    # packed SDPA path (this package)
    restore = la_opt.install_packed_sdpa(model)
    lab.engine.BATCH_VISION = True
    # the batched engine gates the packed branch on _vision_is_flash; force True for the bench
    lab.engine._vision_is_flash = lambda: True  # noqa: E731
    print("[tp] === packed SDPA (this package) ===", flush=True)
    result["packed"] = {}
    for b in BATCHES:
        r = measure(pairs, b)
        result["packed"][f"batch_{b}"] = r
        print(f"[tp] PACK  b={b}: {r['ms_per_image']} ms/img  {r['images_per_s']} img/s  "
              f"vision={r['vision_ms']} prefill={r['prefill_ms']} decode={r['decode_ms']} peak={r['peak_vram_gb']}GB", flush=True)
    restore()

    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = os.environ.get("LA_OUT", "throughput.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[tp] wrote {out}")


if __name__ == "__main__":
    main()
