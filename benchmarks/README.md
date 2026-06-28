# benchmarks/

Reproducible scripts for the numbers in the README.

- `throughput.py` — packed-SDPA vs stock, per-batch ms/img + img/s + per-stage (vision/prefill/decode) + peak VRAM. Needs the vendored `locateanything_batch` engine on `LA_BATCH_ENGINE_PATH` (or importable).

The root-cause scripts that produced the investigation (profiler → backend matrix → reshape-bug postmortem) live in the PerceptGrid repo under `scripts/locateanything_vision_*.py` and are referenced from the README. They are kept there because they depend on the full worker environment; this repo only ships the reproducible speed/quality bench.
