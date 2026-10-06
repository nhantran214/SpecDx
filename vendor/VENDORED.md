# Vendored artefacts

## tonomyx

**Source:** `/home/ethan/AILab/TonomyX`, commit `34c1d2ce70583b2de54573d28ba4f49e0af4cc7f` (master, 2026-09-30). Its `tonomyx/` and `data/` trees are identical to commit `9de7c28`, the one FAA vendored.

**Copied with:** `git archive 34c1d2c tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo`. That is the package, its build file, the canon pack, and the three requirement PDFs that form the Tell corpus.

**Rule:** nothing under `vendor/` is ever edited. Mutants are copies under `work/mutants/`.

**Check:** `bash scripts/verify_vendor.sh` prints `vendor/tonomyx matches 34c1d2c…` and nothing else.
