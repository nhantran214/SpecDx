# Vendored artefacts

## tonomyx

**Source:** `/home/ethan/AILab/TonomyX`, commit `34c1d2ce70583b2de54573d28ba4f49e0af4cc7f` (master, 2026-09-30). Its `tonomyx/` and `data/` trees are identical to commit `9de7c28`, the one FAA vendored.

**Copied with:** `git archive 34c1d2c tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo`. That is the package, its build file, the canon pack, and the three requirement PDFs that form the Tell corpus.

**Rule:** nothing under `vendor/` is ever edited. Mutants are copies under `work/mutants/`.

**Check:** `bash scripts/verify_vendor.sh` prints `vendor/tonomyx matches 34c1d2c…` and nothing else.

## sopbench

**Source:** `https://github.com/Leezekun/SOPBench`, commit `8790ed7406acbd22270f90bd1c4038de3fc5db8f` (2026-09-13). Code is MIT ("The SOPBench Authors"); data is CC BY 4.0 (`DATA_LICENSE`).

**Copied with:** `git archive 8790ed7 env data reference LICENSE DATA_LICENSE README.md requirements.txt | tar -x -C vendor/sopbench`. That is the domain environments, the task data, the reference material, both licences, the README and the requirements.

**No `output/`:** the published trajectories (about 2 GB) are not vendored. They are fetched by `specdx p3 fetch-sopbench`.

**Rule:** nothing under `vendor/` is ever edited.

**Check:** `SOPBENCH_SRC=<clone> bash scripts/verify_vendor.sh` also prints `vendor/sopbench matches 8790ed7…`; without `SOPBENCH_SRC` it prints a skip note for sopbench.
