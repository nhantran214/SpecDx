#!/usr/bin/env bash
# Prove vendor/tonomyx is byte-identical to the pinned TonomyX commit.
set -euo pipefail
SRC=${TONOMYX_SRC:-/home/ethan/AILab/TonomyX}
COMMIT=34c1d2ce70583b2de54573d28ba4f49e0af4cc7f
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
git -C "$SRC" archive "$COMMIT" tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo | tar -x -C "$TMP"
diff -rq -x __pycache__ -x '*.egg-info' "$TMP" "$HERE/vendor/tonomyx"
echo "vendor/tonomyx matches $COMMIT"

# vendor/sopbench: checked only when a clone is given in SOPBENCH_SRC.
SOPBENCH_COMMIT=8790ed7406acbd22270f90bd1c4038de3fc5db8f
if [ -n "${SOPBENCH_SRC:-}" ]; then
  TMP2=$(mktemp -d)
  git -C "$SOPBENCH_SRC" archive "$SOPBENCH_COMMIT" env data reference LICENSE DATA_LICENSE README.md requirements.txt | tar -x -C "$TMP2"
  diff -rq -x __pycache__ "$TMP2" "$HERE/vendor/sopbench"
  echo "vendor/sopbench matches $SOPBENCH_COMMIT"
else
  echo "vendor/sopbench: skipped (set SOPBENCH_SRC to a clone of Leezekun/SOPBench to check)"
fi
