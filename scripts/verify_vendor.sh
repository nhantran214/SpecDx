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
