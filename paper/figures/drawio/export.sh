#!/usr/bin/env bash
# Export every .drawio here to a cropped vector PDF (for LaTeX) and a draft PNG (for review).
set -euo pipefail
cd "$(dirname "$0")"
for f in *.drawio; do
  b=${f%.drawio}
  timeout 180 xvfb-run -a drawio -x -f pdf --crop -o "$b.pdf" "$f" --disable-gpu --no-sandbox >/dev/null 2>&1
  timeout 180 xvfb-run -a drawio -x -f png --width 1600 -o "draft-$b.png" "$f" --disable-gpu --no-sandbox >/dev/null 2>&1
  echo "$f -> $b.pdf"
done
