#!/usr/bin/env bash
# P3 driver (docs/RUNNING.md §11). One command per stage; every finished stage leaves a
# Markdown report in results/p3/<stage>.md, and results/p3/STATUS.md shows all stages.
#
#   ./run_all.sh all             every implemented stage below, in order; skips a stage that
#                                already finished (exit 0 on record); stops at the first
#                                non-zero exit, then rebuilds STATUS.md
#   ./run_all.sh status          rebuild results/p3/STATUS.md
#   ./run_all.sh <stage>         one stage (the list below)
#
# Stage              Plan  What it does
#   local            P3a   tests, lint and a replication smoke, on this machine
#   fetch-sopbench   P3a   download SOPBench's published trajectories (the only network use)
#   replication      P3a   offline replication of the S3 diagnosis on SOPBench
#   s3-screen        P3b   S3 screen                       (pending)
#   s3-dev           P3b   S3 development runs             (pending)
#   s3-calibrate     P3b   S3 calibration                  (pending)
#   s3-gate          P3b   S3 go/no-go check               (pending)
#   s3-baselines-dev P3b   S3 baselines on dev             (pending)
#   s2-screen        P3c   S2 screen                       (pending)
#   s2-dev           P3c   S2 development runs             (pending)
#   s2-calibrate     P3c   S2 calibration                  (pending)
#   s2-gate          P3c   S2 go/no-go check               (pending)
#   s2-baselines-dev P3c   S2 baselines on dev             (pending)
#   freeze           P3b   freeze the preregistration      (pending)
#   s3-test          P3b   S3 test split                   (pending)
#   s2-test          P3c   S2 test split                   (pending)
#   s2-real          P3c   S2 real-subject runs            (pending)
#   ablations        P3d   ablations                       (pending)
#   status           P3a   status dashboard
# A stage that is not implemented yet prints "<stage>: pending — implemented in plan <Px>",
# rebuilds STATUS.md and exits 0.
#
# Exit status: 0 done; 1 a check failed; 2 a usage or environment error; 3 a screen stop;
# 4 a test split is locked.
#
# Environment:
#   P3_RESULTS   where reports go (default results/p3)
#   P3_WORK      work directory: caches and the p3-<stage>.pid files (default work)
# While a stage runs, $P3_WORK/p3-<stage>.pid holds its pid; it is removed when the stage
# ends, however it ends, so STATUS.md shows "running" only for a live stage.
# Later local SOPBench runs need PYTHONHASHSEED=0 (docs/RUNNING.md §11).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=.venv/bin/python
SPECDX=.venv/bin/specdx
RESULTS="${P3_RESULTS:-results/p3}"
WORK="${P3_WORK:-work}"
PIDFILE=""

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }
die()  { printf 'error: %s\n' "$*" >&2; exit 2; }

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 2
}

check_tools() {
  step "Tools"
  [[ -x $PY && -x $SPECDX ]] || die ".venv is missing: see docs/RUNNING.md §2"
  command -v git > /dev/null || die "git is not installed"
  ok "python, specdx, git"
}

cleanup() { [[ -n $PIDFILE ]] && rm -f "$PIDFILE"; return 0; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

p3() { "$SPECDX" p3 "$1" --results "$RESULTS" --work "$WORK" "${@:2}"; }

rebuild_status() { p3 status > /dev/null; }

show_paths() {
  [[ -f $RESULTS/$1.md ]] && echo "report: $RESULTS/$1.md"
  echo "status: $RESULTS/STATUS.md"
  return 0
}

# stage_info NAME -> "PLAN IMPLEMENTED" from STAGES (the single source of truth)
stage_info() {
  "$PY" - "$1" <<'PYEOF'
import sys

from specdx.report.stages import STAGES

for s in STAGES:
    if s.name == sys.argv[1]:
        print(s.plan, "1" if s.implemented else "0")
        break
else:
    sys.exit(1)
PYEOF
}

run_stage() {
  local name=$1 rc=0
  step "$name"
  mkdir -p "$WORK" "$RESULTS"
  PIDFILE="$WORK/p3-$name.pid"
  echo $$ > "$PIDFILE"
  p3 "$name" || rc=$?
  rm -f "$PIDFILE"
  PIDFILE=""
  rebuild_status
  show_paths "$name"
  return "$rc"
}

run_one() {
  local name=$1 info plan impl
  info=$(stage_info "$name") || die "unknown stage '$name' (run ./run_all.sh for the list)"
  read -r plan impl <<< "$info"
  if [[ $impl == 0 ]]; then
    echo "$name: pending — implemented in plan $plan"
    rebuild_status
    echo "status: $RESULTS/STATUS.md"
    return 0
  fi
  if [[ $name == status ]]; then
    rebuild_status
    echo "status: $RESULTS/STATUS.md"
    return 0
  fi
  run_stage "$name"
}

run_all() {
  local name rc=0
  while read -r name; do
    if p3 status --done "$name"; then
      ok "$name: already done, skipped"
      continue
    fi
    run_stage "$name" || rc=$?
    if [[ $rc != 0 ]]; then
      warn "$name stopped with exit $rc"
      return "$rc"
    fi
  done < <("$PY" -c '
from specdx.report.stages import STAGES
for s in STAGES:
    if s.implemented and s.name != "status":
        print(s.name)')
  rebuild_status
  echo "status: $RESULTS/STATUS.md"
}

[[ $# -ge 1 ]] || usage
case "$1" in
  -h|--help|help) usage ;;
esac
check_tools
case "$1" in
  all) run_all ;;
  *) run_one "$1" ;;
esac
