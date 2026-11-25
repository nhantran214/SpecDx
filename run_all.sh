#!/usr/bin/env bash
# P3 driver (docs/RUNNING.md §11). One command per stage; every finished stage leaves a
# Markdown report in results/p3/<stage>.md, and results/p3/STATUS.md shows all stages.
#
#   ./run_all.sh all             every implemented stage below, in order; skips a stage that
#                                already finished (exit 0 on record); stops at the first
#                                non-zero exit, then rebuilds STATUS.md
#   ./run_all.sh status          rebuild results/p3/STATUS.md
#   ./run_all.sh <stage>         one stage (the list below)
#   ./run_all.sh smoke           the four P4a stages, in order, in smoke mode (below)
#
# Stage              Plan  What it does
#   local            P3a   tests, lint and a replication smoke, on this machine
#   fetch-sopbench   P3a   download SOPBench's published trajectories (the only network use)
#   replication      P3a   offline replication of the S3 diagnosis on SOPBench
#   s3-g0v4          P4a   calibrate on the dev domains, then G0v4; freezes the v4 parameters
#   s3-witness       P4a   execution witness of every task of every domain
#   s3-v4            P4a   v4 on the held-out domains, v3 beside it, and the ablations
#   s3-placebo       P4a   permutation placebo on the held-out domains
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
# Smoke mode (P4a): official P4a results come only from the 48 GB host; this machine runs
# quick system checks. P4_SMOKE=1 or `./run_all.sh smoke` runs the P4a stages shrunk (2 G0v4
# replicates on a one-point grid, 20 tasks per domain for the witnesses, one held-out domain,
# 2 placebo seeds). Reports go to $P3_RESULTS/smoke/ (labelled SMOKE, with its own
# STATUS.md); nothing is frozen and no official stage is marked done. `smoke` runs all four
# stages even if one fails, prints a summary (stage, exit code, verdict) and exits with the
# largest stage code, so a crash (2 or more) is never hidden by an expected FAIL (1): the
# smoke placebo has one held-out domain and so always ends FAIL (1).
# `all` (outside smoke mode) includes the OFFICIAL P4a stages; run it on the 48 GB host only.
#
# Environment:
#   P3_RESULTS   where reports go (default results/p3)
#   P4_SMOKE     1: smoke mode (the P4a stages only); unset, empty or 0: off; any other
#                value is an error (exit 2)
#   P3_WORK      work directory: caches and the p3-<stage>.pid files (default work)
# While a stage runs, $P3_WORK/p3-<stage>.pid holds its pid; it is removed when the stage
# ends, however it ends, so STATUS.md shows "running" only for a live stage.
# Later local SOPBench runs need PYTHONHASHSEED=0 (docs/RUNNING.md §11).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

case "${P4_SMOKE:-}" in
  ""|0|1) ;;
  *) printf 'error: P4_SMOKE must be unset, empty, 0 or 1 (got %q)\n' "$P4_SMOKE" >&2; exit 2 ;;
esac

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

SHOWN="$RESULTS"
[[ ${P4_SMOKE:-} == 1 ]] && SHOWN="$RESULTS/smoke"

show_paths() {
  [[ -f $SHOWN/$1.md ]] && echo "report: $SHOWN/$1.md"
  echo "status: $SHOWN/STATUS.md"
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
  local pids="$WORK"
  [[ ${P4_SMOKE:-} == 1 ]] && pids="$WORK/smoke"
  mkdir -p "$WORK" "$RESULTS" "$pids"
  PIDFILE="$pids/p3-$name.pid"
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
  if [[ $name == status ]]; then
    rebuild_status
    echo "status: $SHOWN/STATUS.md"
    return 0
  fi
  if [[ ${P4_SMOKE:-} == 1 && $plan != P4a ]]; then
    die "P4_SMOKE=1 applies to the P4a stages only, not $name"
  fi
  if [[ $impl == 0 ]]; then
    echo "$name: pending — implemented in plan $plan"
    rebuild_status
    echo "status: $RESULTS/STATUS.md"
    return 0
  fi
  run_stage "$name"
}

p4a_stages() {
  "$PY" -c '
from specdx.report.stages import STAGES
for s in STAGES:
    if s.plan == "P4a" and s.implemented:
        print(s.name)'
}

# max_code N...  the largest of the exit codes given (0 for none)
max_code() {
  local m=0 c
  for c in "$@"; do (( c > m )) && m=$c; done
  echo "$m"
}

verdict_of() {
  "$PY" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("verdict") or "-")
except Exception:
    print("-")' "$SHOWN/$1.json"
}

run_smoke() {
  local name rc codes=() names=() i
  export P4_SMOKE=1
  SHOWN="$RESULTS/smoke"
  echo "SMOKE RUN — system check only, not a result"
  while read -r name; do
    rc=0
    run_stage "$name" || rc=$?
    [[ $rc != 0 ]] && warn "smoke $name exited $rc"
    names+=("$name")
    codes+=("$rc")
  done < <(p4a_stages)
  rebuild_status
  step "Smoke summary"
  printf '  %-12s %-5s %s\n' stage exit verdict
  for i in "${!names[@]}"; do
    printf '  %-12s %-5s %s\n' "${names[$i]}" "${codes[$i]}" "$(verdict_of "${names[$i]}")"
  done
  echo "status: $SHOWN/STATUS.md"
  return "$(max_code "${codes[@]}")"
}

run_all() {
  local name rc=0
  warn "all includes the official P4a stages (s3-g0v4 can run for hours and freezes the v4 parameters on PASS): run it on the 48 GB host only; use ./run_all.sh smoke here"
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

main() {
  [[ $# -ge 1 ]] || usage
  case "$1" in
    -h|--help|help) usage ;;
  esac
  check_tools
  case "$1" in
    smoke) run_smoke ;;
    all) if [[ ${P4_SMOKE:-} == 1 ]]; then run_smoke; else run_all; fi ;;
    *) run_one "$1" ;;
  esac
}

# Sourced (by the tests), the functions are defined and nothing runs.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
