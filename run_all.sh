#!/usr/bin/env bash
# P2b and P3 driver (docs/RUNNING.md §11 and §13, experiments/p2-preregistration.md).
#
#   ./run_all.sh local          smoke tests on this machine, before any GPU time.
#                               Writes only to a scratch directory: never to work/, results/
#                               or experiments/.
#   ./run_all.sh all [--yes]    the 48 GB run, every step below in order. Before the gate it
#                               reports the measured seconds per run and asks; --yes skips
#                               the question, and answering no stops `all` (exit 0).
#   ./run_all.sh g2 | preflight | g3 | gate [--yes] | status    one step at a time.
#   ./run_all.sh baselines-dev       every baseline arm on the dev split (after the gate)
#   ./run_all.sh freeze-baselines    choose each baseline's arm and τ on dev; commit the pre-registration
#   ./run_all.sh study               SpecDx and the frozen baselines on the test split (gate PASSED only)
#   ./run_all.sh report              tables A and B and the statistics (dev, and test when present)
#   ./run_all.sh s2                  S2 (τ²-bench) is not built yet
# `all` now runs g2 -> preflight -> g3 -> gate -> baselines-dev -> freeze-baselines -> study ->
# report in the foreground (run it inside tmux). Exit 4: the gate is NOT PASSED, so the test
# study is locked (baselines-dev and the dev report are still produced; nothing is frozen).
#
# The 48 GB steps run SpecDx and PostgreSQL here, and Ollama on the GPU host through an SSH
# tunnel:
#   ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
#     -L 11435:localhost:11434 user@gpu-host
#   SPECDX_OLLAMA_URL=http://localhost:11435 ./run_all.sh all
# On the GPU host: Ollama 0.30.4 with OLLAMA_NUM_PARALLEL=2, OLLAMA_MAX_LOADED_MODELS=2 and
# OLLAMA_KEEP_ALIVE=-1, and the population's models pulled.
# G3 takes hours: run `all` (or `g3`) inside tmux or screen so a dropped login does not kill
# it. If a step stops, run the same command again: finished runs come from the cache. Run as
# single steps, the gate, baselines-dev and study are detached (setsid nohup, with a pid and
# a log in work/); under `all` (FOREGROUND=1) they run in the foreground, one after another,
# still with a pid file while they run, so `status` and the second-process guards see them.
#
# Exit status: 0 done; 1 a check failed (P2 stops, report it); 2 a usage or environment error;
# 3 the G2 stop rule (spec v3.1 §7); 4 the test study is locked: the gate is NOT PASSED, the
# baseline pre-registration is not committed, or the working tree has uncommitted changes
# (spec §3.7).
#
# Environment:
#   SPECDX_OLLAMA_URL   Ollama endpoint (default http://localhost:11434 for local; required
#                       for the 48 GB steps)
#   WORKERS             --workers for g3, gate, baselines-dev and study (default 2; never
#                       above the host's OLLAMA_NUM_PARALLEL)
#   FOREGROUND          1 runs gate, baselines-dev and study in the foreground, logging to
#                       work/NAME.log too (default 0: detached; `all` sets 1)
#   LOCAL_BUDGET        population runs in the local end-to-end smoke run (default 8;
#                       0 skips it)
#   SMOKE_DIR           scratch directory for `local` (default: a new mktemp -d)
#   GPU_SSH             user@gpu-host: G3 reads the host's Ollama log over SSH
#                       (journalctl -u ollama) to classify cached ReadTimeouts
#   OLLAMA_LOG_CMD      instead of GPU_SSH: a command printing the host's Ollama log between
#                       $SINCE and $UNTIL (epoch seconds), e.g. for an `ollama serve` logging
#                       to a file
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=.venv/bin/python
SPECDX=.venv/bin/specdx
CACHE_DIR="$ROOT/work"          # the run cache; passed per command, never exported
RUNS="$CACHE_DIR/runs.jsonl"
GATE_CONFIG=experiments/p2-gate.json
WORKERS="${WORKERS:-2}"
DB_CONTAINER=specdx-postgres

# The P1 run cache is append-only: its first 6505 lines must stay as P2a left them.
CACHE_PREFIX_LINES=6505
CACHE_PREFIX_MD5=db1e1823c66df102648c69eb6a8789b3
OLLAMA_VERSION=0.30.4

# The digests the cached P1b screen runs were keyed under. A host with another digest
# misses the cache and silently runs a different model.
declare -A PINNED_DIGEST=(
  [gpt-oss:20b]=f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe
  [qwen3:30b-a3b]=ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226
  [qwen2.5:14b]=7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6
  [qwen3:14b]=bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
  [mistral-small3.2:24b]=5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b
  [granite3.3:8b]=fd429f23b90980ed1bef53b990894e7b0199331f6ae90c5650240a7d5b70f1f7
  [llama3.1:8b]=46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e
)

# ---------------------------------------------------------------------------- output
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }
die()  { printf 'error: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------------------- shared checks
check_tools() {
  step "Tools"
  [[ -x $PY && -x $SPECDX ]] || die ".venv is missing: see docs/RUNNING.md §2"
  for t in jq curl git docker md5sum; do
    command -v "$t" > /dev/null || die "$t is not installed"
  done
  ok "python, specdx, jq, curl, git, docker"
}

check_database() {
  step "PostgreSQL (container $DB_CONTAINER, port 55434)"
  docker exec "$DB_CONTAINER" pg_isready -U tonomyx -d tonomyx > /dev/null 2>&1 \
    || die "$DB_CONTAINER is not ready: run 'docker start $DB_CONTAINER' (never docker compose up)"
  ok "accepting connections"
}

check_cache() {
  step "Run cache $RUNS (append-only)"
  [[ -f $RUNS ]] || die "$RUNS is missing"
  local lines prefix
  lines=$(wc -l < "$RUNS")
  prefix=$(head -n "$CACHE_PREFIX_LINES" "$RUNS" | md5sum | cut -d' ' -f1)
  [[ $prefix == "$CACHE_PREFIX_MD5" ]] \
    || fail "the first $CACHE_PREFIX_LINES lines changed (md5 $prefix, expected $CACHE_PREFIX_MD5)"
  ok "first $CACHE_PREFIX_LINES lines intact; $lines runs in all"
}

gate_running() {
  [[ -f work/p2-gate.pid ]] && kill -0 "$(cat work/p2-gate.pid)" 2> /dev/null
}

FOREGROUND="${FOREGROUND:-0}"
BASELINE_CACHE="$CACHE_DIR/baseline-llm.jsonl"
PREREG=experiments/p3-baselines-prereg

running() { [[ -f work/$1.pid ]] && kill -0 "$(cat "work/$1.pid")" 2> /dev/null; }

# launch NAME CMD...: foreground under `all` (FOREGROUND=1), else detached with a pid file.
launch() {
  local name=$1 rc=0 pid; shift
  if [[ $FOREGROUND == 1 ]]; then
    # The child itself (not a pipeline subshell) gets the pid file, so kill -STOP/-CONT,
    # `status` and the second-process guards act on it while `all` waits.
    "$@" > >(tee -a "work/$name.log") 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "work/$name.pid"
    # An async child ignores SIGINT in a script: pass ^C (and TERM) on, then reap it.
    trap 'kill "$pid" 2> /dev/null' INT TERM
    wait "$pid" || rc=$?
    if kill -0 "$pid" 2> /dev/null; then wait "$pid" 2> /dev/null || true; fi
    trap - INT TERM
    rm -f "work/$name.pid"
    ((rc == 0)) || fail "$name exited $rc (log work/$name.log); re-run the same step to resume"
    ok "$name finished; log work/$name.log"
    return
  fi
  setsid nohup "$@" > "work/$name.log" 2>&1 < /dev/null &
  echo $! > "work/$name.pid"
  ok "launched $name, pid $(cat "work/$name.pid"); log work/$name.log"
}

gate_passed() { [[ -f results/p2-gate.json ]] && jq -e '.passed == true' results/p2-gate.json > /dev/null 2>&1; }

# The test study refuses to run (spec §3.7): exit 4, never a detached launch that would fail.
locked() { printf 'locked: %s\n' "$*" >&2; exit 4; }

check_baseline_cache() {
  local md5 want_lines want_md5
  step "Baseline cache $BASELINE_CACHE (append-only)"
  read -r want_lines want_md5 < <(jq -r '.baseline_cache | "\(.lines) \(.md5)"' "$PREREG.json" 2> /dev/null || echo "null null")
  [[ $want_lines =~ ^[0-9]+$ && $want_md5 =~ ^[0-9a-f]{32}$ ]] \
    || fail "$PREREG.json records no baseline_cache {lines, md5} (freeze-baselines --cache writes it)"
  [[ -f $BASELINE_CACHE ]] || fail "$BASELINE_CACHE is missing; the freeze recorded $want_lines lines"
  md5=$(head -n "$want_lines" "$BASELINE_CACHE" | md5sum | cut -d' ' -f1)
  [[ $md5 == "$want_md5" ]] || fail "the first $want_lines lines of $BASELINE_CACHE changed since the freeze"
  ok "baseline cache prefix intact ($want_lines lines at the freeze, $(wc -l < "$BASELINE_CACHE") now)"
}

# The baselines' extra dependencies (pip install -e ".[baselines]").
check_baseline_deps() {
  "$PY" -c 'import benchguard, crowdkit, pysat' 2> /dev/null \
    || die "the baselines' dependencies are missing: .venv/bin/python -m pip install -e \".[baselines]\""
}

# check_ollama URL POPULATION_JSON: the host is reachable, runs the pinned Ollama version,
# and holds every population model under its pinned digest.
check_ollama() {
  local url=$1 population=$2 version tags m want got
  step "Ollama at $url"
  version=$(curl -sf --max-time 10 "$url/api/version" | jq -r .version) \
    || die "no Ollama at $url (is the tunnel up?)"
  if [[ $version == "$OLLAMA_VERSION" ]]; then ok "version $version"
  else warn "version $version, the runs were cached under $OLLAMA_VERSION"; fi
  tags=$(curl -sf --max-time 10 "$url/api/tags") || die "cannot list models at $url"
  for m in $(jq -r '.[][]' <<< "$population"); do
    got=$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .digest' <<< "$tags")
    [[ -n $got ]] || fail "$m is not pulled on $url: ollama pull $m"
    want=${PINNED_DIGEST[$m]:-}
    if [[ -z $want ]]; then warn "$m has no pinned digest ($got)"
    elif [[ $got != "$want" ]]; then fail "$m digest $got differs from the cached runs' $want"
    else ok "$m ${got:0:12}"; fi
  done
}

# host_log WHERE SINCE UNTIL: the Ollama host's log between two epoch times, on stdout.
# WHERE=local reads this machine's journal unless OLLAMA_LOG_CMD or GPU_SSH says otherwise.
host_log() {
  local where=$1 since=$2 until=$3
  if [[ -n ${OLLAMA_LOG_CMD:-} ]]; then
    SINCE=$since UNTIL=$until bash -c "$OLLAMA_LOG_CMD"
  elif [[ -n ${GPU_SSH:-} ]]; then
    ssh -o BatchMode=yes "$GPU_SSH" journalctl -u ollama -o cat --no-pager --since "@$since" --until "@$until"
  elif [[ $where == local ]]; then
    journalctl -u ollama -o cat --no-pager --since "@$since" --until "@$until"
  else
    return 1
  fi
}

# smoke_start FILE RUNS_FILE: the smoke run's first cache line and start time, kept in FILE
# so a smoke run resumed after an interruption is still checked from its true start.
smoke_start() {
  local file=$1 runs=$2
  [[ -f $file ]] || echo "$(wc -l < "$runs" 2> /dev/null || echo 0) $(date +%s)" > "$file"
  cat "$file"
}

# smoke_checks REPORT RUNS_FILE START_FILE STRICT WHERE: the G3 checks of docs/RUNNING.md §11
# on a smoke report and the runs appended since the smoke run started. STRICT=1 fails on a
# criterion (G3); STRICT=0 only warns (the local run is too short for every model to act).
smoke_checks() {
  local report=$1 runs=$2 strict=$4 where=$5 from since timeouts bad=0 log
  read -r from since < "$3"
  new() { tail -n +"$((from + 1))" "$runs"; }

  step "Smoke checks"
  # (1) a call that waited in a queue, cached as model behaviour. Amendment 2 (2026-10-19):
  # a cached ReadTimeout passes only when the host log shows the cancelled request decoding
  # for at least 90% of the timeout (a runaway generation), never a request that waited.
  timeouts=$(new | jq -s '[.[] | .claims[]? | select(startswith("ollama call failed: ReadTimeout"))] | length')
  if [[ $timeouts == 0 ]]; then
    ok "(1) no ReadTimeout cached as behaviour"
  elif ! log=$(host_log "$where" "$since" "$(($(date +%s) + 5))") || [[ -z $log ]]; then
    if [[ $strict == 1 ]]; then
      fail "(1) $timeouts cached ReadTimeouts and no host log to classify them: set GPU_SSH or OLLAMA_LOG_CMD and re-run ./run_all.sh g3"
    else warn "(1) $timeouts cached ReadTimeouts; no Ollama log to classify them"; fi
  elif "$PY" -m specdx.eval.ollama_log --timeouts "$timeouts" <<< "$log" | sed 's/^/        /'; then
    ok "(1) $timeouts cached ReadTimeouts, each a runaway generation in the host log (amendment 2)"
  elif [[ $strict == 1 ]]; then
    fail "(1) a cached ReadTimeout is not a runaway generation: a call waited and was cached as behaviour. Do not re-run; report it"
  else
    warn "(1) a cached ReadTimeout is not a runaway generation"
  fi
  # (2) runs and error runs per model (report).
  echo "  (2) runs per model:"
  new | jq -s -r 'group_by(.interpreter.model)[]
    | "        \(.[0].interpreter.model)\t\(length) runs\t\([.[] | select(.status == "error")] | length) errors"'
  # (3) echoed calls (report, no threshold).
  echo "  (3) $(new | jq -s -c '{parsed_from_text: ([.[].harness[]?.parsed_from_text] | add // 0),
    runs_with_duplicate_evt: ([.[] | (.state.tables.evt // []) | select(length != (unique | length))] | length)}')"

  # θ_m ≥ 0.5 for every population model, and at least one reading each (the 2026-10-19 G3
  # amendment: a model that never acts keeps θ = σ(μ_α) = 0.731 and would pass vacuously).
  echo "  θ and readings per population model:"
  while IFS=$'\t' read -r m theta n; do
    local verdict=ok
    if [[ $n == 0 ]]; then verdict="NO READINGS"
    elif jq -en --argjson t "$theta" '$t < 0.5' > /dev/null; then verdict="BELOW 0.5"; fi
    printf '        %s\tθ %s\t%s readings\t%s\n' "$m" "$theta" "$n" "$verdict"
    [[ $verdict == ok ]] || bad=1
  done < <(jq -r --slurpfile g "$GATE_CONFIG_FOR_CHECKS" '. as $r | $g[0].models_by_family[][] as $m
      | ([$r.cells[].readings[$m] // 0] | add // 0) as $n
      | "\($m)\t\($r.theta[$m] // "missing")\t\($n)"' "$report")
  echo "  EM: $(jq -c '{converged, iterations, fit_start}' "$report")  (reported, never a stop)"
  echo "  runs $(jq .runs "$report"), executed $(jq .executed "$report"), defects $(jq '.defects | length' "$report")"

  if [[ $bad == 1 ]]; then
    if [[ $strict == 1 ]]; then fail "G3: a population model is below 0.5 or has no readings; P2 stops before the gate"
    else warn "a model has no readings or θ < 0.5 (expected in a short local run)"; fi
  else
    ok "every population model has readings and θ ≥ 0.5"
  fi
}

# ---------------------------------------------------------------------------- local
stage_local() {
  local smoke=${SMOKE_DIR:-$(mktemp -d -t p2b-smoke.XXXXXX)} budget=${LOCAL_BUDGET:-8}
  local url=${SPECDX_OLLAMA_URL:-http://localhost:11434} population
  mkdir -p "$smoke"
  echo "Local smoke tests; scratch directory $smoke (nothing is written to work/, results/ or experiments/)"

  check_tools
  check_baseline_deps
  check_database
  gate_running && die "a gate is running (work/p2-gate.pid): local smoke would contend for the database"
  check_cache

  step "Unit tests and lint"
  "$PY" -m pytest -q -o addopts="" | tail -1
  .venv/bin/ruff check . > /dev/null && ok "ruff clean"

  step "G2 preview: cached P1b screen under the 0.3 act rate (scratch copy)"
  local rc=0
  "$SPECDX" screen --from results/p1b-screen.json --out "$smoke/p2-screen" > /dev/null || rc=$?
  [[ $rc == 3 ]] && fail "G2 would STOP: fewer than 2 families or 4 acting models"
  [[ $rc == 0 ]] || die "screen --from exited $rc"
  population=$(jq -c .population "$smoke/p2-screen.json")
  ok "population $population"

  check_ollama "$url" "$population"

  step "Doctor"
  SPECDX_OLLAMA_URL=$url "$SPECDX" doctor --profile gpu48 | grep -E '^\| (postgresql|vendor|pdftotext|disk|vram)' || true

  if [[ $budget == 0 ]]; then
    warn "LOCAL_BUDGET=0: end-to-end smoke run skipped"
  else
    step "End-to-end smoke run: base specification, budget $budget, 1 worker, own cache"
    mkdir -p "$smoke/work"
    cp -r "$CACHE_DIR/tell" "$smoke/work/" 2> /dev/null || true   # Tell chunks; read only
    jq --arg screen "$smoke/p2-screen.json" --argjson pop "$population" --argjson b "$budget" \
      '.screen = $screen | .models_by_family = $pop | .budget = $b | .variants = [0]
       | .seeds = [1] | .probe_budget = 4 | .localize_budget = 0' \
      "$GATE_CONFIG" > "$smoke/smoke.json"
    local t0 t1 executed
    smoke_start "$smoke/smoke.start" "$smoke/work/runs.jsonl" > /dev/null
    t0=$(date +%s)
    SPECDX_WORK="$smoke/work" SPECDX_OLLAMA_URL=$url "$SPECDX" run --variant base \
      --config "$smoke/smoke.json" --out "$smoke/smoke-base.json" --workers 1 \
      2>> "$smoke/smoke.log" || { tail -20 "$smoke/smoke.log"; fail "the smoke run crashed (log $smoke/smoke.log)"; }
    t1=$(date +%s)
    executed=$(jq .executed "$smoke/smoke-base.json")
    ok "$executed runs executed in $((t1 - t0)) s$( ((executed > 0)) && echo ", $(((t1 - t0) / executed)) s per run on this GPU")"
    GATE_CONFIG_FOR_CHECKS="$smoke/smoke.json" smoke_checks "$smoke/smoke-base.json" "$smoke/work/runs.jsonl" "$smoke/smoke.start" 0 local

    step "Baselines on the smoke base (one model, own caches)"
    SPECDX_WORK="$smoke/work" SPECDX_OLLAMA_URL=$url "$SPECDX" baselines --split base \
      --config "$smoke/smoke.json" --out "$smoke/baselines-base.json" \
      --models "$(jq -r '[.[][]][0]' <<< "$population")" 2>> "$smoke/smoke.log" > /dev/null \
      || { tail -20 "$smoke/smoke.log"; fail "the baseline smoke run crashed (log $smoke/smoke.log)"; }
    jq -r '.failures | to_entries[] | select(.value | length > 0)
      | "  \(.key) failed on: \(.value | keys | join(", "))"' "$smoke/baselines-base.json"
    ok "baselines ran on the base: $(jq '.defects | length' "$smoke/baselines-base.json") arms"
  fi

  check_cache
  echo
  echo "Local smoke tests passed. Records in $smoke. Next, on the 48 GB host:"
  echo "  SPECDX_OLLAMA_URL=http://localhost:11435 ./run_all.sh all"
}

# ---------------------------------------------------------------------------- 48 GB steps
# Set by stage_gate when the operator answers no; `all` then stops with exit 0. A flag, not
# a return status: a function called as `f || ...` runs with `set -e` suspended.
GATE_DECLINED=0

need_remote() {
  [[ -n ${SPECDX_OLLAMA_URL:-} ]] || die "set SPECDX_OLLAMA_URL to the tunnel, e.g. http://localhost:11435"
}

population_frozen() {
  jq -e '.models_by_family | length > 0' "$GATE_CONFIG" > /dev/null
}

stage_g2() {
  step "G2: re-admit the cached P1b screen under the 0.3 act rate"
  if population_frozen && [[ -f results/p2-screen.json ]]; then
    ok "already frozen: $(jq -c .models_by_family "$GATE_CONFIG")"
    return
  fi
  [[ -z $(git status --porcelain -- results/p2-screen.json results/p2-screen.md "$GATE_CONFIG") ]] \
    || die "results/p2-screen.* or $GATE_CONFIG has uncommitted changes"
  local rc=0 tmp
  "$SPECDX" screen --from results/p1b-screen.json --out results/p2-screen || rc=$?
  if [[ $rc == 3 ]]; then
    echo "G2 STOP (spec v3.1 §7): fewer than 2 families or fewer than 4 acting models. P2 stops; report it."
    exit 3
  fi
  [[ $rc == 0 ]] || die "screen --from exited $rc"
  tmp=$(mktemp)
  jq --slurpfile s results/p2-screen.json '.models_by_family = $s[0].population' "$GATE_CONFIG" > "$tmp"
  mv "$tmp" "$GATE_CONFIG"
  "$PY" -m pytest -q -o addopts="" tests/test_preregistration_v3.py | tail -1
  git add results/p2-screen.json results/p2-screen.md "$GATE_CONFIG"
  git commit -q -m "results: G2 — the cached P1b screen under the 0.3 act rate; population frozen into the gate config" \
    -- results/p2-screen.json results/p2-screen.md "$GATE_CONFIG"
  ok "population $(jq -c .models_by_family "$GATE_CONFIG"); committed $(git rev-parse --short HEAD)"
}

stage_preflight() {
  need_remote
  check_tools
  check_database
  check_cache
  population_frozen || die "the population is not frozen yet: run ./run_all.sh g2"
  check_ollama "$SPECDX_OLLAMA_URL" "$(jq -c .models_by_family "$GATE_CONFIG")"
  step "Workers"
  [[ $WORKERS -le 2 ]] || warn "WORKERS=$WORKERS: it must not exceed the host's OLLAMA_NUM_PARALLEL"
  ok "--workers $WORKERS (the host must have OLLAMA_NUM_PARALLEL ≥ $WORKERS)"
}

stage_g3() {
  need_remote
  population_frozen || die "the population is not frozen yet: run ./run_all.sh g2"
  gate_running && die "a gate is running (work/p2-gate.pid)"
  step "G3: base specification at budget 60 (hours)"
  local smoke_cfg=work/p2-smoke.json t0 t1 executed
  [[ -n ${GPU_SSH:-}${OLLAMA_LOG_CMD:-} ]] \
    || warn "neither GPU_SSH nor OLLAMA_LOG_CMD is set: a cached ReadTimeout would fail check (1) unclassified"
  jq '.budget = 60' "$GATE_CONFIG" > "$smoke_cfg"
  echo "  started (cache line, epoch): $(smoke_start work/p2-smoke.start "$RUNS")"
  t0=$(date +%s)
  SPECDX_WORK="$CACHE_DIR" "$SPECDX" run --variant base --config "$smoke_cfg" \
    --out results/p2-smoke-base.json --workers "$WORKERS" 2>> work/p2-smoke.log \
    || { tail -20 work/p2-smoke.log; die "the smoke run crashed; if the tunnel dropped, re-run ./run_all.sh g3 (finished runs come from the cache)"; }
  t1=$(date +%s)
  executed=$(jq .executed results/p2-smoke-base.json)
  echo "$((t1 - t0)) $executed" > work/p2-smoke.timing
  ok "$executed runs executed in $((t1 - t0)) s"
  GATE_CONFIG_FOR_CHECKS="$GATE_CONFIG" smoke_checks results/p2-smoke-base.json "$RUNS" work/p2-smoke.start 1 remote
  ok "G3 passed"
}

estimate_gate() {
  [[ -f work/p2-smoke.timing ]] || { warn "no G3 timing (work/p2-smoke.timing)"; return; }
  local secs executed
  read -r secs executed < work/p2-smoke.timing
  if ((executed > 0)); then
    echo "  G3 measured $((secs / executed)) s of wall time per executed run ($executed runs, $secs s, --workers $WORKERS)."
    grep -oE ' ok [0-9.]+ s$' work/p2-smoke.log 2> /dev/null | awk '{ s += $2; n++ }
      END { if (n) printf "  Each run took %.0f s on average on the host (%d runs logged).\n", s / n, n }' || true
    echo "  The gate: base + 20 development mutants at budget $(jq .budget "$GATE_CONFIG") with probes;"
    echo "  cached runs are free, so the first mutants are the slowest."
  else
    echo "  G3 executed no new run (all cached); no timing measured."
  fi
}

stage_gate() {
  local yes=${1:-0}
  need_remote
  population_frozen || die "the population is not frozen yet: run ./run_all.sh g2"
  [[ -f results/p2-smoke-base.json ]] || die "G3 has not run: ./run_all.sh g3"
  if gate_running; then
    ok "the gate is already running (pid $(cat work/p2-gate.pid)); ./run_all.sh status"
    return
  fi
  step "Gate v3 (days)"
  estimate_gate
  if [[ $yes != 1 ]]; then
    read -r -p "  Launch the gate now? [y/N] " answer
    if [[ $answer != [yY]* ]]; then
      echo "  Not launched. Re-run ./run_all.sh gate (or all) when ready."
      GATE_DECLINED=1
      return
    fi
  fi
  launch p2-gate env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" gate --config "$GATE_CONFIG" --out results/p2-gate.json --workers "$WORKERS"
  [[ $FOREGROUND == 1 ]] && return
  echo "  Pause: kill -STOP <pid>; resume: kill -CONT <pid>. If it stops (e.g. the tunnel drops),"
  echo "  run ./run_all.sh gate again: finished runs come from the cache (docs/RUNNING.md §7)."
}

# ---------------------------------------------------------------------------- P3 comparison
stage_baselines_dev() {
  need_remote
  [[ -f results/p2-gate.json ]] || die "the gate has not finished (results/p2-gate.json): ./run_all.sh gate"
  running p2-gate && die "the gate is still running"
  running p3-baselines-dev && { ok "baselines-dev is already running"; return; }
  check_baseline_deps
  step "Baselines on the dev split (every arm, every population model)"
  launch p3-baselines-dev env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" baselines --split dev --config "$GATE_CONFIG" --out results/p3-baselines-dev.json \
    --workers "$WORKERS"
}

stage_freeze() {
  step "Freeze the baselines' arms and thresholds (dev split only)"
  [[ -f results/p3-baselines-dev.json ]] || die "run ./run_all.sh baselines-dev first"
  running p3-baselines-dev && die "baselines-dev is still running"
  if git ls-files --error-unmatch "$PREREG.json" > /dev/null 2>&1; then
    ok "already frozen: $PREREG.json"; return
  fi
  [[ -f $PREREG.json ]] \
    && die "$PREREG.json exists but is not committed: check it, then commit it or remove it and re-run"
  "$SPECDX" freeze-baselines --from results/p3-baselines-dev.json --out "$PREREG" \
    --cache "$BASELINE_CACHE" || fail "freeze-baselines failed (a failed dev specification?)"
  local pin
  pin=$("$PY" -m pytest -q -o addopts="" tests/test_preregistration_baselines.py 2>&1) \
    || { tail -20 <<< "$pin"; fail "the pre-registration pin test failed; nothing committed"; }
  tail -1 <<< "$pin"
  git add "$PREREG.json" "$PREREG.md" results/p3-baselines-dev.json
  git commit -q -m "experiments: baseline pre-registration — arms and thresholds frozen on the dev split" \
    -- "$PREREG.json" "$PREREG.md" results/p3-baselines-dev.json
  ok "frozen and committed $(git rev-parse --short HEAD)"
}

stage_study() {
  need_remote
  gate_passed || locked "the gate is NOT PASSED (results/p2-gate.json): the test study stays locked (spec §3.7)"
  git ls-files --error-unmatch "$PREREG.json" > /dev/null 2>&1 \
    || locked "$PREREG.json is not committed: run ./run_all.sh freeze-baselines first"
  [[ -z $(git status --porcelain --untracked-files=no) ]] \
    || locked "the working tree has uncommitted changes: the study runs only on committed files"
  running p3-study && { ok "the study is already running"; return; }
  check_baseline_deps
  check_cache
  check_baseline_cache
  step "Study: SpecDx and the frozen baselines on the test split (days)"
  launch p3-study env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" study --split test --config "$GATE_CONFIG" --baselines "$PREREG.json" \
    --out results/p3-study.json --workers "$WORKERS"
}

stage_report() {
  step "Report"
  [[ -f results/p3-baselines-dev.json ]] || die "nothing to report: run ./run_all.sh baselines-dev"
  local test_args=()
  [[ -f results/p3-study.json ]] && test_args=(--test results/p3-study.json)
  "$SPECDX" report --dev results/p3-baselines-dev.json "${test_args[@]}" --out results/p3-report.md > /dev/null
  ok "results/p3-report.md$( ((${#test_args[@]})) && echo " (dev and test)" || echo " (dev only)")"
}

stage_status() {
  local name
  for name in p2-gate p3-baselines-dev p3-study; do
    step "Status: $name"
    if running "$name"; then ok "running, pid $(cat "work/$name.pid")"
    else warn "not running"; fi
    if [[ -f work/$name.log ]]; then tail -5 "work/$name.log"; fi
  done
  echo
  [[ -f results/p2-gate.md ]] && echo "  result: results/p2-gate.md"
  [[ -f results/p3-report.md ]] && echo "  report: results/p3-report.md"
  echo "  runs in the cache: $(wc -l < "$RUNS")"
  if [[ -f $BASELINE_CACHE ]]; then echo "  baseline LLM calls cached: $(wc -l < "$BASELINE_CACHE")"; fi
}

# ---------------------------------------------------------------------------- main
usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 2; }

case "${1:-}" in
  local)     stage_local ;;
  g2)        check_tools; stage_g2 ;;
  preflight) stage_preflight ;;
  g3)        stage_preflight; stage_g3 ;;
  gate)      stage_gate "$([[ ${2:-} == --yes ]] && echo 1 || echo 0)" ;;
  status)    stage_status ;;
  baselines-dev)    stage_baselines_dev ;;
  freeze-baselines) check_tools; stage_freeze ;;
  study)            stage_study ;;
  report)           stage_report ;;
  s2)               echo "S2 (τ²-bench) needs the τ² adapter, a later P3 step; nothing to run."; exit 2 ;;
  all)
    check_tools
    export FOREGROUND=1
    stage_g2
    stage_preflight
    stage_g3
    stage_gate "$([[ ${2:-} == --yes ]] && echo 1 || echo 0)"
    [[ $GATE_DECLINED == 1 ]] && exit 0
    stage_baselines_dev
    if ! gate_passed; then
      stage_report   # the dev comparison only; nothing is frozen
      locked "the gate is NOT PASSED: the test study stays locked; dev comparison in results/p3-report.md"
    fi
    stage_freeze
    stage_study
    stage_report
    ;;
  *) usage ;;
esac
