# SpecDx: diagnosing executable business specifications with LLM interpreters

Phase P1 (core on the TonomyX canon). The design is in `docs/superpowers/specs/2026-10-06-specdx-design.md` and the plan in `docs/superpowers/plans/2026-10-06-specdx-p1-core.md`.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
docker compose up -d                       # PostgreSQL on port 55434
bash scripts/verify_vendor.sh              # vendor/tonomyx is byte-identical to 34c1d2c
.venv/bin/python -m pytest                 # unit and DB tests (no model needed)

.venv/bin/specdx build-tell                # cache the Tell corpus
.venv/bin/specdx capability qwen3:14b llama3.1:8b
.venv/bin/specdx make-variants --model qwen3:14b --count 2
.venv/bin/specdx mutants
.venv/bin/specdx gate --config experiments/p1-gate.json --out results/p1-gate.json
```

Rules: `vendor/` is never edited, mutants live in `work/`, nothing is imported from FAA.
