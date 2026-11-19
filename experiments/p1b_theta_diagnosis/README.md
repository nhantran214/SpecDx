# P1b θ diagnosis scripts (archived)

These are the analysis scripts behind `results/p1b-theta-diagnosis.md`. They were written for one read-only investigation on 2026-10-14. They are archived as run, not maintained: `# ruff: noqa` exempts them from lint.

## Inputs and outputs

- **Reads.** The P1 run cache at `/home/ethan/AILab/SpecDx/work/runs.jsonl`, read-only. `load.py` uses lines up to 6505:
  - the P1 base population;
  - the 60 smoke population runs, lines 6319–6505.
- **Writes.** Only `data.pkl` and `work/`, next to the scripts. Both are git-ignored.
- `common.py` points `SPECDX_WORK` at that local `work/`, so nothing defaults to the real cache.
- No database and no Ollama.

## Run

Run from this directory with the worktree's virtual environment. The first argument is the directory to put on `sys.path`, `.`.

```bash
../../.venv/bin/python load.py                # map runs to views; writes data.pkl
../../.venv/bin/python summ.py . smoke        # classes per item, agreement, bootstrap of θ ordering
../../.venv/bin/python trace.py . smoke       # stage-2 EM trace, iteration by iteration
../../.venv/bin/python summ.py . p1           # the same summary on the P1 base population
```

The other scripts are the report's counterfactuals and drill-downs:

| Script | What it does |
|---|---|
| `normalise.py` | refit θ after cumulative class-key normalisations (the T0–T5 table) |
| `profile.py` | profile log posterior over a common θ |
| `epsplug.py` | ε plug-in comparison |
| `cap.py` | feasibility ceilings |
| `classes.py`, `dump.py`, `values.py`, `fields.py`, `modal.py`, `slices.py` | data drill-downs |
| `peritem.py`, `sens.py`, `weak.py`, `weights.py` | per-item θ, S2 eligibility, weak findings, hitting-set weights |

Each script's argument list is at its top.

The expected smoke values are θ = 0.0967 / 0.0781 and κ = 2.1464 / 1.3210, matching `results/p1b-smoke-base.json`. The bootstrap P(θ_gpt > θ_qwen) is ≈ 0.5; it varies with R and the random state.
