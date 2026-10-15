"""specdx: the command line for phases P1 and P1b."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from specdx.config import PDF_DIR, WORK_DIR

WORKERS_HELP = ("parallel workers, one database each (spec v2 §5.1); set OLLAMA_NUM_PARALLEL "
                "to at least this on the Ollama host")


def mutants_path(out: Path) -> Path:
    """Per-mutant gate results, next to --out: results/p1b-gate.json gives
    results/p1b-gate-mutants.jsonl."""
    return out.with_name(f"{out.stem}-mutants.jsonl")


class _Stderr(logging.Handler):
    """Writes to whatever sys.stderr is at the time, so redirection is always honoured."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 - logging must never break a run
            self.handleError(record)


def _log_progress() -> None:
    """One line per executed run, plus Ollama retries, on stderr."""
    logger = logging.getLogger("specdx")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, _Stderr) for h in logger.handlers):
        handler = _Stderr()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def main(argv: Sequence[str] | None = None) -> int:
    _log_progress()
    parser = argparse.ArgumentParser(prog="specdx", description="SpecDx phase P1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-tell", help="extract and cache the Tell corpus from the PDFs")
    mv = sub.add_parser("make-variants", help="paraphrase Tell variant 0 into variants 1..n")
    mv.add_argument("--model", default="qwen3:14b")
    mv.add_argument("--count", type=int, default=2)
    mv.add_argument("--force", action="store_true",
                    help="replace paraphrases that already exist (they are frozen otherwise)")
    mv.add_argument("--only", metavar="KEY[,KEY]",
                    help="remake just these entries (standing or statement keys) at --variant; "
                         "the rest of the file is left as it is")
    mv.add_argument("--variant", type=int, metavar="N",
                    help="with --only: the paraphrase index to remake (1 .. count-1)")
    mv.add_argument("--seed", type=int, metavar="S",
                    help="with --only: sample with seed S instead of the variant index")
    mv.add_argument("--original", action="store_true",
                    help="with --only: keep the variant-0 wording for these entries at --variant "
                         "(no faithful paraphrase exists); no model is called")
    cap = sub.add_parser("capability", help="check native tool calling of local models")
    cap.add_argument("models", nargs="+")
    scr = sub.add_parser("screen", help="run the capability screen and choose the population")
    scr.add_argument("--profile",
                     help="a hardware profile: a name under experiments/profiles, or a path "
                          "(required unless --from)")
    scr.add_argument("--from", dest="cached", type=Path,
                     help="apply the current admission rules to this screen's JSON instead of "
                          "running one (spec v3 §7, G2); no run, database or model is used")
    scr.add_argument("--workers", type=int, default=1, help=WORKERS_HELP)
    scr.add_argument("--out", type=Path, required=True,
                     help="writes OUT.md and OUT.json (exit status 3 when the stop rule holds)")
    doc = sub.add_parser("doctor", help="check that this machine is ready to run P1b")
    doc.add_argument("--profile", help="the profile whose candidate models to check "
                                       "(default: the one local VRAM suggests, else gpu16)")
    sub.add_parser("cache-info", help="cached runs per model and digest, and per specification")
    sim = sub.add_parser("simulate", help="run the simulation study's sweeps (spec v3 §8.1); "
                                          "no GPU, no database, no model calls")
    sim.add_argument("--sweep", default="all",
                     help="a sweep name (delta, models, n, tau_true, defect_rate, error_conc, "
                          "arms, regime) or all")
    sim.add_argument("--replicates", type=int,
                     help="replicates per point (default: §8.1's minimum, 10, or 5 on the "
                          "regime map; fewer is labelled exploratory)")
    sim.add_argument("--config", type=Path,
                     help="the base SimConfig as JSON (default: SimConfig's defaults)")
    sim.add_argument("--params", type=Path,
                     help="V3Params as JSON (default: V3Params' defaults)")
    sim.add_argument("--out", type=Path, required=True, help="writes OUT.json and OUT.md")
    sub.add_parser("mutants", help="write every mutant and list its ground truth")
    run = sub.add_parser("run", help="run SpecDx on the base specification or one mutant")
    run.add_argument("--variant", default="base")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1, help=WORKERS_HELP)
    gate = sub.add_parser("gate", help="run the P1 go/no-go gate on the development mutants")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    gate.add_argument("--workers", type=int, default=1, help=WORKERS_HELP)
    args = parser.parse_args(argv)

    if args.command == "build-tell":
        from specdx.spec.tell import pdf_chunks

        chunks = pdf_chunks(PDF_DIR, WORK_DIR / "tell")
        for source, n in sorted(Counter(c.source for c in chunks).items()):
            print(f"{source}: {n} chunks")
        return 0

    if args.command == "make-variants":
        from specdx.interpreters import paraphrase as paraphrases
        from specdx.interpreters.prompt import load_variants, save_variants

        current = load_variants()
        if args.only is not None:
            if args.variant is None:
                print("error: --only needs --variant N, the paraphrase index to remake",
                      file=sys.stderr)
                return 2
            keys = [k.strip() for k in args.only.split(",") if k.strip()]
            try:
                remade = paraphrases.regenerate(current, keys, args.variant, model=args.model,
                                                seed=args.seed, original=args.original)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            save_variants(remade)
            source = "the variant-0 wording" if args.original else args.model
            print(f"remade {', '.join(keys)} at variant {args.variant} with {source}")
            return 0
        if current.count > 1 and not args.force:
            print(
                f"error: {current.count - 1} paraphrase variant(s) already exist and are frozen; "
                "pass --force to replace them (cached runs of the old text are then not reused)",
                file=sys.stderr,
            )
            return 2
        made = paraphrases.make_variants(current, model=args.model, count=args.count)
        save_variants(made)
        print(f"variants: {made.count}")
        return 0

    if args.command == "doctor":
        from specdx.config import DATABASE_URL, OLLAMA_URLS
        from specdx.doctor import run_doctor

        report, healthy = run_doctor(args.profile, database_url=DATABASE_URL, hosts=OLLAMA_URLS,
                                     work_dir=WORK_DIR)
        print(report)
        return 0 if healthy else 1

    if args.command == "cache-info":
        from specdx.interpreters.cache import RunCache, cache_summary

        print(cache_summary(RunCache(WORK_DIR / "runs.jsonl")))
        return 0

    if args.command == "simulate":
        import json

        from specdx.infer.anchored import V3Params
        from specdx.sim.generator import SimConfig
        from specdx.sim.study import (
            SWEEPS,
            config_dict,
            load_config,
            load_params,
            min_replicates,
            run_sweep,
            sweep_markdown,
        )

        if args.replicates is not None and args.replicates < 1:
            print("error: --replicates must be at least 1", file=sys.stderr)
            return 2
        names = list(SWEEPS) if args.sweep == "all" else [args.sweep]
        if any(n not in SWEEPS for n in names):
            print(f"error: unknown sweep {args.sweep!r}; choose from {', '.join(SWEEPS)} or all",
                  file=sys.stderr)
            return 2
        base = load_config(args.config) if args.config else SimConfig()
        params = load_params(args.params) if args.params else V3Params()
        replicates = {n: args.replicates or min_replicates(n) for n in names}
        sweeps = {n: run_sweep(n, base, params, replicates[n]) for n in names}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(
            {"base": config_dict(base), "params": dataclasses.asdict(params),
             "replicates": replicates, "sweeps": sweeps}, indent=2), encoding="utf-8")
        text = sweep_markdown(sweeps, params, replicates)
        args.out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.command == "capability":
        from specdx.eval.pilot import check_model

        print("| model | declares tools | calls a tool | usable |\n|---|---|---|---|")
        for model in args.models:
            c = check_model(model)
            print(f"| {model} | {c.declares_tools} | {c.calls_tool} | {c.usable} |")
        return 0

    if args.command == "screen" and args.cached is not None:
        from specdx.eval.screen import rescreen, screen_markdown

        screen = rescreen(args.cached)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(screen.to_json(), encoding="utf-8")
        text = screen_markdown(screen, f"current rules applied to {args.cached}")
        args.out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)
        return 3 if screen.stop else 0
    if args.command == "screen" and args.profile is None:
        print("error: screen needs --profile, or --from a screen's JSON", file=sys.stderr)
        return 2

    if args.command in ("gate", "run"):
        from specdx.eval.pilot import GateConfig as _GateConfig

        # Refuse an empty or unscreened population before any pack is built; the smoke
        # run is held to it too.
        _GateConfig.load(args.config).check_population()

    from specdx.mutate.operators import all_mutants, dev_split
    from specdx.workbench import Workbench

    bench = Workbench.open(workers=getattr(args, "workers", 1))
    if args.command == "screen":
        from specdx.active.loop import SpecDxConfig
        from specdx.eval.screen import SCREEN_VARIANTS, load_profile, run_screen, screen_markdown

        profile = load_profile(args.profile)
        ctx = bench.loop_context(bench.base, SpecDxConfig(profile.candidates,
                                                          variants=SCREEN_VARIANTS))
        screen = run_screen(bench.runner, bench.base, ctx.lens, ctx.canon, profile.candidates)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(screen.to_json(), encoding="utf-8")
        text = screen_markdown(screen, profile.name)
        args.out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)
        return 3 if screen.stop else 0

    if args.command == "mutants":
        for m in all_mutants(bench.base, bench.pdf, bench.work_dir):
            location = ", ".join(sorted(m.truth.location))
            print(f"{m.id}\t{m.truth.type}\t{m.truth.location_kind}\t{location}")
        return 0

    from specdx.eval.pilot import GateConfig, gate_markdown, run_gate

    config = GateConfig.load(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        from specdx.active.loop import run_specdx

        variant = bench.base
        if args.variant != "base":
            mutants = {m.id: m for m in all_mutants(bench.base, bench.pdf, bench.work_dir)}
            variant = mutants[args.variant].variant
        cfg = config.specdx()
        if args.variant == "base":  # the gate's base run does not localise either
            cfg = dataclasses.replace(cfg, localize=False)
        report = run_specdx(variant, bench.loop_context(variant, cfg), cfg)
        args.out.write_text(report.to_json(), encoding="utf-8")
        for d in report.defects[:10]:
            print(f"{d.score:.2f}\t{d.type}\t{', '.join(d.location)}")
        return 0

    mutants = dev_split(all_mutants(bench.base, bench.pdf, bench.work_dir),
                        config.dev_per_type, config.rng_seed)
    result = run_gate(config, bench, mutants, mutants_out=mutants_path(args.out))
    args.out.write_text(result.to_json(), encoding="utf-8")
    args.out.with_suffix(".md").write_text(gate_markdown(result), encoding="utf-8")
    print(gate_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
