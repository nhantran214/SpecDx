"""specdx: the command line for phases P1, P1b and the baseline comparison study."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from specdx.config import OLLAMA_URL, PDF_DIR, REPO_ROOT, WORK_DIR

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


# The only configuration and pre-registration the test split runs with (spec §3.7).
STUDY_CONFIG = Path("experiments") / "p2-gate.json"
STUDY_PREREG = Path("experiments") / "p3-baselines-prereg.json"
STUDY_GATE = Path("results") / "p2-gate.json"


def study_paths(root: Path, config: Path, prereg_json: Path) -> str | None:
    """Why --config or --baselines is not the frozen file under root, or None."""
    for given, frozen in ((config, root / STUDY_CONFIG), (prereg_json, root / STUDY_PREREG)):
        if given.resolve() != frozen.resolve():
            return f"the study runs only with the frozen {frozen}, not {given}"
    return None


def study_guard(root: Path, gate_json: Path, prereg_json: Path,
                config: Path | None = None) -> str | None:
    """Why the test split must stay locked (spec §3.7), or None: the gate PASSED, the
    pre-registration (and the configuration, when given) is committed, and the working
    tree has no uncommitted changes, so the committed contents are the ones used."""
    import subprocess

    try:
        passed = json.loads(gate_json.read_text(encoding="utf-8")).get("passed") is True
    except (OSError, ValueError, AttributeError):  # missing, malformed or not an object
        passed = False
    if not passed:
        return f"the gate has not PASSED ({gate_json}); the test split stays locked"

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    for path in (prereg_json,) if config is None else (prereg_json, config):
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return f"{path} is outside {root}, so it cannot be committed there"
        if git("ls-files", "--error-unmatch", rel).returncode != 0:
            hint = (": run `specdx freeze-baselines` (docs/RUNNING.md §13)"
                    if path == prereg_json else "")
            return f"{rel} is not committed{hint}"
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        return "the working tree has uncommitted changes"
    return None


def build_parser() -> argparse.ArgumentParser:
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
    bl = sub.add_parser("baselines", help="run every baseline arm on the dev split (or the base)")
    bl.add_argument("--split", choices=("dev", "base"), required=True)
    bl.add_argument("--config", type=Path, required=True)
    bl.add_argument("--out", type=Path, required=True)
    bl.add_argument("--models", nargs="*", default=None)
    bl.add_argument("--workers", type=int, default=1, help=WORKERS_HELP)
    fr = sub.add_parser("freeze-baselines", help="choose and pin each baseline's arm and τ")
    fr.add_argument("--from", dest="source", type=Path, required=True)
    fr.add_argument("--out", type=Path, required=True)
    fr.add_argument("--cache", type=Path, default=None,
                    help="the baseline LLM cache to fingerprint")
    fr.add_argument("--force", action="store_true",
                    help="replace an existing pre-registration (it is frozen otherwise)")
    fr.add_argument("--gate-config", type=Path, default=None,
                    help=f"the gate configuration to fingerprint (default {STUDY_CONFIG})")
    fr.add_argument("--gate-result", type=Path, default=None,
                    help=f"the gate result to fingerprint (default {STUDY_GATE})")
    st = sub.add_parser("study", help="the test split, once the gate PASSED (spec §3.7)")
    st.add_argument("--split", choices=("test",), required=True)
    st.add_argument("--config", type=Path, required=True)
    st.add_argument("--baselines", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--workers", type=int, default=1, help=WORKERS_HELP)
    rp = sub.add_parser("report", help="tables A and B, statistics, for dev and test")
    rp.add_argument("--dev", type=Path, required=True)
    rp.add_argument("--test", type=Path)
    rp.add_argument("--out", type=Path, required=True)
    p3 = sub.add_parser("p3", help="the P3 experiment stages (reports in results/p3)")
    p3sub = p3.add_subparsers(dest="p3_command", required=True)
    for name, text in (("fetch-sopbench", "download SOPBench's published trajectories"),
                       ("replication", "offline replication of the S3 diagnosis"),
                       ("local", "local tests, lint and a replication smoke"),
                       ("status", "rebuild STATUS.md")):
        sp = p3sub.add_parser(name, help=text)
        sp.add_argument("--results", type=Path,
                        default=Path(os.environ.get("P3_RESULTS", "results/p3")))
        sp.add_argument("--work", type=Path, default=Path(os.environ.get("P3_WORK", "work")))
        if name == "status":
            sp.add_argument("--done", metavar="STAGE", default=None,
                            help="exit 0 if STAGE finished with exit code 0, else 1")
        if name != "status" and name != "local":
            sp.add_argument("--manifest", type=Path,
                            default=Path("experiments/p3-sopbench-trajectories.json"))
        if name == "fetch-sopbench":
            sp.add_argument("--dest", type=Path, default=None,
                            help="default: <work>/sopbench-output")
        if name in ("replication", "local"):
            sp.add_argument("--trajectories", type=Path, default=None,
                            help="default: <work>/sopbench-output")
        if name == "replication":
            sp.add_argument("--domains", default=None, help="comma-separated; default: all")
        if name == "local":
            sp.add_argument("--manifest", type=Path,
                            default=Path("experiments/p3-sopbench-trajectories.json"))
    return parser


def _p3(args: argparse.Namespace) -> int:
    from specdx.report import p3
    from specdx.report.stages import (
        STAGES,
        already_done,
        run_stage,
        stale_commit,
        stale_warning,
        update_status,
    )

    results, work = args.results, args.work
    if args.p3_command == "status":
        names = [args.done] if args.done is not None else [s.name for s in STAGES]
        for name in names:
            if (stale := stale_commit(name, results)) is not None:
                print(stale_warning(name, *stale, results), file=sys.stderr)
        if args.done is not None:
            return 0 if already_done(args.done, results) else 1
        update_status(results, work)
        return 0
    command = "specdx p3 " + args.p3_command
    traj = getattr(args, "trajectories", None) or work / "sopbench-output"
    if args.p3_command == "fetch-sopbench":
        dest = args.dest or work / "sopbench-output"

        def stage():
            return p3.fetch_stage(dest, args.manifest)
    elif args.p3_command == "replication":
        domains = [d.strip() for d in args.domains.split(",") if d.strip()] \
            if args.domains else None

        def stage():
            return p3.replication_stage(manifest=args.manifest, trajectories=traj,
                                        domains=domains)
    else:
        def stage():
            return p3.local_stage(manifest=args.manifest, trajectories=traj)
    return run_stage(args.p3_command, stage, command=command, results=results, work=work)


def main(argv: Sequence[str] | None = None) -> int:
    _log_progress()
    args = build_parser().parse_args(argv)

    if args.command == "p3":
        return _p3(args)

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

    # The commands below need no workbench, so they never take the database lock.
    if args.command == "freeze-baselines":
        from specdx.eval.freeze import canonical_json, prereg, prereg_markdown
        from specdx.eval.study import SplitResult

        if args.out.with_suffix(".json").exists() and not args.force:
            print(f"error: {args.out.with_suffix('.json')} already exists and is frozen; "
                  "pass --force to replace it", file=sys.stderr)
            return 1
        result = SplitResult.from_json(args.source.read_text(encoding="utf-8"))
        bad = {a: f for a, f in result.failures.items() if f}
        if bad:
            print(f"error: failed specifications {bad}; re-run baselines-dev", file=sys.stderr)
            return 1
        gate_config = args.gate_config or REPO_ROOT / STUDY_CONFIG
        gate_result = args.gate_result or REPO_ROOT / STUDY_GATE
        for path in (gate_config, gate_result):
            if not path.exists():
                print(f"error: {path} is missing; the pre-registration records its sha256",
                      file=sys.stderr)
                return 1
        try:
            doc = prereg(result, args.source, args.cache, gate_config, gate_result)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # canonical_json is the serialization prereg_markdown hashes.
        args.out.with_suffix(".json").write_text(canonical_json(doc), encoding="utf-8")
        args.out.with_suffix(".md").write_text(prereg_markdown(doc), encoding="utf-8")
        return 0

    if args.command == "report":
        from specdx.eval.freeze import choose
        from specdx.eval.study import SplitResult
        from specdx.eval.study_report import study_markdown

        for path in (args.dev, args.test):
            if path is not None and not path.exists():
                print(f"error: {path} does not exist", file=sys.stderr)
                return 2
        dev = SplitResult.from_json(args.dev.read_text(encoding="utf-8"))
        chosen = choose(dev)
        text = study_markdown(dev, {"specdx": 0.0, **{a: 0.0 for a in dev.defects
                                                      if a != "specdx"}},
                              in_sample={"specdx": 0.0, **{c["arm"]: c["tau"]
                                                           for c in chosen.values()}})
        if args.test is not None:
            test = SplitResult.from_json(args.test.read_text(encoding="utf-8"))
            taus = test.taus
            if taus is None:  # a result written before the study embedded its τ
                if not (REPO_ROOT / STUDY_PREREG).exists():
                    print(f"error: {args.test} embeds no τ and there is no pre-registration "
                          f"({REPO_ROOT / STUDY_PREREG}) for its test section", file=sys.stderr)
                    return 2
                pre = json.loads((REPO_ROOT / STUDY_PREREG).read_text(encoding="utf-8"))
                taus = {c["arm"]: c["tau"] for c in pre["chosen"].values()}
                warning = (f"warning: {args.test} embeds no τ; the test section uses the "
                           f"pre-registration's ({STUDY_PREREG})")
                print(warning, file=sys.stderr)
                text += f"\n> {warning}\n"
            text += "\n" + study_markdown(test, {"specdx": 0.0, **taus})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.command == "study":  # refused before any configuration, pack or lock (exit 4)
        from specdx.eval.freeze import bound_mismatch

        reason = (study_paths(REPO_ROOT, args.config, args.baselines)
                  or study_guard(REPO_ROOT, REPO_ROOT / STUDY_GATE, args.baselines, args.config))
        if not reason:
            try:
                frozen = json.loads(args.baselines.read_text(encoding="utf-8"))
                reason = bound_mismatch(frozen, REPO_ROOT / STUDY_CONFIG, REPO_ROOT / STUDY_GATE)
            except (OSError, ValueError, TypeError) as exc:
                reason = f"cannot read the pre-registration {args.baselines}: {exc}"
        if reason:
            print(f"error: {reason}", file=sys.stderr)
            return 4

    if args.command in ("gate", "run", "baselines", "study"):
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

    if args.command in ("baselines", "study"):
        from specdx.baselines.llm import Gateway
        from specdx.eval.freeze import study_arms
        from specdx.eval.pilot import GateConfig
        from specdx.eval.study import dev_arms, run_split

        config = GateConfig.load(args.config)  # its population was checked above
        allm = all_mutants(bench.base, bench.pdf, bench.work_dir)
        dev = dev_split(allm, config.dev_per_type, config.rng_seed)
        pinned: dict[str, str] = {}
        for r in bench.cache:  # RunCache iterates Behaviours in file order: the last wins
            if r.model_digest:
                pinned[r.interpreter.model] = r.model_digest
        gateway = Gateway(OLLAMA_URL, bench.work_dir / "baseline-llm.jsonl", pinned)
        if args.command == "baselines":
            arms = dev_arms(config.models_by_family)
            if args.models is not None:
                arms = [a for a in arms if a.model is None or a.model in args.models]
            mutants = dev if args.split == "dev" else []
        else:
            doc = json.loads(args.baselines.read_text(encoding="utf-8"))
            arms = [a for a, _ in study_arms(doc)]
            ids = {m.id for m in dev}
            mutants = [m for m in allm if m.id not in ids]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result = run_split(bench, config, mutants, arms, gateway,
                           args.out.with_suffix(".progress.jsonl"), args.split,
                           announce=lambda line: print(line, flush=True))
        if args.command == "study":  # the test report uses exactly these (spec §9.8)
            import hashlib

            result.prereg_sha256 = hashlib.sha256(args.baselines.read_bytes()).hexdigest()
            result.taus = {a.label: tau for a, tau in study_arms(doc)}
        args.out.write_text(result.to_json(), encoding="utf-8")
        print(f"{len(result.defects)} arms on {len(mutants) + 1} specifications -> {args.out}")
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
