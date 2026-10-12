"""specdx: the command line for phases P1 and P1b."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from specdx.config import PDF_DIR, WORK_DIR


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
    sub.add_parser("mutants", help="write every mutant and list its ground truth")
    run = sub.add_parser("run", help="run SpecDx on the base specification or one mutant")
    run.add_argument("--variant", default="base")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    gate = sub.add_parser("gate", help="run the P1 go/no-go gate on the development mutants")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
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

    if args.command == "capability":
        from specdx.eval.pilot import check_model

        print("| model | declares tools | calls a tool | usable |\n|---|---|---|---|")
        for model in args.models:
            c = check_model(model)
            print(f"| {model} | {c.declares_tools} | {c.calls_tool} | {c.usable} |")
        return 0

    if args.command == "gate":
        from specdx.eval.pilot import GateConfig as _GateConfig

        # Refuse an empty or unscreened population before any pack is built.
        _GateConfig.load(args.config).check_population()

    from specdx.mutate.operators import all_mutants, dev_split
    from specdx.workbench import Workbench

    bench = Workbench.open()
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
