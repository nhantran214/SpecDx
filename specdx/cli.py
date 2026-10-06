"""specdx: the command line for phase P1."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from specdx.config import PDF_DIR, WORK_DIR


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="specdx", description="SpecDx phase P1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-tell", help="extract and cache the Tell corpus from the PDFs")
    mv = sub.add_parser("make-variants", help="paraphrase Tell variant 0 into variants 1..n")
    mv.add_argument("--model", default="qwen3:14b")
    mv.add_argument("--count", type=int, default=2)
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
        from specdx.interpreters.paraphrase import make_variants
        from specdx.interpreters.prompt import load_variants, save_variants

        made = make_variants(load_variants(), model=args.model, count=args.count)
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
    result = run_gate(config, bench, mutants)
    args.out.write_text(result.to_json(), encoding="utf-8")
    args.out.with_suffix(".md").write_text(gate_markdown(result), encoding="utf-8")
    print(gate_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
