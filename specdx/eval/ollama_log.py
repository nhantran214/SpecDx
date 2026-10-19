"""Which cancelled Ollama requests were the model generating, not waiting (pre-registration
§5, G3 amendment 2 of 2026-10-19).

P1b smoke check (1) stops P2 when a run holds a cached "ollama call failed: ReadTimeout",
because a call that waited in a queue would be cached as model behaviour. A model can also
exhaust the timeout on its own, by generating without end. On 2026-10-19, qwen2.5:14b
decoded 12,362 tokens at about 69 tokens/s on one local worker, where nothing could queue.
The run record cannot tell the two apart, but the Ollama host's log can.

A client that times out disconnects, and llama-server logs `cancel task, id_task = N`.
Before that, a request that was decoding logs `n_decoded = K, tg = R t/s` for task N every
few seconds. K / R is the time the request spent decoding. A cancelled request is a runaway
when it spent at least DECODE_SHARE of the client timeout decoding. A request that waited
for a slot or a model load decoded little or nothing, and is not a runaway.

Check (1) passes when every cached ReadTimeout of the smoke run is matched by a runaway in
the host log over the same window: timeouts ≤ runaways."""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from specdx.interpreters.executor import TIMEOUT

DECODE_SHARE = 0.9

_LAUNCH = re.compile(r"launch_slot_: .*\| task (\d+) \|")
_DECODED = re.compile(r"\| task (\d+) \| n_decoded =\s*(\d+), tg =\s*([\d.]+) t/s")
_CANCEL = re.compile(r"cancel task, id_task = (\d+)")
_RELEASE = re.compile(r"release: .*\| task (\d+) \|")


@dataclass(frozen=True)
class Cancel:
    """A request the client abandoned: the tokens it decoded, the seconds spent decoding,
    and whether that was at least DECODE_SHARE of the timeout."""

    task: int
    decoded: int
    seconds: float
    runaway: bool


def cancels(lines: Iterable[str], timeout: float = TIMEOUT) -> list[Cancel]:
    """Every cancelled task in a log, in order. Task ids restart when llama-server restarts,
    so a task's progress is forgotten when its slot launches or is released."""
    progress: dict[int, tuple[int, float]] = {}
    out: list[Cancel] = []
    for line in lines:
        if m := _DECODED.search(line):
            progress[int(m[1])] = (int(m[2]), float(m[3]))
        elif m := _CANCEL.search(line):
            task = int(m[1])
            decoded, rate = progress.pop(task, (0, 0.0))
            seconds = decoded / rate if rate > 0 else 0.0
            out.append(Cancel(task, decoded, seconds, seconds >= DECODE_SHARE * timeout))
        elif m := _LAUNCH.search(line) or _RELEASE.search(line):
            progress.pop(int(m[1]), None)
    return out


def main(argv: list[str] | None = None) -> int:
    """Read an Ollama host's log on stdin. Exit 0 when the cached ReadTimeouts are all
    matched by runaways (check (1) passes), 1 otherwise."""
    parser = argparse.ArgumentParser(prog="python -m specdx.eval.ollama_log",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--timeouts", type=int, required=True,
                        help="cached ReadTimeouts in the smoke run's new runs")
    args = parser.parse_args(argv)
    # Logs carry prompt fragments cut mid-character; a bad byte must not stop the check.
    raw = getattr(sys.stdin, "buffer", None)
    found = cancels(io.TextIOWrapper(raw, errors="replace") if raw else sys.stdin)
    for c in found:
        kind = "runaway" if c.runaway else "not a runaway"
        print(f"task {c.task}\t{c.decoded} tokens\t{c.seconds:.0f} s decoding\t{kind}")
    runaways = sum(c.runaway for c in found)
    passed = args.timeouts <= runaways
    print(f"{args.timeouts} cached ReadTimeouts, {runaways} runaways of {len(found)} cancelled "
          f"requests (decoding ≥ {DECODE_SHARE:.0%} of {TIMEOUT:.0f} s): "
          + ("check (1) passes" if passed else "check (1) FAILS"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
