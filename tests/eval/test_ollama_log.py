"""G3 amendment 2 (2026-10-19): a cached ReadTimeout passes check (1) only when the Ollama
host's log shows the cancelled request was decoding for at least 90% of the timeout."""

from pathlib import Path

from specdx.eval.ollama_log import DECODE_SHARE, cancels, main

RUNAWAY = Path(__file__).parent / "data" / "ollama-runaway.log"  # the real 2026-10-19 excerpt

# A request that waited 175 s for a slot, then decoded briefly before the client gave up.
QUEUED = """\
slot launch_slot_: id  1 | task 42 | processing task, is_child = 0
slot print_timing: id  1 | task 42 | n_decoded =    300, tg =  70.00 t/s
srv          stop: cancel task, id_task = 42
slot      release: id  1 | task 42 | stop processing: n_tokens = 1200, truncated = 0
"""


def test_the_real_runaway_is_a_runaway() -> None:
    [c] = cancels(RUNAWAY.read_text(encoding="utf-8").splitlines())
    assert (c.task, c.decoded, c.runaway) == (7993, 12362, True)
    assert 178 < c.seconds < 179  # 12362 / 69.15


def test_a_request_that_waited_is_not_a_runaway() -> None:
    [c] = cancels(QUEUED.splitlines())
    assert c.task == 42 and not c.runaway and c.seconds < DECODE_SHARE * 180


def test_a_request_cancelled_before_decoding_is_not_a_runaway() -> None:
    [c] = cancels(["srv          stop: cancel task, id_task = 9"])
    assert (c.decoded, c.seconds, c.runaway) == (0, 0.0, False)


def test_a_reused_task_id_does_not_inherit_old_progress() -> None:
    # llama-server restarts number tasks from 0 again: task 7993's progress must not carry
    # over to a new task 7993 that is cancelled before it decodes.
    lines = RUNAWAY.read_text(encoding="utf-8").splitlines()
    lines = [x for x in lines if "cancel" not in x]  # the first 7993 completes normally
    lines += ["slot launch_slot_: id  0 | task 7993 | processing task, is_child = 0",
              "srv          stop: cancel task, id_task = 7993"]
    [c] = cancels(lines)
    assert not c.runaway


def test_journald_prefixes_are_accepted() -> None:
    prefixed = [f"Oct 19 09:48:31 host ollama[3162]: {x}"
                for x in RUNAWAY.read_text(encoding="utf-8").splitlines()]
    assert [c.runaway for c in cancels(prefixed)] == [True]


def test_main_passes_only_when_every_timeout_is_a_runaway(monkeypatch, capsys) -> None:
    import io

    log = RUNAWAY.read_text(encoding="utf-8") + QUEUED
    monkeypatch.setattr("sys.stdin", io.StringIO(log))
    assert main(["--timeouts", "1"]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(log))
    assert main(["--timeouts", "2"]) == 1
    assert "check (1) FAILS" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["--timeouts", "0"]) == 0
