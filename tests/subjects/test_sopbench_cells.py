from specdx.infer.cells import SATISFIED, VIOLATED
from specdx.subjects.sopbench.cells import cell_id, cells_for, prompt_variants, tell_for
from specdx.subjects.sopbench.load import S3Run, S3Task


def _task(index=0, should_succeed=True, goal="borrow", prompt="Do it.\n\n  Then more.  \n"):
    return S3Task("library", index, goal, should_succeed, prompt)


def _run(task=0, model="m1", db=True, con=True, proc=True, ok=True, err=False, prompt="P"):
    return S3Run("library", task, model, "fam", "fc", ok, db, con, proc, err, prompt)


def test_each_task_has_one_outcome_and_three_check_cells_with_the_13d_anchors():
    data = cells_for([_task(0, True), _task(1, False, "renew")], []).data
    assert len(data.cells) == 8
    o0 = data.cells[cell_id("library", 0, "outcome")]
    o1 = data.cells[cell_id("library", 1, "outcome")]
    assert (o0.kind, o0.anchor, o0.group) == ("value", "succeeds", "outcome:borrow")
    assert (o1.anchor, o1.group) == ("refused", "outcome:renew")
    assert o0.clause == o0.location == o0.id == "s3:library:0:outcome" and o0.steps == ()
    for part in ("db_match", "constraints", "process"):
        c = data.cells[f"s3:library:0:{part}"]
        assert (c.kind, c.anchor, c.group, c.clause, c.location) == (
            "constraint", SATISFIED, part, c.id, c.id)


def test_readings_follow_evaluations_and_clean_needs_all_three_checks():
    data = cells_for([_task()], [_run(db=False, ok=False), _run(model="m2")]).data
    by = {(r.run_key, r.cell.rsplit(":", 1)[1]): r for r in data.readings}
    assert len(data.readings) == 8
    r = by[("m1|library|0", "outcome")]
    assert (r.value, r.clean, r.scenario, r.model, r.family, r.variant) == (
        "refused", False, "library:0", "m1", "fam", 0)
    assert by[("m1|library|0", "db_match")].value == VIOLATED
    assert by[("m1|library|0", "process")].value == SATISFIED
    assert not any(x.clean for x in data.readings if x.run_key == "m1|library|0")
    assert all(x.clean for x in data.readings if x.run_key == "m2|library|0")
    assert by[("m2|library|0", "outcome")].value == "succeeds"


def test_a_tool_error_run_gives_no_readings_and_is_counted():
    out = cells_for([_task()], [_run(err=True), _run(model="m2")])
    assert out.excluded_tool_error == 1
    assert {r.model for r in out.data.readings} == {"m2"}
    assert "m1|library|0" not in out.violated


def test_violated_lists_only_the_checks_the_run_failed():
    out = cells_for([_task()], [_run(con=False, proc=False), _run(model="m2")])
    assert out.violated["m1|library|0"] == frozenset(
        {"s3:library:0:constraints", "s3:library:0:process"})
    assert out.violated["m2|library|0"] == frozenset()


def test_tell_has_one_chunk_per_task_and_counts_prompt_disagreements():
    # The first run's prompt is the minority render: T must come from the majority (I-3).
    runs = [_run(0, "c", prompt="other"), _run(0, "a", prompt="A\n\n B \n"),
            _run(0, "b", prompt="A\n\n B \n"), _run(1, "a", prompt="Z")]
    corpus, differ = tell_for("library", runs)
    assert differ == 1
    assert [c.id for c in corpus.chunks] == ["s3:library:0", "s3:library:1"]
    c0 = corpus.chunks[0]
    assert c0.source == "sopbench"
    assert [(s.id, s.text) for s in c0.sentences] == [
        ("s3:library:0:00", "A"), ("s3:library:0:01", "B")]


def test_clean_also_needs_the_outcome_reading_to_equal_its_anchor():
    # Ruling I-6: a refusal of a should-succeed task deviates at an anchored value cell, so
    # it violates that cell's golden clause and is not clean, even with every check met.
    tasks = [_task(0, True), _task(1, False, "renew")]
    runs = [_run(0, "refuser", ok=False), _run(0, "doer", ok=True),
            _run(1, "refuser", ok=False), _run(1, "doer", ok=True)]
    clean = {(r.run_key, r.cell.rsplit(":", 1)[1]): r.clean
             for r in cells_for(tasks, runs).data.readings}
    assert not clean[("refuser|library|0", "outcome")]
    assert not clean[("refuser|library|0", "db_match")]
    assert clean[("doer|library|0", "outcome")]
    assert clean[("refuser|library|1", "process")]
    assert not clean[("doer|library|1", "constraints")]


def test_prompt_variants_are_decided_per_run_by_frequency_within_each_task():
    # Ruling I-3: the most common prompt of a task is variant 0; the others follow by
    # decreasing frequency, ties broken by the string; one model may see several renders.
    runs = [_run(0, "a", prompt="old"), _run(0, "b", prompt="new"),
            _run(0, "c", prompt="new"), _run(0, "d", prompt="y"), _run(0, "e", prompt="x"),
            _run(1, "a", prompt="new"), _run(1, "b", prompt="old"), _run(1, "c", prompt="old")]
    v = prompt_variants(runs)
    assert v == {"a|library|0": 1, "b|library|0": 0, "c|library|0": 0, "d|library|0": 3,
                 "e|library|0": 2, "a|library|1": 1, "b|library|1": 0, "c|library|1": 0}
    data = cells_for([_task(0), _task(1)], runs).data
    assert {(r.run_key, r.variant) for r in data.readings} == set(v.items())


def test_cells_for_takes_variants_decided_over_a_larger_population():
    runs = [_run(0, "a", prompt="old"), _run(0, "b", prompt="new")]
    data = cells_for([_task(0)], runs[:1], variants={"a|library|0": 1}).data
    assert {r.variant for r in data.readings} == {1}
