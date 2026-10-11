import math
import random

from specdx.infer.act import Act, act_rate, drafting_steps, fit_rasch, is_act, slice_rows
from specdx.orchestrate.work_orders import load_work_orders

STEPS = load_work_orders()


def test_the_steps_that_draft_are_the_de_steps_with_draft_tables(pack) -> None:
    assert drafting_steps(pack, STEPS) == {
        "DE-SM-02:draft_quote", "DE-FN-04:draft_purchase_order",
        "DE-OP-02:record_goods_receipt", "DE-FN-01:draft_sales_invoice",
        "DE-FN-02:reconcile", "DE-FN-03:draft_payment",
    }


def test_acting_needs_a_completed_step_and_a_new_row_when_the_step_drafts() -> None:
    before = slice_rows((("so", ()),))
    now = slice_rows((("so", (("SO-0031", (("trang_thai", "Nháp"),)),)),))
    assert is_act("done", True, now, before)
    assert is_act("needs_gate", True, now, before)
    assert not is_act("done", True, before, before)  # completed, but wrote nothing
    assert not is_act("error", True, now, before)
    assert is_act("done", False, before, before)  # a step that drafts nothing only completes
    assert not is_act("", False, before, before)


def _synthetic(kappa, delta, trials, seed):
    rng = random.Random(seed)
    acts = []
    for m, k in kappa.items():
        for s, d in delta.items():
            p = 1 / (1 + math.exp(-(k - d)))
            acts += [Act(s, f"{m}|{s}|{t}", m, rng.random() < p) for t in range(trials)]
    return acts


def test_rasch_recovers_planted_capability_and_difficulty() -> None:
    kappa = {"weak": -1.0, "mid": 0.0, "strong": 1.5}
    raw = [-1.5 + 3.0 * j / 29 for j in range(30)]
    delta = {f"s{j:02d}": d - sum(raw) / len(raw) for j, d in enumerate(raw)}
    fit = fit_rasch(_synthetic(kappa, delta, 40, 0))
    assert abs(sum(fit.delta.values())) < 1e-9
    assert all(abs(fit.kappa[m] - k) < 0.3 for m, k in kappa.items())
    pairs = [(fit.delta[s], d) for s, d in delta.items()]
    mx = sum(a for a, _ in pairs) / len(pairs)
    my = sum(b for _, b in pairs) / len(pairs)
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs))
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    assert cov / (sx * sy) > 0.95


def test_an_item_is_observable_only_with_enough_probability_and_enough_acts() -> None:
    acts = []
    for m in ("a", "b", "c"):
        for t in range(6):
            acts.append(Act("easy", f"{m}|easy|{t}", m, True))
            acts.append(Act("rare", f"{m}|rare|{t}", m, m == "a" and t < 2))
            acts.append(Act("never", f"{m}|never|{t}", m, False))
    fit = fit_rasch(acts)
    assert fit.observable("easy")
    assert fit.acted["rare"] == 2 and not fit.observable("rare")  # fewer than m_min = 3
    assert fit.observability["never"] < 0.3 and not fit.observable("never")
    assert fit.observable_items() == {"easy"}
    assert fit.kappa["a"] > fit.kappa["b"]


def test_a_model_that_never_acts_gets_the_lowest_capability() -> None:
    acts = [Act(f"s{j}", f"{m}|{j}|{t}", m, m != "silent" and (t + j) % 4 != 0)
            for m in ("x", "y", "silent") for j in range(8) for t in range(4)]
    fit = fit_rasch(acts)
    assert fit.kappa["silent"] < min(fit.kappa["x"], fit.kappa["y"])
    assert act_rate(acts)["silent"] == 0.0 and act_rate(acts)["x"] == 0.75


def test_no_acts_give_an_empty_fit() -> None:
    fit = fit_rasch([])
    assert fit.kappa == {} and fit.observable_items() == frozenset()
