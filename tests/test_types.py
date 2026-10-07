import json

from specdx.types import Behaviour, Defect, InterpreterId, Scenario, State


def test_deliveries_split_a_receipt_in_two_when_asked() -> None:
    assert Scenario("x", 15, 12, 0).deliveries() == [("GRN-0015", 12)]
    assert Scenario("x", 15, 13, 0, grn_count=2).deliveries() == [
        ("GRN-0015", 7), ("GRN-0016", 6)
    ]


def test_signature_ignores_the_name() -> None:
    assert Scenario("happy", 15, 15, 0).signature() == Scenario("o15", 15, 15, 0).signature()


def test_a_behaviour_survives_a_json_round_trip() -> None:
    b = Behaviour(
        run_key="k",
        interpreter=InterpreterId("qwen3:14b", "qwen", 0, 1),
        scenario=Scenario("grn12", 15, 12, 0),
        state=State(
            {"so": [{"ma_so": "SO-0031"}]}, {"fn_reconcile:PO-0015": "Khớp"}, {"s": "done"}
        ),
        refused=(("s", "write_draft"),),
        claims={"s": "Xong."},
        model_digest="sha256:abc",
    )
    assert Behaviour.from_json(b.to_json()) == b
    assert json.loads(b.to_json())["model_digest"] == "sha256:abc"


def test_a_behaviour_cached_before_model_digests_still_loads() -> None:
    b = Behaviour("k", InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0), State({}))
    raw = json.loads(b.to_json())
    del raw["model_digest"]
    assert Behaviour.from_json(json.dumps(raw)).model_digest == ""


def test_a_defect_key_is_its_type_and_location() -> None:
    d = Defect("hidden", ("golden:so:SO-0031:trang_thai",), ("grn12",), ("k",), 0.9)
    assert d.key == ("hidden", ("golden:so:SO-0031:trang_thai",))
