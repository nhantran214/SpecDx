import json
from pathlib import Path

import pytest

from specdx.subjects.sopbench import SOPBENCH_COMMIT
from specdx.subjects.sopbench.trajectories import (
    MODEL_FAMILIES,
    OPEN_WEIGHTS,
    TrajectoryError,
    UnknownModels,
    fetch,
    parse_name,
    select,
)

TAIL = "dep_full-fmt_structured-tool_{tool}-shuffle_False.json"


def path(model, mode="fc", tool="full", domain="bank", user=None):
    usr = f"usr_{user}-" if user else ""
    return f"output/{domain}/{usr}ast_{model}-mode_{mode}-" + TAIL.format(tool=tool)


def names(*specs):
    out = []
    for spec in specs:
        p, size = spec if isinstance(spec, tuple) else (spec, 50_000)
        n = parse_name(p, size)
        assert n is not None, p
        out.append(n)
    return out


def test_parse_name_reads_model_mode_and_tool_and_canonicalises_fireworks():
    p = path("qwen2.5-72b-it-fireworks")
    n = parse_name(p, 123)
    assert n.canonical == "qwen2.5-72b-instruct"
    assert n.model == "qwen2.5-72b-it-fireworks"
    assert n.mode == "fc"
    assert n.tool_list == "full"
    assert n.user is None
    assert n.domain == "bank"
    assert n.size == 123


def test_parse_name_handles_hyphenated_modes_users_and_rejects_others():
    assert parse_name(path("gpt-4o", mode="act-only"), 1).mode == "act-only"
    assert parse_name(path("gpt-4o", mode="react-v"), 1).mode == "react-v"
    assert parse_name(path("gpt-4o", user="gpt-4o"), 1).user == "gpt-4o"
    assert parse_name("output/bank/readme.txt", 1) is None
    assert parse_name("vendor/x.json", 1) is None


def test_select_drops_user_variants_oracle_tools_humans_and_truncated_files():
    sel = select(
        names(
            path("gpt-4o"),
            path("gpt-4o-mini", user="gpt-4o"),
            path("gpt-5", tool="oracle"),
            path("human"),
            (path("gpt-5-mini"), 2),
        )
    )
    assert [n.canonical for n in sel] == ["gpt-4o"]


def test_select_keeps_one_file_per_domain_and_model_preferring_fc_then_non_fireworks():
    sel = select(
        names(
            path("gpt-4o", mode="react"),
            path("gpt-4o", mode="fc"),
            path("qwen2.5-72b-it-fireworks", mode="fc"),
            path("qwen2.5-72b-instruct", mode="fc"),
            path("gpt-4o", mode="fc", domain="dmv"),
        )
    )
    got = {(n.domain, n.canonical): (n.mode, n.model) for n in sel}
    assert got == {
        ("bank", "gpt-4o"): ("fc", "gpt-4o"),
        ("bank", "qwen2.5-72b-instruct"): ("fc", "qwen2.5-72b-instruct"),
        ("dmv", "gpt-4o"): ("fc", "gpt-4o"),
    }


def test_select_names_every_unknown_model():
    with pytest.raises(UnknownModels, match="mystery-1b") as exc:
        select(names(path("gpt-4o"), path("mystery-1b"), path("other-2b")))
    assert "other-2b" in str(exc.value)


def test_select_on_the_real_listing_knows_every_model():
    rows = json.loads(
        (Path(__file__).parent / "fixtures" / "sopbench_tree.json").read_text()
    )
    parsed = [n for r in rows if (n := parse_name(r["path"], r["size"]))]
    assert parsed
    canon = {n.canonical for n in parsed if n.model != "human"}
    assert canon <= set(MODEL_FAMILIES), sorted(canon - set(MODEL_FAMILIES))
    sel = select(parsed)
    assert sel
    assert len({(n.domain, n.canonical) for n in sel}) == len(sel)


def test_open_weights_are_the_gemma_llama_qwen_and_deepseek_entries():
    assert {
        m for m in MODEL_FAMILIES if m.startswith(("gemma", "llama", "qwen", "deepseek-r1"))
    } == OPEN_WEIGHTS
    assert "gpt-4o" not in OPEN_WEIGHTS


def test_fetch_writes_files_and_a_sorted_manifest_with_sha256(tmp_path):
    sel = select(names(path("gpt-4o", domain="dmv"), path("gpt-4o"), path("qwen3.5-4b")))
    seen = []

    def opener(url):
        seen.append(url)
        return b"[]"

    dest, manifest = tmp_path / "out", tmp_path / "manifest.json"
    result = fetch(sel, dest, manifest, opener=opener)
    assert json.loads(manifest.read_text()) == result
    assert result["commit"] == SOPBENCH_COMMIT
    assert result["fetched_at"]
    files = result["files"]
    assert [f["path"] for f in files] == sorted(f["path"] for f in files)
    assert len(files) == 3
    f = next(f for f in files if f["canonical"] == "qwen3.5-4b")
    assert f["family"] == "qwen" and f["open_weights"] is True and f["mode"] == "fc"
    assert f["sha256"] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    assert (dest / "bank" / Path(f["path"]).name).read_bytes() == b"[]"
    assert all(
        u.startswith(f"https://raw.githubusercontent.com/Leezekun/SOPBench/{SOPBENCH_COMMIT}/")
        for u in seen
    )


def test_fetch_skips_files_already_recorded_in_the_manifest(tmp_path):
    sel = select(names(path("gpt-4o")))
    dest, manifest = tmp_path / "out", tmp_path / "manifest.json"
    fetch(sel, dest, manifest, opener=lambda u: b"[]")

    def boom(url):
        raise AssertionError("should not download")

    again = fetch(sel, dest, manifest, opener=boom)
    assert len(again["files"]) == 1


def test_fetch_rejects_a_non_list_file(tmp_path):
    sel = select(names(path("gpt-4o")))
    with pytest.raises(TrajectoryError):
        fetch(sel, tmp_path / "out", tmp_path / "m.json", opener=lambda u: b"{}")
    with pytest.raises(TrajectoryError):
        fetch(sel, tmp_path / "out", tmp_path / "m.json", opener=lambda u: b"not json")
