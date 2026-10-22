#!/usr/bin/env python3
"""Merge the theme agents' raw records into the related-work catalogue.

  python3 build.py [--pdfs] [--tiers]

Reads raw/*.jsonl and writes:
  references.bib    one verified entry per paper, keys rewritten to <author><year><word>
  catalog.csv       one row per paper (machine-readable)
  catalog.md        the catalogue grouped by theme, then relevance and year
  pdfs/<theme>/     open-access PDFs (with --pdfs)
  tools/venues.json venue tiers from the CORE portal (with --tiers; reused afterwards)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
VENUES = ROOT / "tools" / "venues.json"
lit.SPACING["portal.core.edu.au"] = 1.0
lit.SPACING["arxiv.org"] = 3.1

THEMES = {
    "t1_truth_inference": "Truth inference, label aggregation and latent-class models",
    "t2_irt_benchmark_audit": "IRT, benchmark auditing, label errors and benchmark repair",
    "t3_requirements_nlp": "Requirements quality: ambiguity, incompleteness, inconsistency (NLP4RE/LLM4RE)",
    "t4_spec_defects_oracles": "Defects in specifications and test oracles; specification mutation",
    "t5_diagnosis_mus_dd": "Model-based diagnosis, MUS/MCS, fault localisation and delta debugging",
    "t6_llm_disagreement_ambiguity": "LLM disagreement, uncertainty and ambiguity; differential and N-version testing",
    "t7_active_testing_design": "Active testing, experimental design, bandits and scenario generation",
    "t8_bpm_agents_policies": "Business-process agents and policies, conformance checking, rule verification",
    "t9_agent_failure_attribution": "Agent failure attribution, evaluation reliability and specification gaming",
}
REL_ORDER = {"high": 0, "medium": 1, "low": 2}
CONF_KINDS = {"conference", "workshop"}
STOP = {"a", "an", "the", "on", "of", "for", "and", "in", "to", "with", "via", "from",
        "towards", "toward", "is", "are", "do", "does", "can", "how", "what", "when", "why", "we"}


def norm_title(t: str) -> str:
    return lit._norm(t or "")


def load() -> list[dict]:
    recs = []
    for path in sorted(RAW.glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip {path.name}:{n}: {exc}", file=sys.stderr)
                continue
            r.setdefault("theme", path.stem)
            r["_src"] = f"{path.name}:{n}"
            recs.append(r)
    return recs


def merge(recs: list[dict]) -> list[dict]:
    """Union-find over shared DOI, arXiv id or normalised title."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, r in enumerate(recs):
        keys = []
        if r.get("doi"):
            keys.append("doi:" + r["doi"].lower().strip())
        if r.get("arxiv_id"):
            keys.append("ax:" + re.sub(r"v\d+$", "", str(r["arxiv_id"]).strip()))
        nt = norm_title(r.get("title"))
        if len(nt) > 15:
            keys.append("t:" + nt)
        for k in keys:
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, r in enumerate(recs):
        groups[find(i)].append(r)

    merged = []
    for grp in groups.values():
        # Canonical record: peer-reviewed beats preprint, then the most complete one.
        def score(r):
            pr = 0 if str(r.get("status", "")).startswith("peer") else 1
            return (pr, -sum(1 for v in r.values() if v))
        grp.sort(key=score)
        base = dict(grp[0])
        base["themes"] = []
        for r in grp:
            for field in ("pdf_url", "code_url", "arxiv_id", "doi", "citations", "code_info"):
                if not base.get(field) and r.get(field):
                    base[field] = r[field]
            if r["theme"] not in base["themes"]:
                base["themes"].append(r["theme"])
        base["theme"] = base["themes"][0]
        base["relevance"] = min((r.get("relevance") or "low" for r in grp),
                                key=lambda x: REL_ORDER.get(x, 3))
        for field in ("subtopics", "specdx_components"):
            vals = []
            for r in grp:
                for v in r.get(field) or []:
                    if v not in vals:
                        vals.append(v)
            base[field] = vals
        roles = []
        for r in grp:
            if r.get("role") and r["role"] not in roles:
                roles.append(r["role"])
        base["role"] = "; ".join(roles)
        base["baseline_candidate"] = any(r.get("baseline_candidate") for r in grp)
        notes = [r.get("baseline_notes") for r in grp if r.get("baseline_notes")]
        base["baseline_notes"] = " | ".join(dict.fromkeys(notes)) or None
        rel = [r.get("relation_to_specdx") for r in grp if r.get("relation_to_specdx")]
        base["relation_to_specdx"] = " | ".join(dict.fromkeys(rel)) or None
        base["_srcs"] = [r["_src"] for r in grp]
        merged.append(base)
    return merged


def make_keys(recs: list[dict]) -> None:
    used: set[str] = set()
    for r in sorted(recs, key=lambda r: (r.get("year") or 0, norm_title(r.get("title")))):
        authors = r.get("authors") or ["anon"]
        last = lit._norm(authors[0]).split()
        last = last[-1] if last else "anon"
        word = next((w for w in norm_title(r.get("title")).split() if w not in STOP and len(w) > 2),
                    "paper")
        key = f"{last}{r.get('year') or 'nd'}{word}"
        k, n = key, 1
        while k in used:
            n += 1
            k = f"{key}{chr(ord('a') + n - 2)}"
        used.add(k)
        r["key"] = k


def bib_escape(s: str) -> str:
    return (s or "").replace("&", r"\&").replace("%", r"\%")


def bibtex_for(r: dict) -> str:
    bib = (r.get("bibtex") or "").strip()
    if bib.startswith("@") and "{" in bib:
        bib = re.sub(r"^@(\w+)\s*\{\s*[^,]*,", lambda m: f"@{m.group(1)}{{{r['key']},", bib, count=1)
        # CrossRef writes month=june / sept; BibTeX only defines jan..dec.
        return re.sub(r"(month\s*=\s*)([A-Za-z]+)\b",
                      lambda m: m.group(1) + m.group(2)[:3].lower(), bib)
    kind = r.get("venue_kind")
    authors = " and ".join(r.get("authors") or [])
    fields = {"title": "{" + bib_escape(r.get("title")) + "}", "author": authors,
              "year": str(r.get("year") or "")}
    if kind == "journal":
        etype, fields["journal"] = "article", bib_escape(r.get("venue"))
    elif kind in CONF_KINDS:
        etype, fields["booktitle"] = "inproceedings", bib_escape(r.get("venue"))
    else:
        etype = "misc"
        if r.get("arxiv_id"):
            fields["eprint"], fields["archivePrefix"] = r["arxiv_id"], "arXiv"
        fields["note"] = r.get("status") or ""
    if r.get("doi"):
        fields["doi"] = r["doi"]
    if r.get("url"):
        fields["url"] = r["url"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{etype}{{{r['key']},\n{body}\n}}"


# ------------------------------------------------------------- venue tiers

def _core_rows(url: str) -> list[list[str]]:
    raw = lit.http_get(url)
    if not raw:
        return []
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", raw, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if cells:
            rows.append(cells)
    return rows


def core_conf(acronym: str) -> str | None:
    url = ("https://portal.core.edu.au/conf-ranks/?" + urllib.parse.urlencode(
        {"search": acronym, "by": "acronym", "source": "CORE2023", "sort": "atitle", "page": 1}))
    for cells in _core_rows(url):
        if len(cells) >= 4 and cells[1].lower() == acronym.lower():
            return f"CORE2023 {cells[3]}"
    return None


def core_journal(title: str) -> str | None:
    clean = re.sub(r"\s*\(.*?\)\s*", " ", title).strip()
    clean = re.sub(r"^(The|IEEE|ACM)\s+", "", clean)
    url = ("https://portal.core.edu.au/jnl-ranks/?" + urllib.parse.urlencode(
        {"search": " ".join(clean.split()[:3]), "by": "all", "source": "all"}))
    best = None
    for cells in _core_rows(url):
        if len(cells) >= 3:
            name = re.sub(r"^(The|IEEE|ACM)\s+", "", cells[0])
            s = max(lit.similarity(clean, name),
                    lit.similarity(clean, name.split(":")[0]))
            if s >= 0.9 and (not best or s > best[0]):
                best = (s, f"{cells[1]} {cells[2]}")
    return best[1] if best else None


ACRONYM_ALIASES = {"ESEC/FSE": "FSE", "ESEC-FSE": "FSE", "NIPS": "NeurIPS", "S&P": "SP",
                   "IEEE S&P": "SP", "USENIX Security": "USENIX-Security"}
# Tracks of a ranked conference keep the parent's rank, labelled as a track.
TRACK = re.compile(r"^(?P<parent>[A-Za-z&/\-]+?)[\s\-]*\(?(?P<track>D&B|NIER|SEIP|Industry|Findings|"
                   r"Demo|demo|Companion)\)?$")
WORKSHOP = re.compile(r"(-W$|Workshops?|\bREW\b|ICSTW|LNBIP)", re.I)
# Proceedings published as journals: ranked as the conference they are.
JOURNAL_AS_CONF = [(re.compile(r"Proceedings of the ACM on Software Engineering", re.I), "FSE"),
                   (re.compile(r"Proceedings of the VLDB Endowment", re.I), "VLDB"),
                   (re.compile(r"Proceedings of the AAAI Conference", re.I), "AAAI"),
                   (re.compile(r"Proceedings of the ACM on Programming Languages", re.I), "PLDI")]


def tiers(recs: list[dict], refresh: bool) -> dict:
    table = json.loads(VENUES.read_text()) if VENUES.exists() else {}
    for r in recs:
        kind = r.get("venue_kind")
        if kind in CONF_KINDS and r.get("venue_short"):
            short = r["venue_short"].strip()
            vkey = "conf:" + short
            if refresh and (vkey not in table or not table[vkey].startswith("CORE")):
                m = TRACK.match(short)
                acr = re.sub(r"\s*\d{2,4}$", "", m.group("parent") if m else short)
                acr = ACRONYM_ALIASES.get(acr, acr)
                if WORKSHOP.search(short):
                    table[vkey] = "workshop (not ranked)"
                    continue
                rank = core_conf(acr)
                if rank and m:
                    rank = f"{rank} ({m.group('track')} track)"
                table[vkey] = rank or "unranked/not found"
        elif kind == "journal" and r.get("venue"):
            vkey = "jnl:" + r["venue"].strip()
            if refresh and (vkey not in table or not table[vkey].startswith("CORE")):
                conf = next((a for rx, a in JOURNAL_AS_CONF if rx.search(r["venue"])), None)
                if conf:
                    rank = core_conf(conf)
                    table[vkey] = f"{rank} (as {conf})" if rank else "unranked/not found"
                else:
                    table[vkey] = core_journal(r["venue"]) or "not in CORE journal list"
    if refresh:
        VENUES.write_text(json.dumps(table, indent=1, ensure_ascii=False, sort_keys=True))
    return table


def tier_of(r: dict, table: dict) -> str:
    if r.get("venue_kind") in CONF_KINDS and r.get("venue_short"):
        return table.get("conf:" + r["venue_short"].strip(), "")
    if r.get("venue_kind") == "journal" and r.get("venue"):
        return table.get("jnl:" + r["venue"].strip(), "")
    return ""


# -------------------------------------------------------------------- PDFs

def get_pdfs(recs: list[dict]) -> None:
    for r in recs:
        url = r.get("pdf_url")
        if not url and r.get("arxiv_id"):
            url = f"https://arxiv.org/pdf/{r['arxiv_id']}"
        if not url:
            continue
        out = ROOT / "pdfs" / r["theme"] / f"{r['key']}.pdf"
        if out.exists() and out.stat().st_size > 1000:
            r["local_pdf"] = str(out.relative_to(ROOT))
            continue
        res = lit.fetch(url, str(out))
        if res.get("ok"):
            r["local_pdf"] = str(out.relative_to(ROOT))
        else:
            r["pdf_error"] = res.get("reason")


# ------------------------------------------------------------------ output

CSV_FIELDS = ["key", "title", "authors", "year", "venue", "venue_short", "venue_kind", "tier",
              "status", "doi", "arxiv_id", "url", "pdf_url", "local_pdf", "code_url", "code_stars",
              "code_license", "citations", "citations_source", "themes", "subtopics", "relevance",
              "role", "specdx_components", "baseline_candidate", "summary", "relation_to_specdx",
              "baseline_notes", "read_level", "verified_by"]


def write(recs: list[dict], table: dict) -> None:
    for r in recs:
        r["tier"] = tier_of(r, table)
    recs.sort(key=lambda r: (list(THEMES).index(r["theme"]) if r["theme"] in THEMES else 99,
                             REL_ORDER.get(r.get("relevance"), 3), -(r.get("year") or 0),
                             norm_title(r.get("title"))))

    (ROOT / "references.bib").write_text(
        "% Verified bibliography for the SpecDx related work (generated by tools/build.py).\n"
        "% Metadata comes from CrossRef/doi.org, arXiv or official proceedings pages.\n\n"
        + "\n\n".join(bibtex_for(r) for r in recs) + "\n", encoding="utf-8")

    with open(ROOT / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in recs:
            ci = r.get("code_info") or {}
            row = {f: r.get(f) for f in CSV_FIELDS}
            row.update({
                "authors": "; ".join(r.get("authors") or []),
                "themes": "; ".join(r.get("themes") or []),
                "subtopics": "; ".join(r.get("subtopics") or []),
                "specdx_components": "; ".join(r.get("specdx_components") or []),
                "code_stars": ci.get("stars"), "code_license": ci.get("license"),
            })
            w.writerow(row)

    lines = ["# SpecDx related work: catalogue", "",
             f"{len(recs)} verified papers, grouped by primary theme, then relevance and year. "
             "The citation keys match `references.bib`. Generated by `tools/build.py`; "
             "do not edit by hand (edit `raw/*.jsonl` and rebuild).", ""]
    counts = defaultdict(int)
    for r in recs:
        counts[r["theme"]] += 1
    lines += ["| Theme | Papers |", "|---|---:|"]
    lines += [f"| [{THEMES.get(t, t)}](#{t}) | {counts[t]} |" for t in THEMES if counts[t]]
    lines.append("")
    current = None
    for r in recs:
        if r["theme"] != current:
            current = r["theme"]
            lines += ["", f"<a id=\"{current}\"></a>", f"## {THEMES.get(current, current)}", ""]
        authors = r.get("authors") or []
        auth = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        venue = r.get("venue_short") or r.get("venue") or ""
        status = r.get("status") or ""
        tier = f", {r['tier']}" if r.get("tier") and "not" not in r["tier"] else ""
        link = f"https://doi.org/{r['doi']}" if r.get("doi") else r.get("url")
        extras = []
        if r.get("local_pdf"):
            extras.append(f"[PDF]({r['local_pdf']})")
        if r.get("code_url"):
            stars = (r.get("code_info") or {}).get("stars")
            extras.append(f"[code]({r['code_url']})" + (f" ★{stars}" if stars is not None else ""))
        if r.get("baseline_candidate"):
            extras.append("**baseline candidate**")
        also = [t for t in r.get("themes", [])[1:]]
        lines.append(f"- **[{r['key']}]** {auth} ({r.get('year')}). *{r.get('title')}*. "
                     f"{venue} ({status}{tier}). [{'link'}]({link})"
                     + (" · " + " · ".join(extras) if extras else ""))
        lines.append(f"  - Relevance **{r.get('relevance')}**; role: {r.get('role') or '-'}"
                     + (f"; also in {', '.join(also)}" if also else "")
                     + (f"; components: {', '.join(r['specdx_components'])}" if r.get("specdx_components") else ""))
        if r.get("summary"):
            lines.append(f"  - {r['summary']}")
        if r.get("relation_to_specdx"):
            lines.append(f"  - *Relation to SpecDx:* {r['relation_to_specdx']}")
    (ROOT / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", action="store_true", help="download open-access PDFs")
    ap.add_argument("--tiers", action="store_true", help="look up venue tiers on the CORE portal")
    a = ap.parse_args()
    raw = load()
    recs = merge(raw)
    make_keys(recs)
    table = tiers(recs, refresh=a.tiers)
    if a.pdfs:
        get_pdfs(recs)
    write(recs, table)
    by_status = defaultdict(int)
    for r in recs:
        by_status[(r.get("status") or "?").split(" ")[0]] += 1
    print(json.dumps({"raw_records": len(raw), "papers": len(recs),
                      "by_status": by_status,
                      "baseline_candidates": sum(1 for r in recs if r.get("baseline_candidate")),
                      "with_pdf": sum(1 for r in recs if r.get("local_pdf")),
                      "with_code": sum(1 for r in recs if r.get("code_url"))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
