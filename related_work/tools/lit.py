#!/usr/bin/env python3
"""Literature lookup and verification for the SpecDx related-work survey.

Standard library only. Every network answer is cached under tools/.cache, and
requests to one host are spaced across processes by a lock file, so several
agents can run this at the same time without being rate-limited.

Commands (all print JSON):
  search QUERY [--rows N] [--from-year Y]   CrossRef bibliographic search
  arxiv-search QUERY [--rows N]             arXiv search (title, abstract)
  s2-search QUERY [--rows N]                Semantic Scholar search (often rate-limited)
  openalex-title TITLE                      OpenAlex title lookup (small shared daily budget:
                                            last resort for DOI-less proceedings)
  verify --title T [--year Y] [--doi D] [--arxiv ID]
                                            canonical record + BibTeX + status
  bibtex DOI                                BibTeX from doi.org content negotiation
  refs DOI                                  CrossRef reference list (backward snowballing)
  github OWNER/REPO                         stars, licence, last push, archived
  fetch URL OUT_PATH                        download a PDF (checks the %PDF magic)
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
LOCKS = CACHE / "locks"
UA = "SpecDx-related-work/1.0 (academic literature survey; stdlib urllib)"

# Minimum seconds between two requests to the same host, across processes.
SPACING = {
    "api.crossref.org": 0.3,
    "doi.org": 0.3,
    "export.arxiv.org": 3.1,
    "api.semanticscholar.org": 1.5,
    "api.github.com": 0.8,
}

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"<[^>]+>", " ", text)  # CrossRef titles may carry markup
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _wait_turn(host: str) -> None:
    gap = SPACING.get(host, 0.2)
    LOCKS.mkdir(parents=True, exist_ok=True)
    with open(LOCKS / f"{host}.lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        last = float(raw) if raw else 0.0
        delay = last + gap - time.time()
        if delay > 0:
            time.sleep(delay)
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def http_get(url: str, accept: str | None = None, binary: bool = False,
             use_cache: bool = True, tries: int = 6):
    key = hashlib.sha1(f"{accept}|{url}".encode()).hexdigest()
    path = CACHE / ("bin" if binary else "txt") / key
    if use_cache and path.exists():
        data = path.read_bytes()
        return data if binary else data.decode("utf-8", "replace")
    host = urllib.parse.urlparse(url).netloc
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    err = None
    for attempt in range(tries):
        _wait_turn(host)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = resp.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp%d" % os.getpid())
            tmp.write_bytes(data)
            tmp.replace(path)
            return data if binary else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            err = exc
            if exc.code == 404:
                return None
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            err = exc
            time.sleep(min(30, 2 ** attempt))
    print(f"warning: GET failed after {tries} tries: {url} ({err})", file=sys.stderr)
    return None


# ---------------------------------------------------------------- CrossRef

STATUS_BY_TYPE = {
    "journal-article": "peer-reviewed (journal)",
    "proceedings-article": "peer-reviewed (conference/workshop proceedings)",
    "book-chapter": "book chapter",
    "book": "book",
    "monograph": "book",
    "posted-content": "preprint",
    "report": "report",
    "dissertation": "thesis",
}


def _full_title(it: dict) -> str:
    title = re.sub(r"<[^>]+>", "", (it.get("title") or [""])[0]).strip()
    sub = re.sub(r"<[^>]+>", "", (it.get("subtitle") or [""])[0]).strip()
    return f"{title}: {sub}" if sub and _norm(sub) not in _norm(title) else title


def title_match(query: str, found: str) -> float:
    """Similarity that also accepts a record whose stored title drops the subtitle."""
    s = similarity(query, found)
    nq, nf = _norm(query), _norm(found)
    if len(nf) >= 25 and nq.startswith(nf):
        s = max(s, 0.95)
    return s


def _cr_item(it: dict) -> dict:
    issued = (it.get("issued") or {}).get("date-parts") or [[None]]
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
               for a in it.get("author", [])]
    doi = it.get("DOI", "")
    status = STATUS_BY_TYPE.get(it.get("type"), it.get("type"))
    if doi.lower().startswith("10.48550/"):
        status = "preprint"
    pdf = None
    for link in it.get("link", []) or []:
        if "pdf" in (link.get("content-type") or ""):
            pdf = link.get("URL")
            break
    return {
        "doi": doi,
        "title": _full_title(it),
        "venue": (it.get("container-title") or [None])[0],
        "event": (it.get("event") or {}).get("name"),
        "publisher": it.get("publisher"),
        "year": issued[0][0],
        "type": it.get("type"),
        "status": status,
        "authors": authors,
        "citations_crossref": it.get("is-referenced-by-count"),
        "url": f"https://doi.org/{doi}" if doi else it.get("URL"),
        "pdf_link_crossref": pdf,
    }


def crossref_search(query: str, rows: int = 10, from_year: int | None = None) -> list[dict]:
    params = {"query.bibliographic": query, "rows": str(rows),
              "select": "DOI,title,subtitle,container-title,event,issued,author,type,"
                        "is-referenced-by-count,publisher,URL,link"}
    if from_year:
        params["filter"] = f"from-pub-date:{from_year}"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    raw = http_get(url)
    if not raw:
        return []
    return [_cr_item(it) for it in json.loads(raw)["message"]["items"]]


def crossref_doi(doi: str) -> dict | None:
    raw = http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not raw:
        return None
    return _cr_item(json.loads(raw)["message"])


def crossref_refs(doi: str) -> list[dict]:
    raw = http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not raw:
        return []
    out = []
    for r in json.loads(raw)["message"].get("reference", []) or []:
        out.append({k: r.get(k) for k in ("DOI", "article-title", "author", "year",
                                          "journal-title", "volume-title", "unstructured")
                    if r.get(k)})
    return out


def doi_bibtex(doi: str) -> str | None:
    raw = http_get("https://doi.org/" + doi, accept="application/x-bibtex")
    if raw and raw.lstrip().startswith("@"):
        return raw.strip()
    return None


# ------------------------------------------------------------------ arXiv

def _arxiv_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", ARXIV_NS):
        aid = (e.findtext("a:id", default="", namespaces=ARXIV_NS) or "").rsplit("/abs/", 1)[-1]
        if not aid:
            continue
        base = re.sub(r"v\d+$", "", aid)
        title = " ".join((e.findtext("a:title", default="", namespaces=ARXIV_NS) or "").split())
        out.append({
            "arxiv_id": base,
            "title": title,
            "authors": [a.findtext("a:name", default="", namespaces=ARXIV_NS)
                        for a in e.findall("a:author", ARXIV_NS)],
            "year": int((e.findtext("a:published", default="0", namespaces=ARXIV_NS) or "0")[:4] or 0),
            "updated": (e.findtext("a:updated", default="", namespaces=ARXIV_NS) or "")[:10],
            "primary_category": (e.find("arxiv:primary_category", ARXIV_NS).get("term")
                                 if e.find("arxiv:primary_category", ARXIV_NS) is not None else None),
            "journal_ref": e.findtext("arxiv:journal_ref", default=None, namespaces=ARXIV_NS),
            "doi": e.findtext("arxiv:doi", default=None, namespaces=ARXIV_NS),
            "comment": e.findtext("arxiv:comment", default=None, namespaces=ARXIV_NS),
            "abstract": " ".join((e.findtext("a:summary", default="", namespaces=ARXIV_NS) or "").split()),
            "url": f"https://arxiv.org/abs/{base}",
            "pdf_url": f"https://arxiv.org/pdf/{base}",
        })
    return out


def arxiv_search(query: str, rows: int = 10) -> list[dict]:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 1]
    q = " AND ".join(f"all:{w}" for w in words[:12]) or query
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": rows, "sortBy": "relevance"})
    raw = http_get(url)
    return _arxiv_entries(raw) if raw else []


def arxiv_title(title: str, rows: int = 5) -> list[dict]:
    """Title lookup that survives symbols (τ², LaTeX) by trying looser queries."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", title) if len(w) > 3]
    longest = sorted(set(words), key=len, reverse=True)
    queries = [
        " AND ".join(f"ti:{w}" for w in words[:8]),
        " AND ".join(f"ti:{w}" for w in longest[:4]),
        " AND ".join(f"all:{w}" for w in longest[:6]),
    ]
    seen: dict[str, dict] = {}
    for q in queries:
        if not q:
            continue
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": rows})
        raw = http_get(url)
        for e in (_arxiv_entries(raw) if raw else []):
            seen.setdefault(e["arxiv_id"], e)
        if any(similarity(title, e["title"]) >= 0.9 for e in seen.values()):
            break
    return list(seen.values())


def openalex_title(title: str) -> list[dict]:
    """Last resort for DOI-less proceedings. Costs a share of a small daily budget."""
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": "title.search:" + re.sub(r"[,:|]", " ", title), "per-page": 5,
         "select": "id,doi,title,publication_year,cited_by_count,primary_location,"
                   "open_access,authorships,type"})
    raw = http_get(url)
    if not raw:
        return []
    out = []
    for w in json.loads(raw).get("results", []):
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        out.append({
            "title": w.get("title"), "year": w.get("publication_year"), "doi": w.get("doi"),
            "venue": src.get("display_name"), "venue_type": src.get("type"), "type": w.get("type"),
            "citations_openalex": w.get("cited_by_count"),
            "oa_url": (w.get("open_access") or {}).get("oa_url"),
            "landing_page": loc.get("landing_page_url"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:10],
            "title_similarity": round(similarity(title, w.get("title") or ""), 3),
        })
    return out


def arxiv_id(aid: str) -> dict | None:
    aid = re.sub(r"^(arxiv:)", "", aid.strip(), flags=re.I)
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": aid})
    raw = http_get(url)
    entries = _arxiv_entries(raw) if raw else []
    return entries[0] if entries else None


# ------------------------------------------------------- Semantic Scholar

def s2_search(query: str, rows: int = 10) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": query, "limit": rows,
         "fields": "title,year,venue,publicationVenue,externalIds,citationCount,openAccessPdf,authors"})
    raw = http_get(url, tries=8)
    if not raw:
        return []
    out = []
    for p in json.loads(raw).get("data", []) or []:
        out.append({
            "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "arxiv_id": (p.get("externalIds") or {}).get("ArXiv"),
            "citations_s2": p.get("citationCount"),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            "authors": [a.get("name") for a in p.get("authors", [])][:8],
        })
    return out


# ----------------------------------------------------------------- verify

def _arxiv_bibtex(rec: dict) -> str:
    first = _norm(rec["authors"][0]).split()[-1] if rec.get("authors") else "anon"
    word = next((w for w in _norm(rec["title"]).split() if len(w) > 3), "paper")
    key = f"{first}{rec['year']}{word}"
    authors = " and ".join(rec.get("authors", []))
    return (f"@misc{{{key},\n  title = {{{rec['title']}}},\n  author = {{{authors}}},\n"
            f"  year = {{{rec['year']}}},\n  eprint = {{{rec['arxiv_id']}}},\n"
            f"  archivePrefix = {{arXiv}},\n  primaryClass = {{{rec.get('primary_category') or ''}}},\n"
            f"  url = {{{rec['url']}}},\n  note = {{Preprint}}\n}}")


def verify(title: str | None = None, year: int | None = None, doi: str | None = None,
           arxiv: str | None = None, min_sim: float = 0.9) -> dict:
    out: dict = {"query": {"title": title, "year": year, "doi": doi, "arxiv": arxiv}}
    cr = None
    if doi:
        cr = crossref_doi(doi)
        if cr and title:
            cr["title_similarity"] = round(title_match(title, cr["title"]), 3)
    elif title:
        best = None
        for cand in crossref_search(title, rows=8):
            s = title_match(title, cand["title"])
            if year and cand.get("year") and abs(int(cand["year"]) - int(year)) > 2:
                s -= 0.1
            if not best or s > best[0]:
                best = (s, cand)
        if best and best[0] >= min_sim:
            cr = dict(best[1], title_similarity=round(best[0], 3))
        elif best:
            out["crossref_nearest"] = dict(best[1], title_similarity=round(best[0], 3))

    ax = None
    if arxiv:
        ax = arxiv_id(arxiv)
    elif title or (cr and cr.get("title")):
        t = title or cr["title"]
        cands = arxiv_title(t)
        scored = sorted(((title_match(t, c["title"]), c) for c in cands), key=lambda x: -x[0])
        if scored and scored[0][0] >= min_sim:
            ax = dict(scored[0][1], title_similarity=round(scored[0][0], 3))

    # A CrossRef record of an arXiv DOI is the preprint itself, not a publication.
    if cr and cr.get("status") == "preprint" and not ax and cr["doi"].lower().startswith("10.48550/"):
        ax = arxiv_id(cr["doi"].split("arXiv.", 1)[-1])
    if cr and cr.get("status") == "preprint" and cr["doi"].lower().startswith("10.48550/"):
        cr = None

    if cr:
        out["matched"] = "crossref"
        out["record"] = cr
        out["bibtex"] = doi_bibtex(cr["doi"]) if cr.get("doi") else None
    elif ax:
        out["matched"] = "arxiv"
        out["record"] = {
            "title": ax["title"], "authors": ax["authors"], "year": ax["year"],
            "venue": ax.get("journal_ref") or "arXiv", "status": "preprint",
            "doi": ax.get("doi"), "url": ax["url"],
        }
        if ax.get("journal_ref") or ax.get("doi"):
            out["record"]["status"] = "preprint (journal_ref/doi on arXiv says published; confirm venue)"
        out["bibtex"] = _arxiv_bibtex(ax)
    else:
        out["matched"] = None
        out["record"] = None
    if ax:
        out["arxiv"] = {k: ax.get(k) for k in ("arxiv_id", "url", "pdf_url", "journal_ref",
                                               "comment", "doi", "primary_category", "updated",
                                               "abstract")}
        out["pdf_url"] = ax["pdf_url"]
    elif cr and cr.get("pdf_link_crossref"):
        out["pdf_url"] = cr["pdf_link_crossref"]
    return out


# ----------------------------------------------------------------- GitHub

def github(repo: str) -> dict | None:
    repo = re.sub(r"^https?://github\.com/", "", repo.strip()).strip("/")
    repo = "/".join(repo.split("/")[:2])
    raw = http_get(f"https://api.github.com/repos/{repo}", accept="application/vnd.github+json")
    if not raw:
        return None
    d = json.loads(raw)
    return {
        "repo": d.get("full_name"), "url": d.get("html_url"), "stars": d.get("stargazers_count"),
        "license": (d.get("license") or {}).get("spdx_id"), "pushed_at": d.get("pushed_at"),
        "archived": d.get("archived"), "language": d.get("language"),
        "size_kb": d.get("size"), "default_branch": d.get("default_branch"),
        "description": d.get("description"),
    }


def fetch(url: str, out: str) -> dict:
    data = http_get(url, binary=True, accept="application/pdf")
    if not data:
        return {"ok": False, "reason": "download failed", "url": url}
    if not data[:5].startswith(b"%PDF"):
        return {"ok": False, "reason": "not a PDF", "url": url, "head": data[:80].decode("latin1")}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"ok": True, "path": out, "bytes": len(data)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--rows", type=int, default=10)
    s.add_argument("--from-year", type=int)
    s = sub.add_parser("arxiv-search"); s.add_argument("query"); s.add_argument("--rows", type=int, default=10)
    s = sub.add_parser("s2-search"); s.add_argument("query"); s.add_argument("--rows", type=int, default=10)
    s = sub.add_parser("openalex-title"); s.add_argument("title")
    s = sub.add_parser("verify"); s.add_argument("--title"); s.add_argument("--year", type=int)
    s.add_argument("--doi"); s.add_argument("--arxiv"); s.add_argument("--min-sim", type=float, default=0.9)
    s.add_argument("--no-abstract", action="store_true")
    s = sub.add_parser("bibtex"); s.add_argument("doi")
    s = sub.add_parser("refs"); s.add_argument("doi")
    s = sub.add_parser("github"); s.add_argument("repo")
    s = sub.add_parser("fetch"); s.add_argument("url"); s.add_argument("out")
    a = ap.parse_args(argv)

    if a.cmd == "search":
        res = crossref_search(a.query, a.rows, a.from_year)
    elif a.cmd == "arxiv-search":
        res = arxiv_search(a.query, a.rows)
    elif a.cmd == "s2-search":
        res = s2_search(a.query, a.rows)
    elif a.cmd == "openalex-title":
        res = openalex_title(a.title)
    elif a.cmd == "verify":
        if not (a.title or a.doi or a.arxiv):
            ap.error("verify needs --title, --doi or --arxiv")
        res = verify(a.title, a.year, a.doi, a.arxiv, a.min_sim)
        if a.no_abstract and res.get("arxiv"):
            res["arxiv"].pop("abstract", None)
    elif a.cmd == "bibtex":
        res = {"doi": a.doi, "bibtex": doi_bibtex(a.doi)}
    elif a.cmd == "refs":
        res = crossref_refs(a.doi)
    elif a.cmd == "github":
        res = github(a.repo)
    else:
        res = fetch(a.url, a.out)
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
