#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chiếu Canon Pack (schema.json + data/*.csv) → SQL (DDL + seed) cho PostgreSQL.

Minh chứng tính DI ĐỘNG: canon là DATA trung lập; bộ chiếu này tạo ra SQL cho một
tech stack cụ thể. Muốn stack khác (MySQL, SQLite, kho document, ...) chỉ cần viết
một bộ chiếu tương tự — DỮ LIỆU CANON KHÔNG ĐỔI. Chỉ dùng thư viện chuẩn.

Chạy:  python generate_sql.py   → ghi targets/postgres/ddl.sql & seed.sql
"""
import csv, json, sys, pathlib, glob

HERE = pathlib.Path(__file__).resolve().parent
TYPE = {"text":"TEXT","int":"INT","bigint":"BIGINT","numeric":"NUMERIC","bool":"BOOLEAN",
        "date":"DATE","timestamp":"TIMESTAMPTZ","serial":"BIGSERIAL"}

def ddl(schema):
    out = ["-- DDL sinh từ Canon Pack (schema.json) — target: PostgreSQL"]
    for t, cols in schema["tables"].items():
        lines, pks = [], [c["name"] for c in cols if c.get("pk")]
        for c in cols:
            s = f'  {c["name"]} {TYPE[c["type"]]}'
            if not c.get("nullable") and not c.get("pk"):
                s += " NOT NULL"
            lines.append(s)
        for c in cols:
            if c.get("fk"):
                ft, fc = c["fk"].split("."); lines.append(f'  FOREIGN KEY ({c["name"]}) REFERENCES {ft}({fc})')
        if pks:
            lines.append(f'  PRIMARY KEY ({", ".join(pks)})')
        out.append(f"CREATE TABLE {t} (\n" + ",\n".join(lines) + "\n);")
    return "\n".join(out) + "\n"

def sqlval(v, typ):
    if v is None or str(v).strip() == "":
        return "NULL"
    v = str(v).strip()
    if typ in ("int","bigint","numeric","serial"):
        return v
    if typ == "bool":
        return "true" if v.lower() in ("t","true","1") else "false"
    return "'" + v.replace("'", "''") + "'"

def seed(schema, only=None):
    cats = schema.get("classification", {})
    label = {"master":"MASTER+CONTROL", "operational":"OPERATIONAL+AUDIT"}.get(only or "all","")
    out = [f"-- SEED {label} sinh từ Canon Pack (data/*.csv) — target: PostgreSQL", "BEGIN;"]
    coltype = {tb: {c["name"]: c["type"] for c in cols} for tb, cols in schema["tables"].items()}
    def pick(tb):
        cat = cats.get(tb, "operational")
        if only == "master": return cat in ("master","control")
        if only == "operational": return cat in ("operational","audit")
        return True
    order = [t for t in schema["tables"].keys() if pick(t)]
    import re as _re
    def _tbl(fp):
        st=pathlib.Path(fp).stem
        return st.split("_",1)[1] if _re.match(r"^\d+_",st) else st
    files = {_tbl(f): f for f in glob.glob(str(HERE/"data"/"*.csv"))}
    trunc = ",".join(order)
    out.append(f"TRUNCATE {trunc} RESTART IDENTITY CASCADE;" if order else "")
    for t in order:
        if t not in files: continue
        rows = list(csv.DictReader(open(files[t], encoding="utf-8-sig")))
        if not rows: continue
        cols = list(rows[0].keys())
        for r in rows:
            vals = ", ".join(sqlval(r.get(c), coltype[t].get(c,"text")) for c in cols)
            out.append(f'INSERT INTO {t} ({", ".join(cols)}) VALUES ({vals});')
    out.append("COMMIT;")
    return "\n".join(out) + "\n"

def _seed_all(schema):
    out = ["-- SEED sinh từ Canon Pack (data/*.csv) — target: PostgreSQL", "BEGIN;"]
    coltype = {t: {c["name"]: c["type"] for c in cols} for t, cols in schema["tables"].items()}
    order = list(schema["tables"].keys())
    import re as _re
    def _tbl(fp):
        st=pathlib.Path(fp).stem
        return st.split("_",1)[1] if _re.match(r"^\d+_",st) else st
    files = {_tbl(f): f for f in glob.glob(str(HERE/"data"/"*.csv"))}
    for t in order:
        if t not in files:
            continue
        rows = list(csv.DictReader(open(files[t], encoding="utf-8-sig")))
        if not rows:
            continue
        cols = list(rows[0].keys())
        for r in rows:
            vals = ", ".join(sqlval(r.get(c), coltype[t].get(c,"text")) for c in cols)
            out.append(f'INSERT INTO {t} ({", ".join(cols)}) VALUES ({vals});')
    out.append("COMMIT;")
    return "\n".join(out) + "\n"

def main():
    schema = json.load(open(HERE/"schema.json", encoding="utf-8"))
    d = HERE/"targets"/"postgres"; d.mkdir(parents=True, exist_ok=True)
    (d/"ddl.sql").write_text(ddl(schema), encoding="utf-8")
    (d/"seed_master.sql").write_text(seed(schema, "master"), encoding="utf-8")
    (d/"seed_operational.sql").write_text(seed(schema, "operational"), encoding="utf-8")
    print("Đã sinh:", d/"ddl.sql", ", seed_master.sql, seed_operational.sql (nạp theo thứ tự: ddl → guardrails → master → operational)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
