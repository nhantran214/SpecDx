#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validator TRUNG LẬP CÔNG NGHỆ cho Canon Pack CASE-0031.

Chỉ dùng THƯ VIỆN CHUẨN (csv, json) — không phụ thuộc DB hay framework.
Đọc data/*.csv + invariants.json rồi kiểm mọi bất biến canon.
Có thể chép logic này sang bất kỳ ngôn ngữ nào (SQL/JS/Go) vì bất biến là DATA.

Chạy:  python validate.py            # exit 0 nếu tất cả PASS
"""
import csv, json, sys, pathlib, glob

HERE = pathlib.Path(__file__).resolve().parent

def load_data():
    tables = {}
    for f in sorted(glob.glob(str(HERE / "data" / "*.csv"))):
        name = pathlib.Path(f).stem
        import re as _re
        if _re.match(r"^\d+_", name):  # 06_soline -> soline; de_scope giữ nguyên
            name = name.split("_", 1)[1]
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        tables[name] = rows
    return tables

def is_blank(v):
    return v is None or str(v).strip() == ""

def match(row, where):
    for k, v in (where or {}).items():
        if str(row.get(k, "")).strip() != str(v):
            return False
    return True

def check(inv, tables):
    rows = tables.get(inv["table"], [])
    sel = [r for r in rows if match(r, inv.get("where"))]
    a = inv["assert"]
    if "count" in a:
        return len(sel) == a["count"], f"count={len(sel)} (kỳ vọng {a['count']})"
    if "values_of" in a:
        got = sorted({str(r.get(a["values_of"], "")).strip() for r in sel})
        exp = sorted(set(a["equals_set"]))
        return got == exp, f"{got}"
    if "all" in a:
        cond = a["all"]; c = cond["col"]
        bad = [r for r in sel if not _cell_ok(str(r.get(c, "")).strip(), cond, r)]
        return len(bad) == 0, (f"{len(bad)} dòng vi phạm" if bad else f"{len(sel)} dòng đạt")
    # assert trên MỘT dòng (lấy dòng đầu khớp)
    if not sel:
        return False, "không có dòng khớp"
    return _cell_ok(str(sel[0].get(a.get("col", ""), "")).strip(), a, sel[0]), \
           f"='{str(sel[0].get(a.get('col',''), '')).strip()}'"

def _cell_ok(val, cond, row):
    if "eq" in cond:        return val == str(cond["eq"])
    if "ne" in cond:        return val != str(cond["ne"])
    if "in" in cond:        return val in [str(x) for x in cond["in"]]
    if "contains" in cond:  return str(cond["contains"]) in val
    if "is_blank" in cond:  return is_blank(val) == bool(cond["is_blank"])
    return False

def advanced_checks():
    """Kiểm các file advanced ngoài 25 bảng (scenarios/event_kinds/kb/prices)."""
    import json as _json, csv as _csv, glob as _glob
    out=[]
    def a(name,cond,detail=""): out.append((name,bool(cond),detail))
    try:
        sc=_json.load(open(HERE/"data"/"scenarios.json",encoding="utf-8"))["scenarios"]
        names={s["name"] for s in sc}
        a("Đủ 3 scenario happy/discount8/grn12", names=={"happy","discount8","grn12"}, str(sorted(names)))
        g12=[s for s in sc if s["name"]=="grn12"][0]
        a("grn12: nhận 12 → pay 12, mở 3", g12["grn_received"]==12 and g12["expect"]["pay_units"]==12 and g12["expect"]["open_units"]==3)
        a("Quan hệ tiền: pay_units == grn_received", all(s["expect"]["pay_units"]==s["grn_received"] for s in sc))
        a("discount8: APR-0102 over_cap", [s for s in sc if s["name"]=="discount8"][0]["expect"]["apr_0102"]=="over_cap")
        ek=set(_json.load(open(HERE/"data"/"event_kinds.json",encoding="utf-8"))["kinds"])
        evk={r["kind"] for r in _csv.DictReader(open(HERE/"data"/"21_evt.csv",encoding="utf-8-sig"))}
        a("evt.kind đều trong registry", evk<=ek, f"{sorted(evk-ek)} ngoài registry")
        kbs=_glob.glob(str(HERE/"kb"/"*.md"))
        okfm=all(("kb_id:" in open(f,encoding="utf-8").read() and "domain:" in open(f,encoding="utf-8").read()) for f in kbs)
        a("KB markdown ≥3 file có front-matter", len(kbs)>=3 and okfm, f"{len(kbs)} file")
        pr={r["ma_sp"] for r in _csv.DictReader(open(HERE/"data"/"illustrative"/"prices.csv",encoding="utf-8-sig"))}
        a("Đơn giá minh họa có ITEM-0005/0012 (NON-canon)", {"ITEM-0005","ITEM-0012"}<=pr)
    except Exception as e:
        a("Advanced files hợp lệ", False, str(e))
    return out


def main():
    spec = json.load(open(HERE / "invariants.json", encoding="utf-8"))
    tables = load_data()
    invs = spec["invariants"]
    ok_n = 0
    print(f"\nCanon Pack {spec.get('canon_version','')} — kiểm {len(invs)} bất biến (trung lập công nghệ)\n" + "="*74)
    for inv in invs:
        ok, detail = check(inv, tables)
        ok_n += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {inv['id']:<20} {inv['desc']}" + (f"   ({detail})" if not ok else ""))
    # ===== ADVANCED (v2.0) =====
    for name, ok, detail in advanced_checks():
        ok_n += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if not ok else ""))
    total = len(invs) + len(advanced_checks())
    print("="*74)
    print(f"KẾT QUẢ: {ok_n}/{total} bất biến XANH" + ("  → GROUND OF TRUTH NHẤT QUÁN" if ok_n==total else "  → CÓ SAI LỆCH"))
    return 0 if ok_n == total else 1

if __name__ == "__main__":
    sys.exit(main())
