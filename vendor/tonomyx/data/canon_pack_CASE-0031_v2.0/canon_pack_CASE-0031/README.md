# Canon Pack — CASE-0031 · v2.0-ADVANCED (ground of truth TRUNG LẬP CÔNG NGHỆ)

Khóa cơ sở dữ liệu mẫu + policy + bất biến ở dạng **DATA** (không gắn tech stack).
Đổi stack (PostgreSQL → khác; Agno → NemoClaw/OpenShell → khác) **vẫn dùng lại được**:
chỉ cần viết một "bộ chiếu" (projection) mới; **dữ liệu canon và bất biến KHÔNG đổi**.

Nguồn khóa: **DBook v19 · NBook v17 · MBook v16** · phiên bản `CASE-0031-LOCK-1.0`.

## Thành phần
| Tệp | Vai trò |
|---|---|
| `canon.meta.json` | Phiên bản, nguồn, nguyên tắc, danh sách cột "để trống" (không canon). |
| `schema.json` | Định nghĩa 25 bảng (cột, kiểu, nullable, PK/FK) — **trung lập**, không gắn DB. |
| `data/*.csv` | Dữ liệu đã khóa (25 bảng). Ô để trống = không phải canon. |
| `invariants.json` | Bất biến canon dạng **khai báo** (mọi runner đọc & kiểm được). |
| `validate.py` | Validator **chỉ dùng thư viện chuẩn** (csv+json). Không cần DB/framework. |
| `generate_sql.py` | Bộ chiếu mẫu: pack → DDL + seed cho PostgreSQL. |
| `targets/postgres/` | Kết quả chiếu (ddl.sql, seed.sql) — minh chứng di động. |

## Dùng nhanh
```bash
python validate.py         # kiểm 33 bất biến canon trên data/ (không phụ thuộc stack)
python generate_sql.py     # chiếu sang PostgreSQL (targets/postgres/ddl.sql + seed.sql)
```
Port sang stack khác (MySQL/SQLite/kho document/API): viết một bộ chiếu tương tự
`generate_sql.py` đọc `schema.json` + `data/*.csv`. **Không đụng vào canon.**

## Nguyên tắc khóa
- **KHÓA** dữ kiện canon (Bảng 9.3): mã, số lượng, trạng thái, cổng, quan hệ
  (CUST 20+15; PO-0015=15; GRN=12/15; INSP 2/12→NCR-0007; SHIP trễ 2; VINV=15→'Cần kiểm';
  3 cổng APR-0102/P2, APR-0155/P6 maker-checker, APR-0210/P5; P4 xử lý ngoại lệ không mở APR;
  HR/offer là TH0 APR-TH-01, KHÔNG có CASE-0032).
- **ĐỂ TRỐNG** (blank/null) mọi số sách không xác lập: giá, tiền, chiết khấu, điểm NCC,
  offer, ngưỡng tiền. Validator kiểm cả việc "để trống" này (không được tự điền).

## Cập nhật khi sách đổi
Sửa `data/*.csv` và/hoặc `invariants.json`, tăng `canon_version` trong
`canon.meta.json` + `invariants.json`, rồi chạy lại `validate.py`. Mọi stack đọc lại pack.

## Guardrails & policy do CHÍNH PostgreSQL cưỡng chế (độc lập app-stack)
CSDL cố định = PostgreSQL. Nạp `targets/postgres/guardrails.sql` (sau ddl+seed):
- `no_mutate`/`trg_evt_ro`: EVT append-only.
- `trg_guard_pay`: chặn chi khi chưa có APR 'Đã duyệt' hoặc VINV còn 'Cần kiểm' — RAISE ngay ở DB.
- `fn_policy` / `fn_check_discount` / `fn_reconcile`: policy dạng hàm SQL, mọi app dùng chung.
→ Đổi framework agent (Agno/NemoClaw/n8n/service khác) trên nền PostgreSQL: guardrail & policy KHÔNG mất vì nằm ở CSDL.


## Phân loại dữ liệu & tách seed (vận hành thật)
- `schema.json.classification`: mỗi bảng gắn **master / control / operational / audit** (8/1/15/1).
- Seed tách 2 tầng (khớp vòng đời 6 tháng): `targets/postgres/seed_master.sql` (đổi chậm, có governance) + `targets/postgres/seed_operational.sql` (sinh theo nghiệp vụ, reset riêng mà GIỮ master).
- **Thứ tự nạp:** `ddl.sql` → `seed_master.sql` → `seed_operational.sql` → `guardrails.sql`.
- Reset chỉ giao dịch: chạy lại `seed_operational.sql` (master giữ nguyên).

## v2.0-advanced — độc lập app-stack (CSDL cố định = PostgreSQL)
**CANON (độc lập mọi app-stack):** `data/*.csv` (25 bảng) · `data/de_scope.csv` (bảng 26) · `data/scenarios.json` · `data/event_kinds.json` · `data/illustrative/prices.csv` (NON-canon) · `kb/*.md` · `schema.json` · `invariants.json` · `validate.py` · `targets/postgres/` (ddl + guardrails + seed_master + seed_operational).

**PHỤ THUỘC STACK (tách riêng, có nhãn):** `targets/tonomyx/` — Task/Result Envelope · task/user token · tool-catalog (9) · `policy.rego` · registry evt.kind TonomyX + mapping. Đổi stack khác → viết bản chiếu khác; **canon không đổi**.

**Enforcement ở PostgreSQL:** `guardrails.sql` = EVT append-only · chặn-chi (trg_guard_pay) · registry evt.kind (trg_evt_kind) · hàm `fn_policy/fn_check_discount/fn_reconcile`.

**Thứ tự nạp:** `ddl.sql → guardrails.sql → seed_master.sql → seed_operational.sql`.

**Nguyên tắc tiền (không tự mâu thuẫn sách):** giá/tiền canon vẫn **để trống**; số minh họa ở `data/illustrative/` (NON-canon); invariants chỉ khóa **quan hệ** (pay_units = số đã nhận; grn12 → 12, mở 3), không khóa số tuyệt đối.

**Kiểm thử:** `validate.py` **49/49**; nạp Postgres: chặn-chi + append-only + registry evt.kind + de_scope=9 đều đúng.
