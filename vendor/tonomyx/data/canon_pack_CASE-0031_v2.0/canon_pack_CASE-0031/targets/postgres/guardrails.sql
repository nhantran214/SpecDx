-- Guardrails & policy do CHÍNH PostgreSQL cưỡng chế (CSDL cố định) — độc lập app-stack.
-- Nạp SAU ddl.sql + seed.sql:  psql -f ddl.sql ; psql -f seed.sql ; psql -f guardrails.sql

-- (1) EVT append-only
CREATE OR REPLACE FUNCTION no_mutate() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'EVT la append-only: khong duoc UPDATE/DELETE'; END $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_evt_ro ON evt;
CREATE TRIGGER trg_evt_ro BEFORE UPDATE OR DELETE ON evt FOR EACH ROW EXECUTE FUNCTION no_mutate();

-- (2) Chặn chi khi chưa đủ căn cứ
CREATE OR REPLACE FUNCTION fn_guard_pay() RETURNS trigger AS $$
DECLARE ds text; n_appr int;
BEGIN
  IF NEW.trang_thai = 'Đã chi' AND COALESCE(OLD.trang_thai,'') <> 'Đã chi' THEN
    SELECT count(*) INTO n_appr FROM apr WHERE hoso_ref = NEW.ma_pay AND trang_thai = 'Đã duyệt';
    IF n_appr = 0 THEN
      RAISE EXCEPTION 'Guardrail CSDL: khong duoc chi % khi chua co APR Da duyet', NEW.ma_pay;
    END IF;
    SELECT v.doi_soat INTO ds FROM vinv v WHERE v.ma_vinv = NEW.ma_vinv;
    IF ds = 'Cần kiểm' THEN
      RAISE EXCEPTION 'Guardrail CSDL: khong duoc chi % khi doi soat VINV con Can kiem', NEW.ma_pay;
    END IF;
  END IF;
  RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_guard_pay ON pay;
CREATE TRIGGER trg_guard_pay BEFORE UPDATE ON pay FOR EACH ROW EXECUTE FUNCTION fn_guard_pay();

-- (3) Policy dạng hàm SQL — mọi app-stack gọi CHUNG một luật
CREATE OR REPLACE FUNCTION fn_policy(p_ma text) RETURNS numeric AS $$
  SELECT gia_tri FROM policy WHERE ma_nguong = p_ma ORDER BY version DESC LIMIT 1;
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION fn_check_discount(p_rate numeric) RETURNS boolean AS $$
  SELECT p_rate <= COALESCE(fn_policy('NG-CK-0005'), 0.05);
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION fn_reconcile(p_po text) RETURNS text AS $$
  SELECT CASE WHEN (SELECT sl_dat FROM po WHERE ma_po=p_po)=(SELECT sl_nhan FROM grn WHERE po=p_po)
              THEN 'Khớp' ELSE 'Cần kiểm' END;
$$ LANGUAGE sql STABLE;

-- (4) Registry evt.kind — evt chỉ nhận kind trong danh sách (independent vocab; enforce ở DB)
CREATE TABLE IF NOT EXISTS evt_kind (kind text PRIMARY KEY);
INSERT INTO evt_kind (kind) VALUES ('DRAFTED'),('FLAGGED'),('APPROVED'),('REJECTED'),('RECEIVED'),('INSPECTED'),('SHIPPED'),('RECONCILED'),('PAID'),('SENT') ON CONFLICT (kind) DO NOTHING;
CREATE OR REPLACE FUNCTION fn_evt_kind() RETURNS trigger AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM evt_kind WHERE kind = NEW.kind) THEN
    RAISE EXCEPTION 'evt.kind % khong trong registry', NEW.kind;
  END IF; RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_evt_kind ON evt;
CREATE TRIGGER trg_evt_kind BEFORE INSERT ON evt FOR EACH ROW EXECUTE FUNCTION fn_evt_kind();
