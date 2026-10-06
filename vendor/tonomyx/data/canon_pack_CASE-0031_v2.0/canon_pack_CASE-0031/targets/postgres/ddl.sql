-- DDL sinh từ Canon Pack (schema.json) — target: PostgreSQL
CREATE TABLE cust (
  ma_kh TEXT,
  ten TEXT NOT NULL,
  hang TEXT,
  PRIMARY KEY (ma_kh)
);
CREATE TABLE item (
  ma_sp TEXT,
  ten TEXT NOT NULL,
  don_gia BIGINT,
  ton_kho INT,
  ck_toi_da NUMERIC,
  PRIMARY KEY (ma_sp)
);
CREATE TABLE vend (
  ma_ncc TEXT,
  ten TEXT NOT NULL,
  diem_chat_luong INT,
  PRIMARY KEY (ma_ncc)
);
CREATE TABLE cases (
  case_id TEXT,
  tieu_de TEXT,
  trang_thai TEXT,
  PRIMARY KEY (case_id)
);
CREATE TABLE so (
  ma_so TEXT,
  ma_kh TEXT NOT NULL,
  case_id TEXT NOT NULL,
  trang_thai TEXT,
  FOREIGN KEY (ma_kh) REFERENCES cust(ma_kh),
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_so)
);
CREATE TABLE soline (
  ma_soline TEXT,
  so TEXT NOT NULL,
  ma_sp TEXT NOT NULL,
  sl INT NOT NULL,
  don_gia BIGINT,
  chiet_khau NUMERIC,
  du_hang BOOLEAN NOT NULL,
  FOREIGN KEY (so) REFERENCES so(ma_so),
  FOREIGN KEY (ma_sp) REFERENCES item(ma_sp),
  PRIMARY KEY (ma_soline)
);
CREATE TABLE inv (
  ma_inv TEXT,
  ma_kh TEXT NOT NULL,
  case_id TEXT,
  so_tien BIGINT,
  doi_soat TEXT,
  trang_thai TEXT,
  FOREIGN KEY (ma_kh) REFERENCES cust(ma_kh),
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_inv)
);
CREATE TABLE po (
  ma_po TEXT,
  ma_ncc TEXT NOT NULL,
  case_id TEXT NOT NULL,
  sl_dat INT NOT NULL,
  don_gia BIGINT,
  trang_thai TEXT,
  FOREIGN KEY (ma_ncc) REFERENCES vend(ma_ncc),
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_po)
);
CREATE TABLE grn (
  ma_grn TEXT,
  po TEXT NOT NULL,
  sl_nhan INT NOT NULL,
  FOREIGN KEY (po) REFERENCES po(ma_po),
  PRIMARY KEY (ma_grn)
);
CREATE TABLE insp (
  ma_insp TEXT,
  grn TEXT NOT NULL,
  ket_qua TEXT,
  FOREIGN KEY (grn) REFERENCES grn(ma_grn),
  PRIMARY KEY (ma_insp)
);
CREATE TABLE ncr (
  ma_ncr TEXT,
  insp TEXT NOT NULL,
  loi TEXT,
  muc_do TEXT,
  FOREIGN KEY (insp) REFERENCES insp(ma_insp),
  PRIMARY KEY (ma_ncr)
);
CREATE TABLE ship (
  ma_ship TEXT,
  po TEXT NOT NULL,
  trang_thai TEXT,
  tre_ngay INT,
  FOREIGN KEY (po) REFERENCES po(ma_po),
  PRIMARY KEY (ma_ship)
);
CREATE TABLE vinv (
  ma_vinv TEXT,
  po TEXT NOT NULL,
  case_id TEXT,
  so_tien BIGINT,
  doi_soat TEXT,
  trang_thai TEXT,
  FOREIGN KEY (po) REFERENCES po(ma_po),
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_vinv)
);
CREATE TABLE pay (
  ma_pay TEXT,
  ma_vinv TEXT NOT NULL,
  so_tien BIGINT,
  trang_thai TEXT,
  FOREIGN KEY (ma_vinv) REFERENCES vinv(ma_vinv),
  PRIMARY KEY (ma_pay)
);
CREATE TABLE emp (
  ma_nv TEXT,
  ho_ten TEXT NOT NULL,
  phong TEXT,
  chuc_danh TEXT,
  luong BIGINT,
  trang_thai TEXT,
  PRIMARY KEY (ma_nv)
);
CREATE TABLE cand (
  ma_cand TEXT,
  vi_tri TEXT,
  case_id TEXT,
  muc_offer BIGINT,
  khung_max BIGINT,
  trang_thai TEXT,
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_cand)
);
CREATE TABLE att (
  ma_att TEXT,
  ma_nv TEXT NOT NULL,
  ngay DATE,
  loai TEXT,
  so_ngay NUMERIC,
  FOREIGN KEY (ma_nv) REFERENCES emp(ma_nv),
  PRIMARY KEY (ma_att)
);
CREATE TABLE esg (
  ma_esg TEXT,
  chi_tieu TEXT,
  ky TEXT,
  gia_tri NUMERIC,
  trang_thai TEXT,
  PRIMARY KEY (ma_esg)
);
CREATE TABLE apr (
  ma_apr TEXT,
  loai TEXT,
  noi_dung TEXT,
  case_id TEXT,
  hoso_ref TEXT,
  nguoi_tao TEXT,
  gia_tri BIGINT,
  muc_tu_hanh TEXT,
  trang_thai TEXT,
  created_at TIMESTAMPTZ,
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (ma_apr)
);
CREATE TABLE policy (
  ma_nguong TEXT,
  doi_tuong TEXT,
  gia_tri NUMERIC,
  nguon TEXT,
  version INT,
  PRIMARY KEY (ma_nguong, version)
);
CREATE TABLE evt (
  id BIGSERIAL,
  case_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  ref TEXT,
  actor TEXT,
  basis TEXT,
  at TIMESTAMPTZ,
  FOREIGN KEY (case_id) REFERENCES cases(case_id),
  PRIMARY KEY (id)
);
CREATE TABLE kb (
  ma_kb TEXT,
  tieu_de TEXT,
  nguon TEXT,
  version INT,
  PRIMARY KEY (ma_kb)
);
CREATE TABLE orc (
  ma_orc TEXT,
  ten TEXT,
  PRIMARY KEY (ma_orc)
);
CREATE TABLE de (
  ma_de TEXT,
  orc TEXT NOT NULL,
  vai TEXT,
  FOREIGN KEY (orc) REFERENCES orc(ma_orc),
  PRIMARY KEY (ma_de)
);
CREATE TABLE app_user (
  ma_user TEXT,
  ho_ten TEXT,
  role TEXT,
  quyen TEXT,
  PRIMARY KEY (ma_user)
);
CREATE TABLE de_scope (
  ma_de TEXT,
  tools TEXT,
  read_tables TEXT,
  draft_tables TEXT,
  create_apr TEXT,
  kb_domain TEXT,
  autonomy TEXT,
  budget_usd NUMERIC,
  send_email TEXT,
  FOREIGN KEY (ma_de) REFERENCES de(ma_de),
  PRIMARY KEY (ma_de)
);
