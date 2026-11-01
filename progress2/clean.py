"""Progress 2 — Clean

Input : data/staging/unsw_flows.parquet  (ผลจาก Progress 1)
Output: data/processed/clean_flows.parquet
        reports/cleaning_report.csv      (กฎแต่ละข้อแก้ไปกี่แถว)

ต้องทำ
1. attack_cat: ตัดช่องว่างหน้า/หลัง, Backdoors → Backdoor, ค่าว่างที่ label = 0 → Normal (ต้องเหลือ 10 ค่า)
2. ค่าว่างใน ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd → เติม 0 (ct_ftp_cmd ว่างเป็นช่องว่าง " " ไม่ใช่ NaN)
3. is_ftp_login ค่า 2 และ 4 → 1
4. sport, dsport: hex ไม่เกิน 65535 → int(x, 16), hex เกิน 65535 และ "-" → ค่าว่าง
5. ลบแถวซ้ำ เป็นขั้นสุดท้าย
6. บันทึกจำนวนแถวที่ถูกแก้ของทุกกฎลง cleaning_report.csv

"""
import sys
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo
 
FILL_ZERO = ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]  # ค่าว่าง = ไม่มี HTTP/FTP -> จำนวนเป็น 0 จริง
PORTS = ["sport", "dsport"]
PORT_MAX = 65535  # port มีค่าได้ 0-65535
N_CLASSES = 10    # Normal + 9 ประเภทการโจมตี
 
 
def require_file(path: Path) -> None:  # หยุดพร้อมบอกชื่อไฟล์ ถ้ายังไม่ได้รันขั้นก่อนหน้า
    if not path.exists():
        raise FileNotFoundError(f"not found: {path} (run the previous step first)")
 
 
# 1: attack_cat (ค่าดิบมี 13 ค่า + ค่าว่าง -> ต้องเหลือ 10 คลาส)
def clean_attack_cat(df: pd.DataFrame, log: list) -> None:
    raw = df["attack_cat"]
    s = raw.str.strip()  # " Fuzzers " -> "Fuzzers"
    log.append(("attack_cat", "cut leading/trailing spaces", int((raw.notna() & (raw != s)).sum())))  # ไม่นับค่าว่าง (NaN != NaN)
 
    is_backdoors = s == "Backdoors"
    log.append(("attack_cat", "Backdoors -> Backdoor", int(is_backdoors.sum())))
    s = s.mask(is_backdoors, "Backdoor")  # Backdoors กับ Backdoor คือคลาสเดียวกัน
 
    is_normal = s.isna() & (df["label"] == 0)  # ค่าว่างที่เป็นทราฟฟิกปกติ
    log.append(("attack_cat", "null (label 0) -> Normal", int(is_normal.sum())))
    df["attack_cat"] = s.mask(is_normal, "Normal")
 
 
# 2-3: ค่าว่าง -> 0, is_ftp_login 2/4 -> 1
def clean_fill_zero(df: pd.DataFrame, cols: list[str], log: list) -> None:
    for col in cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):  # ct_ftp_cmd อ่านมาเป็นข้อความ (มี " " ปน) -> แปลงเป็นตัวเลขก่อน
            s = pd.to_numeric(s.str.strip().replace("", None), errors="raise")
        log.append((col, "null -> 0", int(s.isna().sum())))
        df[col] = s.fillna(0).astype("int64")  # เติม 0 แล้วเป็นจำนวนเต็ม
 
    if "is_ftp_login" in cols:
        bad = df["is_ftp_login"].isin([2, 4])  # flag ต้องเป็น 0/1 เท่านั้น
        log.append(("is_ftp_login", "2 / 4 -> 1", int(bad.sum())))
        df.loc[bad, "is_ftp_login"] = 1  # ค่า > 0 = มีการ login
 
 
# 4: port (hex -> int, ค่าที่ไม่รู้ค่าจริง -> ปล่อยว่าง ห้ามเดา)
def clean_port(df: pd.DataFrame, col: str, log: list) -> None:
    s = df[col].str.strip()
    is_dash = s == "-"
    is_hex = s.str.lower().str.startswith("0x", na=False)
 
    num = pd.to_numeric(s.where(~is_hex & ~is_dash), errors="coerce").astype("float64")  # เลขฐาน 10 ปกติ
    num[is_hex] = s[is_hex].map(lambda x: int(x, 16)).astype("float64")  # "0xcc09" -> 52233
 
    too_big = num > PORT_MAX
    log.append((col, "hex (<= 65535) -> int", int((is_hex & ~too_big).sum())))
    log.append((col, "hex exceed 65535 -> null", int((is_hex & too_big).sum())))
    log.append((col, '"-" -> null', int(is_dash.sum())))
 
    unknown = num.isna() & ~is_dash & s.notna()  # ค่าแปลกที่ไม่อยู่ในกฎ -> หยุด
    if unknown.any():
        raise ValueError(f"{col}: unknown values {s[unknown].unique()[:10].tolist()}")
 
    df[col] = num.mask(too_big).astype("Int64")  # Int64 = จำนวนเต็มที่ว่างได้
 
 
# 5: ลบแถวซ้ำ (ทำหลังแก้ค่าทั้งหมด เพราะ " Fuzzers" กับ "Fuzzers" จะกลายเป็นแถวเดียวกัน)
def drop_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    data_cols = [c for c in df.columns if c != "source_file"]  # ไม่นับ source_file: แถวที่รอยต่อไฟล์ซ้ำกันข้ามไฟล์
    dup = df.duplicated(subset=data_cols, keep="first")  # เก็บแถวแรก ลบแถวที่ซ้ำ
    for i, n in dup.groupby(df["source_file"]).sum().items():  # นับว่าลบจากไฟล์ไหนกี่แถว
        log.append(("(all)", f"drop duplicates (file {i})", int(n)))
    return df[~dup].reset_index(drop=True)
 
 
def main() -> None:
    cfg = load_config()
    src = p(cfg["paths"]["staging"]) / "unsw_flows.parquet"
    out_dir, rep_dir = p(cfg["paths"]["processed"]), p(cfg["paths"]["reports"])
    require_file(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
 
    print("loading unsw_flows.parquet")
    df = pd.read_parquet(src)
    rows_in = len(df)
    print(f"  {rows_in:,} rows")
 
    print("cleaning")
    log: list[tuple[str, str, int]] = []  # (column, rule, rows) -> cleaning_report.csv
    clean_attack_cat(df, log)
    clean_fill_zero(df, FILL_ZERO, log)
    for col in PORTS:
        clean_port(df, col, log)
 
    print("drop duplicates")
    df = drop_duplicates(df, log)
    log.append(("(all)", "rows before clean", rows_in))
    log.append(("(all)", "rows after clean", len(df)))
 
    n_cls = df["attack_cat"].nunique(dropna=False)  # เช็กตั้งแต่ตอนนี้ ไม่ต้องรอถึง check.py
    if n_cls != N_CLASSES:
        raise ValueError(f"attack_cat has {n_cls} values, expected {N_CLASSES}: {sorted(df['attack_cat'].astype(str).unique())}")
 
    df.to_parquet(out_dir / "clean_flows.parquet", index=False)
    report = pd.DataFrame(log, columns=["column", "rule", "rows"])
    report.to_csv(rep_dir / "cleaning_report.csv", index=False, encoding="utf-8-sig")
    print("saved -> data/processed/clean_flows.parquet")
    print("saved -> reports/cleaning_report.csv\n")
    print(report.to_string(index=False))
 
 
if __name__ == "__main__":
    main()