"""Progress 2 Clean: staging UNSW -> processed clean_flows + cleaning_report."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config

PORT_MAX = 65535 


def to_abs(relative_path: str) -> Path: # ทำให้ path ที่อ้างอิงจาก root ของ repo เป็น absolute path
    return ROOT / relative_path


def normalize_attack_cat(df: pd.DataFrame) -> list[dict]: # ทำความสะอาดคอลัมน์ attack_cat: ตัดช่องว่าง, แก้ Backdoors -> Backdoor, เติม Normal เมื่อ label=0 และ attack_cat ว่าง
    report: list[dict] = []

    before = df["attack_cat"].astype("string") # ทำให้เป็น string ก่อน (เพื่อให้ .str.strip() ใช้ได้)
    trimmed = before.str.strip()
    changed = int((trimmed.fillna("<NA>") != before.fillna("<NA>")).sum())
    df["attack_cat"] = trimmed
    report.append({"rule": "attack_cat_trim", "affected_rows": changed}) # เก็บจำนวนแถวที่ถูกแก้ไข (trim) ไว้ใน report

    mask_backdoors = df["attack_cat"] == "Backdoors" # แก้ไขค่า Backdoors -> Backdoor
    count_backdoors = int(mask_backdoors.sum())
    df.loc[mask_backdoors, "attack_cat"] = "Backdoor" 
    report.append({"rule": "attack_cat_backdoors_to_backdoor", "affected_rows": count_backdoors})

    mask_empty = df["attack_cat"].isna() | (df["attack_cat"] == "") # เติมค่า Normal เมื่อ label=0 และ attack_cat ว่าง
    mask_normal = mask_empty & (df["label"] == 0)
    count_normal = int(mask_normal.sum()) # รายงานจำนวนแถวที่ถูกเติมค่า Normal
    df.loc[mask_normal, "attack_cat"] = "Normal" # วางค่า attack_cat ที่ว่างและ label=0 ให้เป็น Normal
    report.append({"rule": "attack_cat_fill_normal_when_label_0", "affected_rows": count_normal})

    return report


def fill_missing_ftp_http(df: pd.DataFrame) -> list[dict]: # เติมค่า 0 ให้กับคอลัมน์ ct_flw_http_mthd และ is_ftp_login ที่เป็น NaN
    report: list[dict] = []

    for col in ["ct_flw_http_mthd", "is_ftp_login"]:
        mask = df[col].isna() # ตรวจสอบว่ามีค่า NaN ในคอลัมน์หรือไม่
        count = int(mask.sum()) #นับจำนวนแถวที่เป็น NaN
        df.loc[mask, col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64") # แปลงเป็นตัวเลข (Int64) และเติมค่า NaN เป็น 0
        report.append({"rule": f"{col}_fill_na_to_0", "affected_rows": count})

    raw = df["ct_ftp_cmd"].astype("string") # แปลงเป็น string ก่อน (เพื่อให้ .str.strip() ใช้ได้)
    mask_blank = raw.isna() | (raw.str.strip() == "") # ตรวจสอบว่ามีค่า NaN หรือช่องว่างในคอลัมน์ ct_ftp_cmd
    count_blank = int(mask_blank.sum())
    df.loc[mask_blank, "ct_ftp_cmd"] = 0 # เติมค่า 0 ให้กับแถวที่เป็น NaN หรือช่องว่าง
    df["ct_ftp_cmd"] = pd.to_numeric(
        df["ct_ftp_cmd"].astype("string").str.strip(), errors="coerce"
    ).fillna(0).astype("Int64")
    report.append({"rule": "ct_ftp_cmd_fill_blank_to_0", "affected_rows": count_blank})

    return report


def map_is_ftp_login(df: pd.DataFrame) -> list[dict]:
    numeric = pd.to_numeric(df["is_ftp_login"], errors="coerce") # แปลงค่าในคอลัมน์ is_ftp_login เป็นตัวเลข (Int64) และเติมค่า NaN เป็น 0
    mask = numeric.isin([2, 4]) # ตรวจสอบว่ามีค่า 2 หรือ 4 ในคอลัมน์ is_ftp_login
    count = int(mask.sum())
    df.loc[mask, "is_ftp_login"] = 1 # แปลงค่า 2 หรือ 4 เป็น 1 (True) ในคอลัมน์ is_ftp_login
    return [{"rule": "is_ftp_login_2_4_to_1", "affected_rows": count}] 


def normalize_port_value(value: object) -> object:
    if pd.isna(value): # ตรวจสอบว่าค่าเป็น NaN หรือไม่
        return pd.NA

    text = str(value).strip() # แปลงค่าเป็น string และตัดช่องว่างด้านหน้า/หลัง
    if text == "" or text == "-":
        return pd.NA # ถ้าค่าเป็นช่องว่างหรือ "-" ให้คืนค่า NaN

    low = text.lower() # ตรวจสอบว่าค่าเป็นเลขฐาน 16 หรือไม่ (เริ่มต้นด้วย "0x") ถ้าใช่ให้แปลงเป็นเลขฐาน 10 และตรวจสอบว่ามากกว่า PORT_MAX หรือไม่
    if low.startswith("0x"):
        try:
            decoded = int(low, 16)
        except ValueError:
            return pd.NA
        return decoded if decoded <= PORT_MAX else pd.NA

    try:
        return int(text)
    except ValueError:
        return pd.NA


def normalize_ports(df: pd.DataFrame) -> list[dict]: # ทำความสะอาดคอลัมน์ sport และ dsport: แปลงค่าเป็น Int64, แปลงเลขฐาน 16 เป็นเลขฐาน 10, แปลง "-" เป็น NaN, ตรวจสอบว่าไม่เกิน PORT_MAX
    report: list[dict] = []

    for col in ["sport", "dsport"]:
        before = df[col].astype("string") 
        hex_mask = before.str.lower().str.startswith("0x", na=False) 

        hex_ok_count = 0
        hex_over_count = 0
        for value in before[hex_mask].tolist(): # วนลูปตรวจสอบค่าที่เป็นเลขฐาน 16 ว่ามีค่าเกิน PORT_MAX หรือไม่
            try:
                decoded = int(value, 16)
                if decoded <= PORT_MAX:
                    hex_ok_count += 1
                else:
                    hex_over_count += 1 # ไม่เกิน PORT_MAX ให้เพิ่มจำนวน hex_over_count
            except ValueError:
                hex_over_count += 1

        dash_count = int((before.str.strip() == "-").sum())

        df[col] = df[col].map(normalize_port_value).astype("Int64") # แปลงค่าในคอลัมน์ sport และ dsport เป็น Int64 และเติมค่า NaN ให้กับค่าที่ไม่สามารถแปลงได้

        report.append({"rule": f"{col}_hex_to_int", "affected_rows": hex_ok_count})
        report.append({"rule": f"{col}_hex_overflow_to_na", "affected_rows": hex_over_count})
        report.append({"rule": f"{col}_dash_to_na", "affected_rows": dash_count})

    return report


def deduplicate_last(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]: # ลบแถวซ้ำโดยเก็บแถวสุดท้ายไว้ และคืนค่า DataFrame ใหม่พร้อมรายงานจำนวนแถวที่ถูกลบ
    before = len(df)
    dedup = df.drop_duplicates(ignore_index=True)
    removed = before - len(dedup)
    return dedup, [{"rule": "drop_duplicate_rows", "affected_rows": removed}]


def main() -> None:
    cfg = load_config()

    staging_path = to_abs(cfg["paths"]["staging"]) / "unsw_flows.parquet" # กำหนด path ของไฟล์ unsw_flows.parquet ที่อยู่ในโฟลเดอร์ staging
    processed_dir = to_abs(cfg["paths"]["processed"]) 
    reports_dir = to_abs(cfg["paths"]["reports"])

    if not staging_path.exists(): # ตรวจสอบว่าไฟล์ unsw_flows.parquet มีอยู่หรือไม่ ถ้าไม่พบให้แจ้งข้อผิดพลาด
        raise FileNotFoundError(f"not found: {staging_path}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {staging_path}")
    flows = pd.read_parquet(staging_path)

    report_rows: list[dict] = [] # เก็บรายงานการทำความสะอาดข้อมูล
    report_rows.extend(normalize_attack_cat(flows))
    report_rows.extend(fill_missing_ftp_http(flows))
    report_rows.extend(map_is_ftp_login(flows))
    report_rows.extend(normalize_ports(flows))
    flows, dedup_report = deduplicate_last(flows)
    report_rows.extend(dedup_report)

    attack_unique = flows["attack_cat"].dropna().astype("string").nunique()
    if attack_unique != 10:
        raise ValueError(f"attack_cat unique count = {attack_unique}, expected 10")
    report_rows.append({"rule": "attack_cat_unique_after_clean", "affected_rows": int(attack_unique)})

    out_clean = processed_dir / "clean_flows.parquet"
    out_report = reports_dir / "cleaning_report.csv"

    flows.to_parquet(out_clean, index=False)
    pd.DataFrame(report_rows).to_csv(out_report, index=False)

    print(f"saved {out_clean}")
    print(f"saved {out_report}")
    print(f"clean rows: {len(flows):,}")


if __name__ == "__main__":
    main()
