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

# ── ให้ import utils.py ที่ root ของ repo ได้ ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, p


# =====================================================================
# กฎที่ 1: แก้ไข attack_cat
# =====================================================================
# ปัญหาในข้อมูลดิบ:
#   - มีช่องว่างหน้า/หลังชื่อ เช่น " Fuzzers" แทน "Fuzzers"
#   - "Backdoors" กับ "Backdoor" เป็นค่าเดียวกัน ต้องรวมเป็น "Backdoor"
#   - แถวที่ label = 0 (ปกติ) มี attack_cat เป็น NaN → ต้องเติม "Normal"
#   - หลังแก้แล้วต้องเหลือ 10 ค่า: Normal, Analysis, Backdoor, DoS,
#     Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms
# =====================================================================
def rule1_attack_cat(df: pd.DataFrame) -> int:
    """แก้ไข attack_cat: ตัดช่องว่าง, Backdoors→Backdoor, NaN ที่ label=0 → Normal
    คืนค่า: จำนวนแถวที่ถูกแก้ไข
    """
    changed = 0

    # --- 1a. ตัดช่องว่างหน้า/หลัง ---
    # ข้อมูลดิบมี attack_cat แบบ " Fuzzers", " Generic " → strip ให้เหลือแค่ชื่อ
    original = df["attack_cat"].copy()                  # เก็บค่าเดิมไว้เทียบ
    df["attack_cat"] = df["attack_cat"].str.strip()     # ตัดช่องว่างหน้า/หลัง
    changed += (original != df["attack_cat"]).sum()      # นับจำนวนแถวที่เปลี่ยนไป

    # --- 1b. Backdoors → Backdoor ---
    # dataset มีทั้ง "Backdoors" และ "Backdoor" ซึ่งหมายถึงสิ่งเดียวกัน
    mask_bd = df["attack_cat"] == "Backdoors"           # หาแถวที่เป็น "Backdoors"
    changed += mask_bd.sum()                             # นับจำนวนที่จะแก้
    df.loc[mask_bd, "attack_cat"] = "Backdoor"          # เปลี่ยนเป็น "Backdoor"

    # --- 1c. ค่าว่าง (NaN) ที่ label = 0 -> Normal ---
    # แถวที่ label = 0 คือ traffic ปกติ แต่ attack_cat ถูกปล่อยว่าง
    # ต้องเติม "Normal" เพื่อให้ทุกแถวมี attack_cat ครบ
    label_col = "Label" if "Label" in df.columns else "label"
    mask_normal = df["attack_cat"].isna() & (df[label_col] == 0)  # NaN + label=0
    changed += mask_normal.sum()                                 # นับจำนวนที่จะเติม
    df.loc[mask_normal, "attack_cat"] = "Normal"                # เติม Normal

    return int(changed)


# =====================================================================
# กฎที่ 2: เติม 0 ให้คอลัมน์ ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd
# =====================================================================
# ปัญหา:
#   - ct_flw_http_mthd: ว่างเป็น NaN → flow ที่ไม่ใช่ HTTP ไม่มี method → เติม 0
#   - is_ftp_login: ว่างเป็น NaN → flow ที่ไม่ใช่ FTP ไม่มี login → เติม 0
#   - ct_ftp_cmd: ว่างเป็นช่องว่าง " " (ไม่ใช่ NaN!) → ต้องเปลี่ยน " " เป็น 0 ด้วย
# เหตุผลที่เติม 0: ค่าว่างในคอลัมน์เหล่านี้หมายถึง "ไม่เกิดเหตุการณ์" ไม่ใช่ "ข้อมูลหาย"
# =====================================================================
def rule2_fill_zero(df: pd.DataFrame) -> int:
    """เติม 0 ให้ ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd
    คืนค่า: จำนวนแถวที่ถูกแก้ไข (รวมทุกคอลัมน์)
    """
    changed = 0

    # --- 2a. ct_flw_http_mthd: NaN → 0 ---
    mask = df["ct_flw_http_mthd"].isna()                # หาแถวที่เป็น NaN
    changed += mask.sum()
    df.loc[mask, "ct_flw_http_mthd"] = 0

    # --- 2b. is_ftp_login: NaN → 0 ---
    mask = df["is_ftp_login"].isna()                     # หาแถวที่เป็น NaN
    changed += mask.sum()
    df.loc[mask, "is_ftp_login"] = 0

    # --- 2c. ct_ftp_cmd: " " (ช่องว่าง) และ NaN -> 0 ---
    # คอลัมน์นี้ถูกอ่านเป็น string ตอน ingest (dtype=str)
    # มีค่าว่างเป็นช่องว่าง " " หรือ NaN -> แปลงเป็นตัวเลข (Int64) แล้วเติม 0
    cmd_clean = df["ct_ftp_cmd"].astype(str).str.strip()
    mask = df["ct_ftp_cmd"].isna() | (cmd_clean == "") | (cmd_clean == "nan")
    changed += mask.sum()
    df["ct_ftp_cmd"] = pd.to_numeric(cmd_clean.replace("", "0"), errors="coerce").fillna(0).astype("Int64")

    return int(changed)


# =====================================================================
# กฎที่ 3: is_ftp_login ค่า 2 และ 4 → 1
# =====================================================================
# ปัญหา:
#   - is_ftp_login ควรเป็น 0 (ไม่ login) หรือ 1 (login สำเร็จ)
#   - แต่ในข้อมูลมีค่า 2 และ 4 ซึ่งไม่ใช่ค่าที่ถูกต้อง
#   - ค่า > 0 ทุกค่าหมายถึง "มี FTP login" → เปลี่ยนเป็น 1
# =====================================================================
def rule3_ftp_login(df: pd.DataFrame) -> int:
    """แก้ is_ftp_login: ค่า 2 และ 4 → 1
    คืนค่า: จำนวนแถวที่ถูกแก้ไข
    """
    mask = df["is_ftp_login"].isin([2, 4])               # หาแถวที่มีค่า 2 หรือ 4
    changed = mask.sum()
    df.loc[mask, "is_ftp_login"] = 1                     # เปลี่ยนเป็น 1 (login = จริง)
    return int(changed)


# =====================================================================
# กฎที่ 4: แปลง sport, dsport ที่เป็น hex → int  (vectorized)
# =====================================================================
# ปัญหา:
#   - sport/dsport ส่วนใหญ่เป็นตัวเลข เช่น "80", "443"
#   - บางค่าเป็น hex เช่น "0xcc09" (= 52233) → ต้องแปลงเป็น int
#   - hex ที่แปลงแล้วเกิน 65535 (port สูงสุด) → ไม่ใช่ port จริง → ใส่ NaN
#   - ค่า "-" → ไม่มี port → ใส่ NaN
#
# เวอร์ชันนี้ใช้ vectorized operations แทน .apply() ทีละแถว
# เร็วขึ้นประมาณ 10-50 เท่า เพราะ pandas ทำงานเป็น batch ใน C
# =====================================================================
def _convert_port_vectorized(series: pd.Series) -> tuple[pd.Series, int]:
    """แปลงคอลัมน์ port ทั้งคอลัมน์แบบ vectorized
    คืนค่า: (Series ที่แปลงแล้ว, จำนวนแถวที่เปลี่ยน)
    """
    changed = 0
    s = series.astype(str).str.strip()          # แปลงทั้งคอลัมน์เป็น string + ตัดช่องว่าง
    result = pd.array([pd.NA] * len(s), dtype="Int64")  # เริ่มจาก NaN ทั้งหมด (Int64 รองรับ NaN)

    # ---- กลุ่ม 1: ค่า "-" → NaN (ไม่ต้องทำอะไร เพราะ result เริ่มจาก NaN) ----
    mask_dash = s == "-"
    changed += mask_dash.sum()

    # ---- กลุ่ม 2: ค่า hex เช่น "0xcc09" ----
    mask_hex = s.str.lower().str.startswith("0x", na=False)
    if mask_hex.any():
        # แปลง hex → int ทีละตัว (ต้องใช้ apply เฉพาะกลุ่ม hex ซึ่งมีน้อยมาก)
        hex_vals = s[mask_hex].apply(lambda x: int(x, 16))
        # เฉพาะค่าที่ ≤ 65535 เท่านั้นที่ใช้ได้
        valid_hex = hex_vals[hex_vals <= 65535]
        result[valid_hex.index] = valid_hex.values
        changed += mask_hex.sum()       # hex ทุกตัวถูกแก้ (ไม่ว่าจะ valid หรือไม่)

    # ---- กลุ่ม 3: ค่าตัวเลขปกติ เช่น "80", "443" ----
    mask_normal = ~mask_dash & ~mask_hex & (s != "nan") & (s != "")
    if mask_normal.any():
        # pd.to_numeric แปลงเป็นตัวเลขทั้ง batch — ค่าที่แปลงไม่ได้จะเป็น NaN
        nums = pd.to_numeric(s[mask_normal], errors="coerce")
        valid_nums = nums.dropna().astype(int)
        result[valid_nums.index] = valid_nums.values

    return pd.Series(result, index=series.index, dtype="Int64"), int(changed)


def rule4_ports(df: pd.DataFrame) -> int:
    """แปลง sport, dsport: hex→int, hex>65535 และ '-' → NaN
    คืนค่า: จำนวนแถวที่ถูกแก้ไข (รวมทั้ง 2 คอลัมน์)
    """
    changed = 0

    for col in ["sport", "dsport"]:
        df[col], n = _convert_port_vectorized(df[col])
        changed += n

    return int(changed)


# =====================================================================
# กฎที่ 5: ลบแถวซ้ำ (เป็นขั้นสุดท้าย)
# =====================================================================
# ต้องทำหลังจากแก้ค่าทุกอย่างแล้ว เพราะ:
#   - ก่อนแก้: 2 แถวอาจดูต่างกัน (เช่น "Backdoors" vs "Backdoor")
#   - หลังแก้: กลายเป็นแถวเดียวกัน → ต้อง drop
# =====================================================================
def rule5_drop_duplicates(df: pd.DataFrame) -> int:
    """ลบแถวซ้ำ
    คืนค่า: จำนวนแถวที่ถูกลบ
    """
    before = len(df)
    df.drop_duplicates(inplace=True)            # ลบแถวที่ทุกคอลัมน์เหมือนกัน
    df.reset_index(drop=True, inplace=True)     # เรียงเลขแถวใหม่
    removed = before - len(df)
    return int(removed)


# =====================================================================
# กฎที่ 6: บันทึก cleaning_report.csv
# =====================================================================
# สร้างตาราง 2 คอลัมน์:
#   - rule: ชื่อกฎ (เช่น "1_attack_cat")
#   - rows_affected: จำนวนแถวที่ถูกแก้ไขโดยกฎนั้น
# =====================================================================
def save_report(report: list[dict], report_path: Path) -> None:
    """บันทึก cleaning report เป็น CSV"""
    report_df = pd.DataFrame(report)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"  saved {report_path}")


# =====================================================================
# main: รันทุกกฎตามลำดับ
# =====================================================================
def main() -> None:
    cfg = load_config()

    # ── เตรียม path ──
    staging = p(cfg["paths"]["staging"])
    processed = p(cfg["paths"]["processed"])
    reports = p(cfg["paths"]["reports"])
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # ── โหลดข้อมูล ──
    input_path = staging / "unsw_flows.parquet"
    print(f"loading {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # ── list เก็บผลลัพธ์ของแต่ละกฎ สำหรับ report ──
    report = []

    # ── กฎที่ 1: attack_cat ──
    print("\nrule 1: attack_cat (strip + Backdoors->Backdoor + NaN->Normal)")
    n = rule1_attack_cat(df)
    report.append({"rule": "1_attack_cat", "rows_affected": n})
    print(f"  affected: {n:,} rows")
    print(f"  unique values ({df['attack_cat'].nunique()}): {sorted(df['attack_cat'].dropna().unique().tolist())}")

    # ── กฎที่ 2: เติม 0 ──
    print("\nrule 2: fill 0 (ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd)")
    n = rule2_fill_zero(df)
    report.append({"rule": "2_fill_zero", "rows_affected": n})
    print(f"  affected: {n:,} rows")

    # ── กฎที่ 3: is_ftp_login ──
    print("\nrule 3: is_ftp_login (2,4 -> 1)")
    n = rule3_ftp_login(df)
    report.append({"rule": "3_ftp_login", "rows_affected": n})
    print(f"  affected: {n:,} rows")

    # ── กฎที่ 4: sport, dsport ──
    print("\nrule 4: sport/dsport (hex->int, >65535 and '-' -> NaN)")
    n = rule4_ports(df)
    report.append({"rule": "4_ports", "rows_affected": n})
    print(f"  affected: {n:,} rows")

    # ── กฎที่ 5: ลบแถวซ้ำ (ต้องทำสุดท้าย) ──
    print("\nrule 5: drop duplicates")
    rows_before = len(df)
    n = rule5_drop_duplicates(df)
    report.append({"rule": "5_drop_duplicates", "rows_affected": n})
    print(f"  removed: {n:,} rows ({rows_before:,} -> {len(df):,})")

    # ── บันทึกผลลัพธ์ ──
    output_path = processed / "clean_flows.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nsaved {output_path}")
    print(f"  final: {len(df):,} rows, {df.shape[1]} columns")

    # ── กฎที่ 6: บันทึก report ──
    save_report(report, reports / "cleaning_report.csv")

    # ── สรุป ──
    print("\n-- cleaning summary --")
    for r in report:
        print(f"  {r['rule']}: {r['rows_affected']:,}")


if __name__ == "__main__":
    main()
