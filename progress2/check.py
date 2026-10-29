"""Progress 2 — ตรวจเกณฑ์ผ่าน

Input : ไฟล์ output ทั้งหมดของ clean.py และ join.py

ต้องตรวจ
1. attack_cat เหลือ 10 ค่า ไม่มีค่าว่าง
2. ไม่มีแถวซ้ำ
3. ชนิดข้อมูลถูก และไม่มีค่าว่างในคอลัมน์ที่ห้ามว่าง
4. จำนวนแถวก่อนและหลัง join เท่ากัน
5. ip_lookup.csv มี IP ครบทุกตัว

วิธีที่ใช้: Pandas Assertions
  เขียน assertion function แต่ละข้อ ตรวจเงื่อนไขด้วย pandas vectorized operations
  ไม่ต้องติดตั้ง library เพิ่ม, เร็ว, สมาชิกทุกคนอ่านรู้เรื่อง

ตัวเลขสรุป
"""

import sys
from pathlib import Path

import pandas as pd

# ── ให้ import utils.py ที่ root ของ repo ได้ ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, p


# =====================================================================
# ข้อที่ 1: attack_cat เหลือ 10 ค่า ไม่มีค่าว่าง
# =====================================================================
# ทำไมต้องเช็ค?
#   attack_cat เป็น "target" ของโมเดล (features.yaml: target: attack_cat)
#   ถ้ามีค่าว่าง (NaN) หรือค่าผิดๆ (เช่น "Backdoors" ที่ยังไม่ถูก strip)
#   → โมเดลจะเทรนผิดหรือ class ไม่ครบ 10 ตัว
#
# ค่าที่ถูกต้องทั้ง 10 ค่า (Normal + 9 ประเภทการโจมตี):
#   Normal, Analysis, Backdoor, DoS, Exploits,
#   Fuzzers, Generic, Reconnaissance, Shellcode, Worms
#
# เงื่อนไขผ่าน:
#   1. จำนวน unique values = 10
#   2. จำนวน NaN = 0
#   3. ค่า unique ทั้ง 10 ตรงกับที่กำหนด
# =====================================================================
EXPECTED_ATTACK_CATS = sorted([
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
])


def check1_attack_cat(df: pd.DataFrame) -> dict:
    """ตรวจ attack_cat: ต้องมี 10 ค่า ไม่มี NaN
    คืนค่า: dict ที่มี check, passed, detail
    """
    # --- นับค่า NaN ---
    # isna() คืน True/False ทุกแถว แล้ว .sum() นับจำนวน True
    null_count = df["attack_cat"].isna().sum()

    # --- หาค่า unique ที่ไม่ใช่ NaN ---
    # dropna() เอา NaN ออกก่อน แล้ว unique() เอาค่าที่ไม่ซ้ำ
    actual = sorted(df["attack_cat"].dropna().unique().tolist())

    # --- เทียบกับค่าที่ถูกต้อง ---
    n_unique = len(actual)
    match = actual == EXPECTED_ATTACK_CATS

    passed = (null_count == 0) and match

    # --- สร้างข้อความสรุป ---
    detail = (
        f"unique={n_unique}, nulls={null_count:,}, "
        f"values={actual}"
    )
    if not match:
        # แสดงค่าที่ขาดหายหรือเกินมา
        missing = set(EXPECTED_ATTACK_CATS) - set(actual)
        extra = set(actual) - set(EXPECTED_ATTACK_CATS)
        if missing:
            detail += f", MISSING={missing}"
        if extra:
            detail += f", EXTRA={extra}"

    return {"check": "1_attack_cat", "passed": passed, "detail": detail}


# =====================================================================
# ข้อที่ 2: ไม่มีแถวซ้ำ
# =====================================================================
# ทำไมต้องเช็ค?
#   แถวซ้ำทำให้:
#   - โมเดลเห็นข้อมูลบางตัวซ้ำมากกว่าปกติ -> overfit
#   - metric (accuracy, f1 ฯลฯ) ถูกนับซ้ำ -> วัดผลผิด
#   - ขนาด dataset ใหญ่เกินจริง -> เปลืองหน่วยความจำ
#
# เงื่อนไขผ่าน: duplicated().sum() == 0
#
# หมายเหตุ: clean.py กฎที่ 5 ทำ drop_duplicates แล้ว
#   ข้อนี้เช็คซ้ำอีกทีเพื่อยืนยันว่า join ไม่ได้สร้างแถวซ้ำขึ้นมาใหม่
# =====================================================================
def check2_no_duplicates(df: pd.DataFrame) -> dict:
    """ตรวจว่าไม่มีแถวซ้ำ
    คืนค่า: dict ที่มี check, passed, detail
    """
    # duplicated() คืน True สำหรับแถวที่ซ้ำ (แถวแรกที่เจอจะเป็น False)
    # .sum() นับจำนวนแถวซ้ำ
    n_dup = df.duplicated().sum()
    passed = n_dup == 0

    detail = f"duplicated_rows={n_dup:,}, total_rows={len(df):,}"
    return {"check": "2_no_duplicates", "passed": passed, "detail": detail}


# =====================================================================
# ข้อที่ 3: ชนิดข้อมูลถูก + ไม่มีค่าว่างในคอลัมน์ที่ห้ามว่าง
# =====================================================================
# ทำไมต้องเช็ค?
#   3a. dtype ผิด (เช่น sport เป็น string แทน int)
#       → โมเดลอ่านค่าผิดหรือ crash ตอนเทรน
#   3b. ค่าว่างในคอลัมน์ geo (src_country, dst_country, src_asn, dst_asn)
#       → แปลว่า join.py มีบัค ไม่ได้ map ค่ากลับมาให้ครบ
#
# เงื่อนไขผ่าน:
#   3a. base columns ที่เป็นตัวเลข -> dtype เป็น numeric (int/float/Int64)
#       categorical columns -> dtype เป็น object/string/category
#   3b. geo columns ทั้ง 4 ตัว -> isna().sum() == 0
# =====================================================================
def check3_dtypes_and_nulls(df: pd.DataFrame) -> dict:
    """ตรวจ dtype ของคอลัมน์หลัก + ค่าว่างในคอลัมน์ geo
    คืนค่า: dict ที่มี check, passed, detail
    """
    problems = []

    # --- 3a. ตรวจ dtype ---
    # คอลัมน์ตัวเลข (base features ที่ต้องเป็น numeric)
    # ใช้ pd.api.types.is_numeric_dtype() เพราะรองรับทั้ง int64, float64, Int64
    numeric_cols = [
        "dur", "sbytes", "dbytes", "sloss", "dloss",
        "sload", "dload", "spkts", "dpkts",
        "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "sjit", "djit", "sintpkt", "dintpkt",
        "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_flw_http_mthd",
        "is_ftp_login", "ct_ftp_cmd",
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "sttl", "dttl", "ct_state_ttl",
    ]

    # ตรวจเฉพาะคอลัมน์ที่มีอยู่จริงใน DataFrame
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                problems.append(f"{col}: expected numeric, got {df[col].dtype}")

    # คอลัมน์ข้อความ (categorical) ต้องเป็น object/string/category
    cat_cols = ["proto", "state", "service"]
    for col in cat_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                problems.append(f"{col}: expected string/object, got {df[col].dtype}")

    # --- 3b. ตรวจค่าว่างในคอลัมน์ geo ---
    # join.py ต้อง map ค่า country และ asn ให้ทุกแถว
    # ถ้ามี NaN แปลว่า map พลาด
    geo_cols = ["src_country", "dst_country", "src_asn", "dst_asn"]
    for col in geo_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                problems.append(f"{col}: {n_null:,} nulls")
        else:
            problems.append(f"{col}: column MISSING")

    passed = len(problems) == 0
    detail = "OK" if passed else "; ".join(problems)

    return {"check": "3_dtypes_nulls", "passed": passed, "detail": detail}


# =====================================================================
# ข้อที่ 4: จำนวนแถวก่อนและหลัง join เท่ากัน
# =====================================================================
# ทำไมต้องเช็ค?
#   join.py ใช้ dict.map() (ไม่ใช่ merge) ดังนั้นจำนวนแถวไม่ควรเปลี่ยน
#   ถ้าเปลี่ยน → มีบัคร้ายแรง:
#   - แถวเพิ่ม = one-to-many join (IP ซ้ำในตาราง lookup)
#   - แถวลด = filter/drop ที่ไม่ตั้งใจ
#
# เงื่อนไขผ่าน: len(clean_flows) == len(features)
#
# หมายเหตุ: join.py เองก็เช็คอยู่แล้ว (raise ValueError ถ้าไม่เท่ากัน)
#   แต่ข้อนี้เช็คซ้ำจากไฟล์ output จริง เพื่อยืนยันว่าไฟล์ที่บันทึกถูกต้อง
# =====================================================================
def check4_row_count(df_clean: pd.DataFrame, df_features: pd.DataFrame) -> dict:
    """ตรวจจำนวนแถว clean_flows vs features ต้องเท่ากัน
    คืนค่า: dict ที่มี check, passed, detail
    """
    n_clean = len(df_clean)
    n_feat = len(df_features)
    passed = n_clean == n_feat

    detail = f"clean_flows={n_clean:,}, features={n_feat:,}"
    if not passed:
        diff = n_feat - n_clean
        detail += f", diff={diff:+,}"

    return {"check": "4_row_count", "passed": passed, "detail": detail}


# =====================================================================
# ข้อที่ 5: ip_lookup.csv มี IP ครบทุกตัว
# =====================================================================
# ทำไมต้องเช็ค?
#   ip_lookup.csv เก็บข้อมูล IP -> country/ASN ทุกตัวที่เจอใน dataset
#   ถ้า IP บางตัวหายไปจาก lookup → แถวที่ใช้ IP นั้นจะไม่มีข้อมูล geo
#   → วิเคราะห์ภูมิศาสตร์ไม่ครบ
#
# เงื่อนไขผ่าน:
#   IP ทุกตัวที่อยู่ใน features.parquet (srcip + dstip)
#   ต้องอยู่ใน ip_lookup.csv ด้วย (ไม่มี IP ที่หายไป)
# =====================================================================
def check5_ip_lookup(df_features: pd.DataFrame,
                     lookup_path: Path) -> dict:
    """ตรวจว่า ip_lookup.csv มี IP ครบทุกตัวที่ใช้ใน features
    คืนค่า: dict ที่มี check, passed, detail
    """
    # --- โหลด ip_lookup.csv ---
    if not lookup_path.exists():
        return {
            "check": "5_ip_lookup",
            "passed": False,
            "detail": f"FILE NOT FOUND: {lookup_path}",
        }

    lookup = pd.read_csv(lookup_path, dtype=str)

    # --- รวบรวม IP ที่ไม่ซ้ำจาก features ---
    # ต้องรวมทั้ง srcip และ dstip เพราะ lookup ต้องครอบคลุมทั้งสองฝั่ง
    ips_in_features = set(
        pd.concat([df_features["srcip"], df_features["dstip"]])
        .dropna()
        .unique()
    )

    # --- IP ที่อยู่ใน lookup ---
    ips_in_lookup = set(lookup["ip"].dropna().unique())

    # --- หา IP ที่หายไป (อยู่ใน features แต่ไม่อยู่ใน lookup) ---
    missing = ips_in_features - ips_in_lookup

    passed = len(missing) == 0

    detail = (
        f"ips_in_features={len(ips_in_features):,}, "
        f"ips_in_lookup={len(ips_in_lookup):,}, "
        f"missing={len(missing):,}"
    )
    if missing and len(missing) <= 10:
        # แสดง IP ที่หายถ้ามีไม่เกิน 10 ตัว (เพื่อ debug ง่าย)
        detail += f", examples={sorted(missing)}"

    return {"check": "5_ip_lookup", "passed": passed, "detail": detail}


# =====================================================================
# บันทึก report + แสดงผลสรุป
# =====================================================================
# สร้าง check_report.csv ที่มี 3 คอลัมน์: check, passed, detail
# เพื่อเก็บเป็นหลักฐานว่าข้อมูลผ่านเกณฑ์ทุกข้อ
# =====================================================================
def save_check_report(results: list[dict], report_path: Path) -> None:
    """บันทึก check_report.csv + พิมพ์สรุป"""
    report = pd.DataFrame(results)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"  saved {report_path}")


def print_summary(results: list[dict]) -> None:
    """พิมพ์ตาราง pass/fail ไปที่หน้าจอ"""
    print("\n" + "=" * 60)
    print("  CHECK REPORT SUMMARY")
    print("=" * 60)

    all_passed = True
    for r in results:
        # ใช้ PASS/FAIL แทนเครื่องหมาย Unicode เพื่อรองรับ cp874
        status = "PASS" if r["passed"] else "** FAIL **"
        print(f"  [{status}] {r['check']}")
        print(f"           {r['detail']}")
        if not r["passed"]:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        n_fail = sum(1 for r in results if not r["passed"])
        print(f"  RESULT: {n_fail} CHECK(S) FAILED")
    print("=" * 60)


# =====================================================================
# main: รันทุกข้อตรวจตามลำดับ
# =====================================================================
def main() -> None:
    cfg = load_config()

    # ── เตรียม path ──
    processed = p(cfg["paths"]["processed"])
    reports = p(cfg["paths"]["reports"])

    # ── โหลดข้อมูล ──
    # clean_flows.parquet = ผลจาก clean.py (ก่อน join)
    print("loading clean_flows.parquet")
    df_clean = pd.read_parquet(processed / "clean_flows.parquet")
    print(f"  {len(df_clean):,} rows, {df_clean.shape[1]} columns")

    # features.parquet = ผลจาก join.py (หลัง join -> มีคอลัมน์ geo เพิ่ม)
    print("loading features.parquet")
    df_feat = pd.read_parquet(processed / "features.parquet")
    print(f"  {len(df_feat):,} rows, {df_feat.shape[1]} columns")

    # ── รัน checks ──
    results = []

    # ข้อ 1: attack_cat
    print("\ncheck 1: attack_cat (10 values, no nulls)")
    r = check1_attack_cat(df_feat)
    results.append(r)
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'}: {r['detail']}")

    # ข้อ 2: ไม่มีแถวซ้ำ
    print("\ncheck 2: no duplicates")
    r = check2_no_duplicates(df_feat)
    results.append(r)
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'}: {r['detail']}")

    # ข้อ 3: dtype + nulls
    print("\ncheck 3: dtypes and nulls")
    r = check3_dtypes_and_nulls(df_feat)
    results.append(r)
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'}: {r['detail']}")

    # ข้อ 4: จำนวนแถว
    print("\ncheck 4: row count (clean vs features)")
    r = check4_row_count(df_clean, df_feat)
    results.append(r)
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'}: {r['detail']}")

    # ข้อ 5: ip_lookup
    print("\ncheck 5: ip_lookup coverage")
    lookup_path = reports / "ip_lookup.csv"
    r = check5_ip_lookup(df_feat, lookup_path)
    results.append(r)
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'}: {r['detail']}")

    # ── สรุป ──
    print_summary(results)

    # ── บันทึก report ──
    save_check_report(results, reports / "check_report.csv")


if __name__ == "__main__":
    main()
