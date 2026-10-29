"""Progress 2 — Join กับ ProdIPData

Input : data/processed/clean_flows.parquet, data/staging/prodipdata.parquet
Output: data/processed/features.parquet  (clean + คอลัมน์ geo)
        reports/ip_lookup.csv            (IP ทุกตัวใน UNSW → ประเภท / ประเทศ / ASN)
        reports/join_report.csv          (จำนวน IP ที่หาเจอ / หาไม่เจอ / private ฯลฯ)

ต้องทำ
1. แยกประเภท IP: PUBLIC / PRIVATE / LOOPBACK / MULTICAST (ระวัง 127.x ถูกนับเป็น private ด้วย)
2. IP แบบ PUBLIC → แปลงเป็นรูปแบบเดียวกับ IP24Prefix (เช่น 175.45.176.3 → 175.45.176.0/24) แล้วหาในตาราง
3. หาไม่เจอ → UNKNOWN, ไม่ใช่ PUBLIC → ใส่ชื่อประเภทแทน
4. ทำทั้ง srcip และ dstip ได้คอลัมน์ประเทศและ ASN ของทั้งสองฝั่ง
5. จำนวนแถวก่อนและหลัง join ต้องเท่ากัน
6. ProdIPData ใช้ "-" แทนค่าว่าง

"""

import sys
from ipaddress import IPv4Address, IPv4Network, AddressValueError
from pathlib import Path

import pandas as pd

# ── ให้ import utils.py ที่ root ของ repo ได้ ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, p


# =====================================================================
# ขั้นที่ 1: แยกประเภท IP
# =====================================================================
# ใช้โมดูล ipaddress ของ Python เพื่อจำแนก IP ออกเป็น 4 ประเภท:
#   - LOOPBACK  : 127.0.0.0/8 (ต้องตรวจก่อน PRIVATE เพราะ ipaddress
#                 นับ 127.x เป็น private ด้วย)
#   - PRIVATE   : 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
#   - MULTICAST : 224.0.0.0/4
#   - PUBLIC    : ที่เหลือทั้งหมด (is_global = True)
# =====================================================================
def classify_ip(ip_str: str) -> str:
    """จำแนกประเภทของ IP address
    คืนค่า: 'PUBLIC', 'PRIVATE', 'LOOPBACK', 'MULTICAST', หรือ 'INVALID'
    """
    try:
        addr = IPv4Address(ip_str.strip())
    except (AddressValueError, ValueError, AttributeError):
        return "INVALID"                    # IP ไม่ถูกรูปแบบ

    # ต้องตรวจ loopback ก่อน private เพราะ is_private จะ return True สำหรับ 127.x ด้วย
    if addr.is_loopback:                    # 127.0.0.0/8
        return "LOOPBACK"
    if addr.is_private:                     # 10.x, 172.16-31.x, 192.168.x
        return "PRIVATE"
    if addr.is_multicast:                   # 224.0.0.0/4
        return "MULTICAST"
    return "PUBLIC"                         # ที่เหลือคือ IP สาธารณะ


# =====================================================================
# ขั้นที่ 2: แปลง IP เป็น /24 prefix
# =====================================================================
# ProdIPData ใช้ key เป็น IP24Prefix เช่น "175.45.176.0/24"
# ต้องแปลง IP เช่น "175.45.176.3" -> "175.45.176.0/24"
# โดยการ mask octet สุดท้ายเป็น 0 แล้วต่อ /24
# =====================================================================
def ip_to_prefix24(ip_str: str) -> str:
    """แปลง IP address เป็น /24 prefix string
    เช่น '175.45.176.3' -> '175.45.176.0/24'
    """
    try:
        net = IPv4Network(ip_str.strip() + "/24", strict=False)
        return str(net)                     # เช่น '175.45.176.0/24'
    except (AddressValueError, ValueError, AttributeError):
        return ""


# =====================================================================
# ขั้นที่ 3: สร้าง lookup table จาก IP ที่ไม่ซ้ำทั้งหมด
# =====================================================================
# แทนที่จะ classify + lookup ทุกแถว (2 ล้านแถว)
# เราหา IP ที่ไม่ซ้ำก่อน (จำนวนน้อยกว่ามาก) แล้วสร้าง lookup table
# จากนั้น map กลับไปที่ DataFrame หลัก -> เร็วกว่ามาก
# =====================================================================
def build_ip_lookup(df: pd.DataFrame, ip_table: pd.DataFrame,
                    missing_val: str) -> pd.DataFrame:
    """สร้าง lookup table: IP ทุกตัว -> ประเภท, ประเทศ, ASN

    Parameters
    ----------
    df : DataFrame ของ clean_flows (มีคอลัมน์ srcip, dstip)
    ip_table : DataFrame ของ ProdIPData (มีคอลัมน์ IP24Prefix, CountryCode, ASNID ฯลฯ)
    missing_val : ค่าที่ ProdIPData ใช้แทนค่าว่าง (เช่น "-")

    Returns
    -------
    DataFrame ที่มีคอลัมน์: ip, ip_type, prefix24, country, asn, asn_name, ...
    """
    # --- รวบรวม IP ที่ไม่ซ้ำจากทั้ง srcip และ dstip ---
    all_ips = pd.concat([df["srcip"], df["dstip"]]).dropna().unique()
    print(f"  unique IPs: {len(all_ips):,}")

    # --- จำแนกประเภท IP ---
    lookup = pd.DataFrame({"ip": all_ips})
    lookup["ip_type"] = lookup["ip"].apply(classify_ip)

    # --- แปลง public IP เป็น /24 prefix ---
    mask_public = lookup["ip_type"] == "PUBLIC"
    lookup["prefix24"] = ""
    lookup.loc[mask_public, "prefix24"] = lookup.loc[mask_public, "ip"].apply(ip_to_prefix24)

    # --- join กับ ProdIPData เฉพาะ public IP ---
    # ใช้ left join เพื่อให้ทุก IP ยังอยู่ (หาไม่เจอจะเป็น NaN)
    lookup = lookup.merge(
        ip_table[["IP24Prefix", "CountryCode", "ASNID", "ASNName",
                  "ASNOrg", "ASNType", "ContinentCode",
                  "CityName", "Latitude", "Longitude"]],
        left_on="prefix24",
        right_on="IP24Prefix",
        how="left",
    )

    # --- จัดการค่าว่าง ---
    # ProdIPData ใช้ "-" แทนค่าว่าง -> เปลี่ยนเป็น NaN
    for col in ["CountryCode", "ASNID", "ASNName", "ASNOrg", "ASNType",
                "ContinentCode", "CityName"]:
        if col in lookup.columns:
            lookup[col] = lookup[col].replace(missing_val, pd.NA)

    # --- IP ที่ไม่ใช่ PUBLIC -> ใส่ชื่อประเภทแทนประเทศ/ASN ---
    # เช่น PRIVATE, LOOPBACK, MULTICAST -> country = "PRIVATE", asn = "PRIVATE"
    for ip_type in ["PRIVATE", "LOOPBACK", "MULTICAST", "INVALID"]:
        mask_type = lookup["ip_type"] == ip_type
        if mask_type.any():
            lookup.loc[mask_type, "CountryCode"] = ip_type
            lookup.loc[mask_type, "ASNID"] = ip_type

    # --- public IP ที่หาไม่เจอใน ProdIPData -> UNKNOWN ---
    mask_unknown = mask_public & lookup["CountryCode"].isna()
    lookup.loc[mask_unknown, "CountryCode"] = "UNKNOWN"
    lookup.loc[mask_unknown, "ASNID"] = "UNKNOWN"

    # ลบคอลัมน์ที่ไม่ต้องการ
    lookup = lookup.drop(columns=["IP24Prefix"], errors="ignore")

    return lookup


# =====================================================================
# ขั้นที่ 4: Join กลับเข้า DataFrame หลัก
# =====================================================================
# ใช้ lookup table ที่สร้างมา map ค่า country และ ASN ให้ทั้ง srcip และ dstip
# ผลลัพธ์: เพิ่ม 4 คอลัมน์ -> src_country, dst_country, src_asn, dst_asn
# =====================================================================
def join_geo(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join ข้อมูล geo เข้ากับ DataFrame หลัก ผ่าน lookup table

    สร้างคอลัมน์ใหม่: src_country, dst_country, src_asn, dst_asn
    """
    rows_before = len(df)

    # สร้าง dict สำหรับ map เร็วกว่า merge ซ้ำ 2 รอบ
    ip_to_country = lookup.set_index("ip")["CountryCode"].to_dict()
    ip_to_asn = lookup.set_index("ip")["ASNID"].to_dict()

    # --- map srcip ---
    df["src_country"] = df["srcip"].map(ip_to_country)
    df["src_asn"] = df["srcip"].map(ip_to_asn)

    # --- map dstip ---
    df["dst_country"] = df["dstip"].map(ip_to_country)
    df["dst_asn"] = df["dstip"].map(ip_to_asn)

    # --- ตรวจจำนวนแถว ---
    rows_after = len(df)
    if rows_before != rows_after:
        raise ValueError(
            f"row count changed after join: {rows_before:,} -> {rows_after:,}"
        )
    print(f"  rows before/after join: {rows_before:,} / {rows_after:,} (OK)")

    return df


# =====================================================================
# ขั้นที่ 5: สร้าง reports
# =====================================================================
def save_join_report(lookup: pd.DataFrame, report_path: Path) -> None:
    """บันทึก join_report.csv: สรุปจำนวน IP แต่ละประเภทและสถานะการ lookup"""
    # --- นับตามประเภท IP ---
    type_counts = lookup["ip_type"].value_counts().to_dict()

    # --- นับสถานะ lookup สำหรับ public IP ---
    mask_public = lookup["ip_type"] == "PUBLIC"
    public_total = mask_public.sum()
    found = mask_public & (lookup["CountryCode"] != "UNKNOWN")
    not_found = mask_public & (lookup["CountryCode"] == "UNKNOWN")

    rows = []
    rows.append({"metric": "total_unique_ips", "value": len(lookup)})
    for ip_type, count in sorted(type_counts.items()):
        rows.append({"metric": f"ip_type_{ip_type}", "value": count})
    rows.append({"metric": "public_found_in_prodipdata", "value": int(found.sum())})
    rows.append({"metric": "public_not_found (UNKNOWN)", "value": int(not_found.sum())})
    rows.append({"metric": "public_lookup_rate",
                 "value": f"{found.sum() / public_total * 100:.1f}%" if public_total else "N/A"})

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"  saved {report_path}")

    # แสดงสรุป
    print("\n-- join report --")
    for _, r in report.iterrows():
        print(f"  {r['metric']}: {r['value']}")


def save_ip_lookup(lookup: pd.DataFrame, lookup_path: Path) -> None:
    """บันทึก ip_lookup.csv: IP ทุกตัว -> ประเภท / ประเทศ / ASN"""
    lookup.to_csv(lookup_path, index=False, encoding="utf-8-sig")
    print(f"  saved {lookup_path}")


# =====================================================================
# main: รันทุกขั้นตอนตามลำดับ
# =====================================================================
def main() -> None:
    cfg = load_config()
    missing_val = cfg["prodipdata"]["missing_value"]    # "-"

    # ── เตรียม path ──
    processed = p(cfg["paths"]["processed"])
    staging = p(cfg["paths"]["staging"])
    reports = p(cfg["paths"]["reports"])
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # ── โหลดข้อมูล ──
    print("loading clean_flows")
    df = pd.read_parquet(processed / "clean_flows.parquet")
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("loading prodipdata")
    ip_table = pd.read_parquet(staging / "prodipdata.parquet")
    print(f"  {len(ip_table):,} rows")

    # ── สร้าง lookup table ──
    print("\nbuilding IP lookup table...")
    lookup = build_ip_lookup(df, ip_table, missing_val)

    # ── join geo กลับเข้า DataFrame หลัก ──
    print("\njoining geo data...")
    df = join_geo(df, lookup)
    print(f"  added columns: src_country, dst_country, src_asn, dst_asn")

    # ── บันทึกผลลัพธ์ ──
    output_path = processed / "features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nsaved {output_path}")
    print(f"  final: {len(df):,} rows, {df.shape[1]} columns")

    # ── บันทึก reports ──
    save_ip_lookup(lookup, reports / "ip_lookup.csv")
    save_join_report(lookup, reports / "join_report.csv")


if __name__ == "__main__":
    main()
