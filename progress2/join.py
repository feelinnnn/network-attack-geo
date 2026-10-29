"""Progress 2 Join: clean_flows + prodipdata -> features + reports."""

import ipaddress
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config


def to_abs(relative_path: str) -> Path:
    return ROOT / relative_path


def classify_ip(ip_text: object) -> str: # จำแนกประเภทของ IP address เป็น PUBLIC, PRIVATE, LOOPBACK, MULTICAST หรือ UNKNOWN
    if pd.isna(ip_text):
        return "UNKNOWN"

    text = str(ip_text).strip()
    if text == "":
        return "UNKNOWN"

    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "UNKNOWN"

    # Must be before private: 127.x is also private in Python.
    if ip_obj.is_loopback:
        return "LOOPBACK"
    if ip_obj.is_multicast:
        return "MULTICAST"
    if ip_obj.is_private:
        return "PRIVATE"
    return "PUBLIC"


def to_ip24_prefix(ip_text: object) -> str: # แปลง IP address เป็น /24 prefix
    try:
        ip_obj = ipaddress.ip_address(str(ip_text).strip())
    except ValueError:
        return ""

    if ip_obj.version != 4:
        return ""

    a, b, c, _ = str(ip_obj).split(".")
    return f"{a}.{b}.{c}.0/24"


def build_ip_lookup(flows: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame: # สร้างตาราง lookup ของ IP address โดยรวมค่า srcip และ dstip จาก flows และเชื่อมกับ prodipdata
    src = flows[["srcip"]].rename(columns={"srcip": "ip"})
    dst = flows[["dstip"]].rename(columns={"dstip": "ip"})
    lookup = pd.concat([src, dst], ignore_index=True).drop_duplicates(ignore_index=True)

    lookup["ip_type"] = lookup["ip"].map(classify_ip)
    lookup["IP24Prefix"] = ""

    pub_mask = lookup["ip_type"] == "PUBLIC" # จำแนกเฉพาะ IP address ที่เป็น PUBLIC และแปลงเป็น /24 prefix
    lookup.loc[pub_mask, "IP24Prefix"] = lookup.loc[pub_mask, "ip"].map(to_ip24_prefix)

    merged = lookup.merge(prod, how="left", on="IP24Prefix")

    for col in [
        "CountryCode",
        "ASNID",
        "ASNName",
        "ASNOrg",
        "ASNType",
        "ContinentCode",
        "CityName",
    ]:
        merged[col] = merged[col].astype("string")

    public_found = pub_mask & merged["CountryCode"].notna()
    public_unknown = pub_mask & merged["CountryCode"].isna()
    non_public = ~pub_mask

    merged.loc[public_found, "country"] = merged.loc[public_found, "CountryCode"]
    merged.loc[public_found, "asn"] = merged.loc[public_found, "ASNID"]

    merged.loc[public_unknown, "country"] = "UNKNOWN"
    merged.loc[public_unknown, "asn"] = "UNKNOWN"

    merged.loc[non_public, "country"] = merged.loc[non_public, "ip_type"]
    merged.loc[non_public, "asn"] = merged.loc[non_public, "ip_type"]

    merged["country"] = merged["country"].astype("string").fillna("UNKNOWN")
    merged["asn"] = merged["asn"].astype("string").fillna("UNKNOWN")

    for col in ["ASNName", "ASNOrg", "ASNType", "ContinentCode", "CityName"]:
        merged[col] = merged[col].fillna("UNKNOWN")

    merged["Latitude"] = pd.to_numeric(merged["Latitude"], errors="coerce")
    merged["Longitude"] = pd.to_numeric(merged["Longitude"], errors="coerce")
    merged["Latitude"] = merged["Latitude"].where(merged["Latitude"].notna(), pd.NA)
    merged["Longitude"] = merged["Longitude"].where(merged["Longitude"].notna(), pd.NA)

    return merged[
        [
            "ip",
            "ip_type",
            "IP24Prefix",
            "country",
            "asn",
            "ASNName",
            "ASNOrg",
            "ASNType",
            "ContinentCode",
            "CityName",
            "Latitude",
            "Longitude",
        ]
    ]


def attach_geo_columns(flows: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame: # เพิ่มคอลัมน์ geo จาก lookup ให้กับ flows โดย join ตาม srcip และ dstip
    src_lookup = lookup[["ip", "country", "asn"]].rename(
        columns={"ip": "srcip", "country": "src_country", "asn": "src_asn"}
    )
    dst_lookup = lookup[["ip", "country", "asn"]].rename(
        columns={"ip": "dstip", "country": "dst_country", "asn": "dst_asn"}
    )

    features = flows.merge(src_lookup, how="left", on="srcip")
    features = features.merge(dst_lookup, how="left", on="dstip")

    for col in ["src_country", "dst_country", "src_asn", "dst_asn"]:
        features[col] = features[col].astype("string").fillna("UNKNOWN")

    return features


def build_side_metrics(flows: pd.DataFrame, lookup: pd.DataFrame, side_name: str) -> list[dict]: # สร้างรายงานสถิติของ IP address สำหรับ srcip หรือ dstip โดยนับจำนวน IP address ตามประเภท (PUBLIC, PRIVATE, LOOPBACK, MULTICAST, UNKNOWN) และจำนวน unique IP address
    ip_col = f"{side_name}ip"

    side_rows = flows[[ip_col]].rename(columns={ip_col: "ip"})
    side_rows = side_rows.merge(lookup[["ip", "ip_type", "country"]], how="left", on="ip")
    side_rows["ip_type"] = side_rows["ip_type"].astype("string").fillna("UNKNOWN")
    side_rows["country"] = side_rows["country"].astype("string").fillna("UNKNOWN")

    side_unique = side_rows.drop_duplicates(subset=["ip"]).copy()

    def metric_pack(df: pd.DataFrame, prefix: str) -> list[dict]: # สร้างรายงานสถิติของ IP address สำหรับ srcip หรือ dstip โดยนับจำนวน IP address ตามประเภท (PUBLIC, PRIVATE, LOOPBACK, MULTICAST, UNKNOWN) และจำนวน unique IP address
        public = df["ip_type"] == "PUBLIC"
        private = df["ip_type"] == "PRIVATE"
        loopback = df["ip_type"] == "LOOPBACK"
        multicast = df["ip_type"] == "MULTICAST"
        unknown = df["ip_type"] == "UNKNOWN"

        public_found = public & (df["country"] != "UNKNOWN")
        public_unknown = public & (df["country"] == "UNKNOWN")

        return [
            {"metric": f"{side_name}_{prefix}_public_ips", "value": int(public.sum())},
            {"metric": f"{side_name}_{prefix}_public_lookup_found", "value": int(public_found.sum())},
            {"metric": f"{side_name}_{prefix}_public_lookup_unknown", "value": int(public_unknown.sum())},
            {"metric": f"{side_name}_{prefix}_private_ips", "value": int(private.sum())},
            {"metric": f"{side_name}_{prefix}_loopback_ips", "value": int(loopback.sum())},
            {"metric": f"{side_name}_{prefix}_multicast_ips", "value": int(multicast.sum())},
            {"metric": f"{side_name}_{prefix}_invalid_or_unknown_ips", "value": int(unknown.sum())},
        ]

    # Backward-compatible metric names (treated as row-level counts).
    legacy_rows = metric_pack(side_rows, "rows")
    for item in legacy_rows:
        item["metric"] = item["metric"].replace(f"{side_name}_rows_", f"{side_name}_")

    # Explicit unique-IP counts per side.
    unique_ip_metrics = metric_pack(side_unique, "unique_ips")

    return legacy_rows + unique_ip_metrics


def main() -> None: # ฟังก์ชันหลักสำหรับสร้าง features.parquet และรายงาน join_report.csv จาก clean_flows.parquet และ prodipdata.parquet
    cfg = load_config()

    processed_dir = to_abs(cfg["paths"]["processed"])
    reports_dir = to_abs(cfg["paths"]["reports"])
    clean_path = processed_dir / "clean_flows.parquet"
    prod_path = to_abs(cfg["paths"]["staging"]) / "prodipdata.parquet"

    if not clean_path.exists(): # ตรวจสอบว่าไฟล์ clean_flows.parquet มีอยู่หรือไม่ ถ้าไม่พบให้แจ้งข้อผิดพลาด
        raise FileNotFoundError(f"not found: {clean_path}")
    if not prod_path.exists():
        raise FileNotFoundError(f"not found: {prod_path}")

    reports_dir.mkdir(parents=True, exist_ok=True)

    flows = pd.read_parquet(clean_path)
    prod = pd.read_parquet(
        prod_path,
        columns=[
            "IP24Prefix",
            "CountryCode",
            "ASNID",
            "ASNName",
            "ASNOrg",
            "ASNType",
            "ContinentCode",
            "CityName",
            "Latitude",
            "Longitude",
        ],
    )

    # ProdIPData uses '-' as missing value.
    prod = prod.replace({"-": pd.NA}).drop_duplicates(subset=["IP24Prefix"], ignore_index=True)

    lookup = build_ip_lookup(flows, prod)
    features = attach_geo_columns(flows, lookup)

    if len(features) != len(flows):
        raise ValueError(f"row mismatch after join: before={len(flows):,}, after={len(features):,}")

    out_features = processed_dir / "features.parquet"
    out_lookup = reports_dir / "ip_lookup.csv"
    out_report = reports_dir / "join_report.csv"

    features.to_parquet(out_features, index=False)
    lookup.sort_values("ip").to_csv(out_lookup, index=False)

    pub_mask = lookup["ip_type"] == "PUBLIC" 
    join_rows = [
        {"metric": "rows_before_join", "value": len(flows)},
        {"metric": "rows_after_join", "value": len(features)},
        {"metric": "unique_ips", "value": len(lookup)},
        {"metric": "public_ips", "value": int(pub_mask.sum())},
        {"metric": "public_lookup_found", "value": int((pub_mask & (lookup["country"] != "UNKNOWN")).sum())},
        {"metric": "public_lookup_unknown", "value": int((pub_mask & (lookup["country"] == "UNKNOWN")).sum())},
        {"metric": "private_ips", "value": int((lookup["ip_type"] == "PRIVATE").sum())},
        {"metric": "loopback_ips", "value": int((lookup["ip_type"] == "LOOPBACK").sum())},
        {"metric": "multicast_ips", "value": int((lookup["ip_type"] == "MULTICAST").sum())},
        {"metric": "invalid_or_unknown_ips", "value": int((lookup["ip_type"] == "UNKNOWN").sum())},
    ]

    join_rows.extend(build_side_metrics(flows, lookup, "src"))
    join_rows.extend(build_side_metrics(flows, lookup, "dst"))

    join_report = pd.DataFrame(join_rows)
    join_report.to_csv(out_report, index=False)

    print(f"saved {out_features}")
    print(f"saved {out_lookup}")
    print(f"saved {out_report}")


if __name__ == "__main__":
    main()
