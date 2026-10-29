"""Progress 2 Check: validate clean/join outputs against acceptance criteria."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config


def to_abs(relative_path: str) -> Path:
    return ROOT / relative_path 


def require_file(path: Path) -> None: # ตรวจสอบว่าไฟล์มีอยู่หรือไม่ ถ้าไม่พบให้แจ้งข้อผิดพลาด
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")


def validate_attack_cat(clean: pd.DataFrame) -> None: # ตรวจสอบว่าคอลัมน์ attack_cat มี 10 class และไม่มีค่า null
    null_count = int(clean["attack_cat"].isna().sum())
    uniq = clean["attack_cat"].astype("string").dropna().nunique()
    if null_count != 0:
        raise ValueError(f"attack_cat has nulls: {null_count}")
    if uniq != 10:
        raise ValueError(f"attack_cat unique count = {uniq}, expected 10")


def validate_no_duplicates(clean: pd.DataFrame) -> None: # ตรวจสอบว่าไม่มีแถวซ้ำกัน
    dup_count = int(clean.duplicated().sum())
    if dup_count != 0:
        raise ValueError(f"duplicate rows found: {dup_count}") # วนตรวจสอบว่ามีแถวซ้ำกันหรือไม่ ถ้ามีให้แจ้งข้อผิดพลาด


def validate_geo_schema(features: pd.DataFrame, geo_cols: list[str]) -> None: # ตรวจสอบว่า features มีคอลัมน์ geo ที่ต้องการและไม่มีค่า null
    missing_geo = [col for col in geo_cols if col not in features.columns]
    if missing_geo:
        raise ValueError(f"missing geo columns: {missing_geo}")

    required_non_null = ["srcip", "dstip", "attack_cat", "label"] + geo_cols # ตรวจสอบว่ามีคอลัมน์ที่ต้องการและไม่มีค่า null
    for col in required_non_null:
        null_count = int(features[col].isna().sum()) 
        if null_count != 0: 
            raise ValueError(f"column '{col}' has nulls: {null_count}")

    for col in ["sport", "dsport"]: # ตรวจสอบว่าคอลัมน์ sport และ dsport เป็น Int64 หรือ int64
        if str(features[col].dtype) not in {"Int64", "int64"}: # สำหรับคอลัมน์ sport และ dsport ต้องเป็น Int64 หรือ int64 ถ้าไม่ใช่ให้แจ้งข้อผิดพลาด
            raise TypeError(f"column '{col}' dtype={features[col].dtype}, expected Int64/int64")


def validate_join_row_count(clean: pd.DataFrame, features: pd.DataFrame) -> None: # ตรวจสอบว่าจำนวนแถวของ clean_flows และ features เท่ากัน
    if len(clean) != len(features):
        raise ValueError(f"row count mismatch: clean={len(clean):,}, features={len(features):,}") # ถ้าจำนวนแถวไม่เท่ากันให้แจ้งข้อผิดพลาด


def validate_lookup_coverage(features: pd.DataFrame, lookup: pd.DataFrame) -> None: # ตรวจสอบว่า ip_lookup.csv ครอบคลุมค่า srcip และ dstip ทั้งหมดใน features
    all_ips = pd.concat(
        [
            features[["srcip"]].rename(columns={"srcip": "ip"}),
            features[["dstip"]].rename(columns={"dstip": "ip"}), # รวมค่า srcip และ dstip ทั้งหมดใน features เป็น DataFrame เดียว โดยเปลี่ยนชื่อคอลัมน์เป็น "ip"
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)

    merged = all_ips.merge(lookup[["ip"]], on="ip", how="left", indicator=True) 
    missing = int((merged["_merge"] == "left_only").sum()) 
    if missing != 0:
        raise ValueError(f"ip_lookup.csv missing IPs: {missing}")


def main() -> None: # ฟังก์ชันหลักสำหรับตรวจสอบไฟล์ clean_flows.parquet, features.parquet และ ip_lookup.csv ว่าตรงตามเกณ
    cfg = load_config()
    features_cfg = load_config("features.yaml")

    processed_dir = to_abs(cfg["paths"]["processed"])
    reports_dir = to_abs(cfg["paths"]["reports"])

    clean_path = processed_dir / "clean_flows.parquet"
    features_path = processed_dir / "features.parquet"
    lookup_path = reports_dir / "ip_lookup.csv"
    cleaning_report_path = reports_dir / "cleaning_report.csv"
    join_report_path = reports_dir / "join_report.csv"

    for path in [clean_path, features_path, lookup_path, cleaning_report_path, join_report_path]:
        require_file(path)

    clean = pd.read_parquet(clean_path)
    features = pd.read_parquet(features_path)
    lookup = pd.read_csv(lookup_path)

    validate_attack_cat(clean)
    validate_no_duplicates(clean)
    validate_geo_schema(features, features_cfg["geo"])
    validate_join_row_count(clean, features)
    validate_lookup_coverage(features, lookup)

    print("[OK] attack_cat has 10 classes and no null")
    print("[OK] no duplicate rows")
    print("[OK] required schema + non-null checks passed")
    print("[OK] rows before/after join are equal")
    print("[OK] ip_lookup.csv covers all srcip/dstip values")
    print("summary")
    print(f"  clean rows: {len(clean):,}")
    print(f"  features rows: {len(features):,}")
    print(f"  attack_cat unique: {clean['attack_cat'].nunique()}")
    print(f"  lookup unique IPs: {lookup['ip'].nunique():,}")


if __name__ == "__main__":
    main()
