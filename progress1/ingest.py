"""Progress 1 — Ingest: Raw -> Parquet

รัน (จาก root ของ repo):  python progress1/ingest.py

สิ่งที่ทำ
1. อ่านชื่อคอลัมน์จาก NUSW-NB15_features.csv (ไฟล์ข้อมูลไม่มี header)
2. โหลด UNSW-NB15_1.csv ถึง _4.csv แล้วต่อกัน พร้อมคอลัมน์ source_file
3. โหลด ProdIPData (เฉพาะคอลัมน์ใน config) แล้วตรวจว่าเป็น snapshot เดียวและ prefix ไม่ซ้ำ
4. บันทึกเป็น data/staging/unsw_flows.parquet และ data/staging/prodipdata.parquet
5. ตรวจจำนวนแถวกับค่าที่คาดไว้ใน config (ทุกคนต้องได้ตรงกัน)

ยังไม่ทำความสะอาดข้อมูล (เป็นงานของ Progress 2)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo


# UNSW-NB15
def read_column_names(features_path: Path) -> list[str]:  # ฟังก์ชันอ่านชื่อคอลัมน์ คืนค่าเป็น list ของชื่อ
    feat = pd.read_csv(features_path, encoding="latin-1")  # ไฟล์นี้ไม่ใช่ UTF-8
    feat.columns = feat.columns.str.strip()  # ตัดช่องว่าง
    return (
        feat["Name"].str.strip().str.lower()  # เอาคอลัมน์ Name -> ตัดช่องว่างหัวท้าย -> ทำเป็นตัวพิมพ์เล็ก
        .str.replace(" ", "", regex=False)  # ลบช่องว่างที่อยู่กลางชื่อ
        .tolist()
    )


def load_unsw(cfg: dict) -> pd.DataFrame:  # โหลด UNSW
    raw_dir = p(cfg["paths"]["raw_unsw"])
    cols = read_column_names(raw_dir / cfg["unsw"]["features_file"])
    expected = cfg["unsw"]["expected_columns"]  # จำนวนคอลัมน์ที่ควรได้ (49) จาก config

    if len(cols) != expected:  # ถ้าได้ชื่อคอลัมน์ไม่ครบหรือเกิน
        raise ValueError(f"got {len(cols)} columns, expected {expected}")

    parts = []  # list เก็บตารางของแต่ละไฟล์ ไว้ต่อกัน
    for i, name in enumerate(cfg["unsw"]["files"], start=1):  # วนทีละไฟล์
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")
        df = pd.read_csv(
            path,
            header=None,  # ไฟล์ไม่มีหัวตาราง แถวแรกคือข้อมูลเลย
            names=cols,  # ใส่ชื่อคอลัมน์ 49 ตัวที่อ่านมา
            encoding="utf-8-sig",
            dtype={"srcip": str, "dstip": str, "sport": str, "dsport": str,  # บังคับ 6 คอลัมน์นี้เป็นข้อความ
                   "ct_ftp_cmd": str, "attack_cat": str},                     # (มีค่าแปลก ๆ เช่น 0xcc09, "-", " ")
            low_memory=False,  # อ่านทั้งไฟล์ก่อนเดาชนิดข้อมูล กัน data type ปนกัน
        )
        print(f"  {name}: {len(df):,} rows, {len(df.columns):,} columns")
        df["source_file"] = i  # เพิ่มคอลัมน์บอกว่าแถวนี้มาจากไฟล์ที่เท่าไร (1-4)
        parts.append(df)

    flows = pd.concat(parts, ignore_index=True)  # ต่อ 4 ตารางเป็นตารางเดียว และเรียงเลขแถวใหม่
    check_mixed_types(flows)  # ตรวจว่าไม่มีคอลัมน์ไหนชนิดข้อมูลปนกัน
    return flows


def check_mixed_types(df: pd.DataFrame) -> None:  # ฟังก์ชันตรวจชนิดข้อมูลปนกัน
    for col in df.columns:
        if df[col].dtype == object:  # ตรวจเฉพาะคอลัมน์ object (คอลัมน์ที่อาจมีหลายชนิดปนกัน)
            kinds = df[col].dropna().map(type).unique()  # ตัดค่าว่างออก แล้วดูว่ามีชนิดข้อมูลอะไรบ้าง
            if len(kinds) > 1:
                raise TypeError(f"column '{col}' has mixed types {[k.__name__ for k in kinds]}")


# ProdIPData
def load_prodipdata(cfg: dict) -> pd.DataFrame:  # โหลด ProdIPData
    pc = cfg["prodipdata"]
    path = p(cfg["paths"]["raw_prodipdata"]) / pc["file"]
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}\ndownload from {pc['source']}")

    ip = pd.read_parquet(path, columns=pc["columns"], dtype_backend="pyarrow")  # อ่านเฉพาะคอลัมน์ที่ใช้
    print(f"  {path.name}: {len(ip):,} rows, {len(ip.columns):,} columns")

    months = ip["snapshot_month"].unique().tolist()
    if months != [pc["snapshot_month"]]:  # ไฟล์ต้องเป็น snapshot เดือนที่ตกลงกันเท่านั้น
        raise ValueError(f"snapshot_month = {months}, expected ['{pc['snapshot_month']}']")
    dup = ip["IP24Prefix"].duplicated().sum()  # 1 prefix ต้องมี 1 แถว ไม่งั้น join แล้วแถวจะเพิ่ม
    if dup:
        raise ValueError(f"duplicate IP24Prefix: {dup:,} rows")
    return ip


def check_expected(flows: pd.DataFrame, ip: pd.DataFrame, cfg: dict) -> None: # ตัวเลขที่ทุกคนต้องได้ตรงกัน ถ้าไม่ตรงแปลว่าไฟล์ดิบไม่ครบหรืออ่านผิด
    checks = {
        "unsw rows": (len(flows), cfg["unsw"]["expected_rows"]),
        "unsw label": (flows["label"].value_counts().sort_index().to_dict(), cfg["unsw"]["expected_label"]),
        "prodipdata rows": (len(ip), cfg["prodipdata"]["expected_rows"]),
    }
    for name, (got, want) in checks.items():
        status = "OK " if got == want else "FAIL"
        print(f"  [{status}] {name}: {got}" + ("" if got == want else f" (expected {want})"))
    if any(got != want for got, want in checks.values()):
        raise SystemExit("ERROR: check_expected failed")


def summarize(flows: pd.DataFrame) -> None:
    print("\n-- summarize --")
    print(f"1. rows    : {len(flows):,}")  # 2,540,047
    print(f"2. columns : {flows.shape[1] - 1} (excluding source_file)")  # 49
    print(f"3. label 0 / 1 : {flows['label'].value_counts().sort_index().to_dict()}")  # {0: 2218764, 1: 321283}
    # ค่าดิบ: 13 ค่าที่ไม่ว่าง (มีช่องว่างหน้า/หลัง และ Backdoor/Backdoors) + ค่าว่าง 2,218,764 แถว (= Normal)
    print(f"4. attack_cat  :\n{flows['attack_cat'].value_counts(dropna=False).to_string()}")


def main() -> None:
    cfg = load_config()
    staging = p(cfg["paths"]["staging"])
    staging.mkdir(parents=True, exist_ok=True)

    print("loading UNSW-NB15")
    flows = load_unsw(cfg)
    flows.to_parquet(staging / "unsw_flows.parquet", index=False)
    print("  saved data/staging/unsw_flows.parquet")

    print("loading ProdIPData")
    ip = load_prodipdata(cfg)
    ip.to_parquet(staging / "prodipdata.parquet", index=False)
    print("  saved data/staging/prodipdata.parquet")

    print("\n-- check --")
    check_expected(flows, ip, cfg)
    summarize(flows)


if __name__ == "__main__":
    main()
