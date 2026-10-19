"""Progress 1 — Ingest: Raw CSV -> Parquet

รัน (จาก root ของ repo):  python progress1/ingest.py

สิ่งที่ทำ
1. อ่านชื่อคอลัมน์จาก NUSW-NB15_features.csv (ไฟล์ข้อมูลไม่มี header)
2. โหลด UNSW-NB15_1.csv ถึง _4.csv แล้วต่อกัน พร้อมคอลัมน์ source_file
3. บันทึกเป็น data/staging/unsw_flows.parquet

ยังไม่ทำความสะอาดข้อมูล (เป็นงานของ Progress 2)
"""

import pandas as pd

from utils import load_config, p


def read_feature_names(features_path: str) -> list[str]:
    """อ่านชื่อคอลัมน์ 49 ตัวจากไฟล์ features CSV ของ UNSW-NB15"""
    features_df = pd.read_csv(features_path, encoding="cp1252")
    # คอลัมน์ "Name" เก็บชื่อ feature — strip ช่องว่างที่อาจติดมา
    col_names = features_df["Name"].str.strip().tolist()
    return col_names


def load_csv_files(
    raw_dir: str,
    filenames: list[str],
    col_names: list[str],
) -> pd.DataFrame:
    """โหลด CSV หลายไฟล์ แล้ว concat พร้อมเพิ่มคอลัมน์ source_file"""
    frames: list[pd.DataFrame] = []
    for fname in filenames:
        filepath = p(raw_dir) / fname
        print(f"  [loading] {filepath.name} ...")
        df = pd.read_csv(
            filepath,
            header=None,
            names=col_names,
            encoding="utf-8",
            low_memory=False,
        )
        df["source_file"] = fname
        frames.append(df)
        print(f"  [done] {filepath.name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def main() -> None:
    cfg = load_config()

    # --- 1. อ่านชื่อคอลัมน์ ---
    features_path = p(cfg["paths"]["raw_unsw"]) / cfg["unsw"]["features_file"]
    col_names = read_feature_names(str(features_path))
    expected = cfg["unsw"]["expected_columns"]
    assert len(col_names) == expected, (
        f"คาดว่าจะมี {expected} คอลัมน์ แต่พบ {len(col_names)}"
    )
    print(f"[columns] {len(col_names)} columns -- OK")

    # --- 2. โหลด CSV ทั้ง 4 ไฟล์ ---
    print("[load] Loading CSV files ...")
    df = load_csv_files(
        raw_dir=cfg["paths"]["raw_unsw"],
        filenames=cfg["unsw"]["files"],
        col_names=col_names,
    )
    print(f"\n[merged] Total: {len(df):,} rows x {df.shape[1]} columns")

    # --- 3. บันทึก Parquet ---
    # แปลง object columns ที่มี mixed types (เช่น sport มีทั้ง int กับ hex string)
    # ให้เป็น string เพื่อให้ pyarrow เขียนได้
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str)

    staging_dir = p(cfg["paths"]["staging"])
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / "unsw_flows.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n[saved] {out_path}")
    print(f"   Size: {out_path.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
