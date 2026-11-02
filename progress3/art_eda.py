"""Progress 3 — EDA (Exploratory Data Analysis) Main Driver

สคริปต์หลักสำหรับประสานงานและเรียกใช้ฟังก์ชันวิเคราะห์ข้อมูลในโฟลเดอร์ EDA/
Input : data/processed/features.parquet  (ผลจาก Progress 2)
Output: reports/figures/art/*.png        (กราฟ 7 ด้าน)
        reports/eda_art.md               (รายงานสรุปข้อค้นพบ)

เกณฑ์:
- ทำ EDA ขั้นต่ำ 5 อย่าง (สคริปต์นี้ทำ 7 อย่างครอบคลุมทุกมิติ)
- แต่ละอย่างตอบโจทย์: "เห็นอะไร -> เลยตัดสินใจ / เข้าใจอะไร"
- บันทึกรูปใน reports/figures/art/
- สรุปข้อค้นพบและเหตุผลการตัดสินใจใน reports/eda_art.md

วิธีรัน (จาก root ของ repo):
    python progress3/eda.py
"""

from pathlib import Path
import sys
import pandas as pd

# กำหนด Path และเพิ่ม directory ของ progress3 ลงใน sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import โมดูลย่อยจากโฟลเดอร์ EDA
from EDA import (
    run_class_distribution,
    run_ttl_leakage,
    run_geo_shortcut,
    run_label_conflicts,
    run_flow_features,
    run_service_protocol,
    run_connection_counts,
)

DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
FIG_DIR = BASE_DIR / "reports" / "figures" / "art"
MD_PATH = BASE_DIR / "reports" / "eda_art.md"


def get_data(columns=None) -> pd.DataFrame:
    """โหลดข้อมูล Parquet จากโฟลเดอร์ processed"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed features not found at {DATA_PATH}. Run Progress 2 first.")
    print(f"[*] Loading dataset from: {DATA_PATH}")
    return pd.read_parquet(DATA_PATH, columns=columns)


# ----------------------------------------------------------------------
# ฟังก์ชันเรียกการวิเคราะห์แต่ละหัวข้อ (Wrapper Functions)
# ----------------------------------------------------------------------

def eda_1_class_distribution(df: pd.DataFrame = None):
    """EDA 1: วิเคราะห์การกระจายตัวของ 10 คลาสเป้าหมายและความไม่สมดุล (Target Class Imbalance)"""
    return run_class_distribution(df=df, fig_dir=FIG_DIR)


def eda_2_ttl_leakage(df: pd.DataFrame = None):
    """EDA 2: วิเคราะห์การรั่วไหลของฟีเจอร์ TTL (sttl, dttl)"""
    return run_ttl_leakage(df=df, fig_dir=FIG_DIR)


def eda_3_geo_shortcut(df: pd.DataFrame = None):
    """EDA 3: วิเคราะห์ความเสี่ยงการจำทางลัดของประเทศ/ASN (Shortcut Learning Risk)"""
    return run_geo_shortcut(df=df, fig_dir=FIG_DIR)


def eda_4_label_conflicts(df: pd.DataFrame = None):
    """EDA 4: วิเคราะห์ปัญหา Label Conflict (ฟีเจอร์ซ้ำซ้อนแต่ต่างคลาส)"""
    conflict_path = BASE_DIR / "reports" / "label_conflicts.csv"
    return run_label_conflicts(df=df, fig_dir=FIG_DIR, conflict_path=conflict_path)


def eda_5_flow_features(df: pd.DataFrame = None):
    """EDA 5: วิเคราะห์ความเบ้ของสถิติโฟลว์ (dur, sbytes, sload) และ Log Transformation"""
    return run_flow_features(df=df, fig_dir=FIG_DIR)


def eda_6_service_protocol(df: pd.DataFrame = None):
    """EDA 6: วิเคราะห์ความสัมพันธ์ระหว่างการโจมตีกับเครือข่ายบริการ (service, proto)"""
    return run_service_protocol(df=df, fig_dir=FIG_DIR)


def eda_7_connection_counts(df: pd.DataFrame = None):
    """EDA 7: วิเคราะห์กลุ่มฟีเจอร์การนับ (Connection Count: ct_*) เทียบกับประเภทการโจมตี"""
    return run_connection_counts(df=df, fig_dir=FIG_DIR)


# ----------------------------------------------------------------------
# ฟังก์ชันรันกระบวนการ EDA ทั้งหมด (Pipeline Orchestrator)
# ----------------------------------------------------------------------

def run_all_eda(df: pd.DataFrame = None):
    """รันกระบวนการ EDA ครบทั้ง 7 ด้านและบันทึกรูปทั้งหมดลงใน reports/figures/art/"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output Figures Directory: {FIG_DIR}")

    # โหลดชุดข้อมูลครั้งเดียวถ้าไม่ได้ส่ง df เข้ามา (ประหยัดเวลาและแรม)
    if df is None:
        df = get_data()
        print(f"[*] Dataset loaded successfully ({len(df):,} rows, {len(df.columns)} columns)")

    print("\n" + "=" * 60)
    print("STARTING FULL EDA PIPELINE (7 TASKS)")
    print("=" * 60)

    # 1. Class Distribution
    eda_1_class_distribution(df)

    # 2. TTL Leakage
    eda_2_ttl_leakage(df)

    # 3. Geo Shortcut
    eda_3_geo_shortcut(df)

    # 4. Label Conflicts
    eda_4_label_conflicts(df)

    # 5. Flow Features
    eda_5_flow_features(df)

    # 6. Service & Protocol
    eda_6_service_protocol(df)

    # 7. Connection Counts
    eda_7_connection_counts(df)

    print("\n" + "=" * 60)
    print("[SUCCESS] All 7 EDA tasks completed successfully!")
    print(f"[+] All figures saved in: {FIG_DIR}")
    print(f"[+] Detailed findings report available at: {MD_PATH}")
    print("=" * 60)


def main():
    """จุดเริ่มต้นทำงานเมื่อรัน python progress3/eda.py"""
    run_all_eda()


if __name__ == "__main__":
    main()
