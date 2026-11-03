"""Progress 4 — แบ่ง train / val / test ตามเวลา (ทุกคนใช้ไฟล์ตัวเดียวกัน ห้ามแก้)

Input : data/processed/features.parquet   (ผลจาก Progress 2)
Output: data/processed/train.parquet      70% แรกตามเวลา   (22 ม.ค. 22:49 -> 18 ก.พ. ~16:36)
        data/processed/val.parquet        10% ถัดไป        (18 ก.พ. ~16:36 -> ~18:50) ใช้จูน
        data/processed/test.parquet       20% สุดท้าย       (18 ก.พ. ~18:50 -> 23:21) ใช้วัดผลครั้งเดียว
        reports/split_report.csv          ช่วงเวลา / จำนวนแถว / % โจมตี / จำนวนแต่ละคลาส ของแต่ละชุด

แบ่งตามเวลา (ไม่สุ่ม)
- EDA เจอ drift: อัตราการโจมตีรอบ 1 = 1.4% รอบ 2 = 8.2% -> ต้องเรียนจากอดีตแล้วทายอนาคตแบบของจริง
- 70/10/20: test ใหญ่ คะแนนสุดท้ายนิ่ง (Worms ใน test 56 แถว) ส่วน val ยังครบ 10 คลาส

ต้องตรวจ
1. ทุกชุดมีครบ 10 คลาส
2. ช่วงเวลาไม่ทับกัน (train < val < test)
3. แถวรวม 3 ชุด = แถวทั้งหมด

รัน (จาก root ของ repo):  python progress4/split.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo

RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}  # สัดส่วนตามลำดับเวลา
N_CLASSES = 10
TZ = "Australia/Sydney"  # ใช้แค่ตอนพิมพ์ / รายงานช่วงเวลา


# หาจุดตัดเป็น "ค่าเวลา" ไม่ใช่เลขแถว -> แถวที่ stime เท่ากันจะอยู่ชุดเดียวกันเสมอ
def time_cuts(stime: pd.Series) -> tuple[int, int]:
    s = stime.sort_values(kind="stable").reset_index(drop=True)
    c1 = int(s.iloc[int(len(s) * RATIOS["train"])])                       # เวลาที่ตำแหน่ง 70%
    c2 = int(s.iloc[int(len(s) * (RATIOS["train"] + RATIOS["val"]))])     # เวลาที่ตำแหน่ง 80%
    return c1, c2


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c1, c2 = time_cuts(df["stime"])
    parts = {
        "train": df[df["stime"] < c1],                         # ก่อนจุดตัดแรก
        "val": df[(df["stime"] >= c1) & (df["stime"] < c2)],   # ระหว่างสองจุดตัด
        "test": df[df["stime"] >= c2],                         # ตั้งแต่จุดตัดที่สอง
    }
    return {k: v.sort_values("stime", kind="stable").reset_index(drop=True) for k, v in parts.items()}


# 1 แถวต่อ 1 ชุด: ช่วงเวลา, จำนวนแถว, % โจมตี, จำนวนแต่ละคลาส
def report(parts: dict[str, pd.DataFrame], classes: list[str]) -> pd.DataFrame:
    rows = []
    for name, d in parts.items():
        t = pd.to_datetime(d["stime"], unit="s", utc=True).dt.tz_convert(TZ)
        row = {"split": name, "start": f"{t.min():%Y-%m-%d %H:%M}", "end": f"{t.max():%Y-%m-%d %H:%M}",
               "rows": len(d), "attack_%": round(100 * d["label"].mean(), 2)}
        row.update(d["attack_cat"].value_counts().reindex(classes, fill_value=0).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    proc, rep = p(cfg["paths"]["processed"]), p(cfg["paths"]["reports"])
    src = proc / "features.parquet"
    if not src.exists():
        raise FileNotFoundError(f"not found: {src} (run progress2 first)")

    print("loading features.parquet")
    df = pd.read_parquet(src)
    df["attack_cat"] = df["attack_cat"].astype(str)
    classes = df["attack_cat"].value_counts().index.tolist()  # เรียงจากมากไปน้อย

    print("split 70 / 10 / 20 by time")
    parts = split(df)

    # ตรวจ
    missing = {k: sorted(set(classes) - set(d["attack_cat"])) for k, d in parts.items()}
    if any(missing.values()):  # ทุกชุดต้องครบ 10 คลาส
        raise ValueError(f"some splits miss classes: {missing}")
    if not (parts["train"]["stime"].max() < parts["val"]["stime"].min()
            and parts["val"]["stime"].max() < parts["test"]["stime"].min()):  # ห้ามทับเวลากัน
        raise ValueError("time ranges overlap")
    if sum(len(d) for d in parts.values()) != len(df):  # แถวต้องไม่หาย / ไม่ซ้ำ
        raise ValueError("row count changed")
    if len(classes) != N_CLASSES:
        raise ValueError(f"attack_cat has {len(classes)} classes, expected {N_CLASSES}")

    for name, d in parts.items():
        d.to_parquet(proc / f"{name}.parquet", index=False)
    rpt = report(parts, classes)
    rpt.to_csv(rep / "split_report.csv", index=False, encoding="utf-8-sig")

    print("saved -> data/processed/train.parquet, val.parquet, test.parquet")
    print("saved -> reports/split_report.csv\n")
    print(rpt.set_index("split").T.to_string())


if __name__ == "__main__":
    main()
