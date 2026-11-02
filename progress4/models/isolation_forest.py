"""Progress 4 — Isolation Forest (Art)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_iforest.csv          (metrics.py เขียนให้เมื่อเรียก evaluate_binary)
        reports/figures/iforest_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_iforest.md             (สรุปผล)

รัน (จาก root ของ repo):  python progress4/models/isolation_forest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import SEED, evaluate, load_split  
