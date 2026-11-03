"""Progress 4 — Random Forest (Pimmy)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_rf.csv          (metrics.py เขียนให้เมื่อเรียก evaluate)
        reports/results_rf_ttl.csv      (รอบมี TTL)
        reports/figures/rf_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_rf.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="rf_ttl"

รัน (จาก root ของ repo):  python progress4/models/random_forest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import FEAT, SEED, evaluate, load_split  
