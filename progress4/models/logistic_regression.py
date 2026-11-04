"""Progress 4 — Logistic Regression (Art)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_logreg.csv          (metrics.py เขียนให้เมื่อเรียก evaluate)
        reports/results_logreg_ttl.csv      (รอบมี TTL)
        reports/figures/logreg_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_logreg.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว ห้ามเอากลับเข้ามาตอนจูน
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="logreg_ttl"

รัน (จาก root ของ repo):  python progress4/models/logistic_regression.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import FEAT, SEED, evaluate, load_split  
