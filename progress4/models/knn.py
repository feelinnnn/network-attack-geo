"""Progress 4 — KNN (Feeling)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_knn.csv          (metrics.py เขียนให้เมื่อเรียก evaluate)
        reports/results_knn_ttl.csv      (รอบมี TTL)
        reports/figures/knn_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_knn.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว ห้ามเอากลับเข้ามาตอนจูน
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="knn_ttl"

รัน (จาก root ของ repo):  python progress4/models/knn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import FEAT, SEED, evaluate, load_split  
