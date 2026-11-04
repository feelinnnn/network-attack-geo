"""Progress 4 — LightGBM (Keen)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_lgbm.csv          (metrics.py เขียนให้เมื่อเรียก evaluate)
        reports/results_lgbm_ttl.csv      (รอบมี TTL)
        reports/figures/lgbm_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_lgbm.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว ห้ามเอากลับเข้ามาตอนจูน
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="lgbm_ttl"

รัน (จาก root ของ repo):  python progress4/models/lightgbm_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import FEAT, SEED, evaluate, load_split  
