# Progress 4 — Model

1 คน 1 โมเดล จูนบน **val** แล้วเลือกค่าที่ดีที่สุดไปวัด **test** ครั้งเดียว

| ไฟล์ | โมเดล | คนทำ |
|------|-------|------|
| `models/logistic_regression.py` | Logistic Regression | Art |
| `models/isolation_forest.py` | Isolation Forest | Art |
| `models/random_forest.py` | Random Forest | Pimmy |
| `models/lightgbm_model.py` | LightGBM | Keen |
| `models/knn.py` | KNN | Feeling |

**Split:** ตามเวลา 70 / 10 / 20 (`split.py`) · seed 42
**Metrics:** Macro F1, PR-AUC, Recall (`metrics.py`) · leakage (มี / ไม่มี TTL) ทำทีหลัง

## เริ่มงาน

```bash
git fetch origin
git switch ชื่อ-branch-ตัวเอง
git merge origin/main
pip install -r requirements.txt
python progress4/split.py
```

## ที่เก็บไฟล์

| อะไร | เก็บที่ |
|------|--------|
| โค้ด | `progress4/models/<โมเดล>.py` |
| ผลคะแนน | `reports/results_<โมเดล>.csv` (`metrics.py` เขียนให้) |
| กราฟ | `reports/figures/<โมเดล>_*.png` |
| สรุป | `reports/model_<โมเดล>.md` |

ห้ามแก้ `split.py`, `metrics.py`
