"""Progress 4 — Isolation Forest (Art)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_iforest.csv          (metrics.py เขียนให้เมื่อเรียก evaluate_binary)
        reports/figures/iforest_*.png        (กราฟของตัวเอง ถ้ามี)
        reports/model_iforest.md             (สรุปผล)

รัน (จาก root ของ repo):  python progress4/models/isolation_forest.py
"""
# %% [1] Imports & Setup Path
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ค้นหา Root ของ Repo (โฟลเดอร์ที่มี configs/) แบบอัตโนมัติ
current_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
repo_root = next((p for p in [current_dir, *current_dir.parents] if (p / "configs").exists()), Path.cwd())

# เพิ่มโฟลเดอร์ repo root และ progress4 เข้า sys.path
for p in [repo_root, repo_root / "progress4"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metrics import FEAT, SEED, evaluate_binary, load_split

print("✓ โหลด Libraries และตั้งค่าระบบเรียบร้อย")

# ==============================================================================
# PART 1: ISOLATION FOREST (WITHOUT TTL — BASE MODEL)
# ==============================================================================

# %% [2] PART 1: โหลดข้อมูล และเตรียม Preprocessor แบบ Without TTL (38 คอลัมน์)
print("\n" + "=" * 60)
print("PART 1: เริ่มการทำงานรอบ Without TTL (Base Model)")
print("=" * 60)

# 1. โหลดข้อมูลโดยไม่ส่ง extra -> load_split จะดึงเฉพาะ FEAT["base"] (38 คอลัมน์)
print("กำลังโหลดข้อมูล Train และ Val (Base)...")
X_train_base, y_train_base = load_split("train")
X_val_base, y_val_base = load_split("val")

print(f"  - ขนาด X_train_base: {X_train_base.shape} (38 ฟีเจอร์)")
print(f"  - ขนาด X_val_base  : {X_val_base.shape}")

# 2. แยกประเภทคอลัมน์
cat_cols_base = [c for c in FEAT["categorical"] if c in X_train_base.columns]
num_cols_base = [c for c in X_train_base.columns if c not in cat_cols_base]

print(f"  - Categorical ({len(cat_cols_base)} คอลัมน์): {cat_cols_base}")
print(f"  - Numeric ({len(num_cols_base)} คอลัมน์): ฟีเจอร์ Flow พื้นฐาน (ไม่มี TTL)")

# 3. สร้าง Preprocessor และ Pipeline
prep_base = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_base),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_base),
    ]
)

# พารามิเตอร์สำหรับ Isolation Forest
# - contamination: สัดส่วน anomaly ที่คาดว่าจะเจอ (ใน dataset มี attack ~12-13%)
# - max_samples: 256 สุ่มตัวอย่างมาสร้างแต่ละต้นไม้ ช่วยลด masking/swamping และทำให้เร็วมาก
# - n_jobs: -1 ใช้ทุก CPU core
params_base = {
    "n_estimators": 100,
    "max_samples": 256,
    "contamination": 0.13,
    "ttl": False,
}

model_base = Pipeline([
    ("prep", prep_base),
    ("clf", IsolationForest(
        n_estimators=params_base["n_estimators"],
        max_samples=params_base["max_samples"],
        contamination=params_base["contamination"],
        random_state=SEED,
        n_jobs=-1,
    )),
])
print("✓ ประกอบ Pipeline Base Model เรียบร้อย")

# %% [3] PART 1: ฝึกสอนและประเมินผล Without TTL (Base Model)
print("กำลังฝึกสอนโมเดล Isolation Forest (Without TTL)...")
model_base.fit(X_train_base)
print("✓ เทรน Base Model เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
# หมายเหตุสำหรับ Isolation Forest ใน scikit-learn:
# - predict(X) ให้ผลลัพธ์เป็น 1 (Normal) และ -1 (Anomaly/Attack) -> is_attack = (pred == -1)
# - decision_function(X) ยิ่งค่ามากยิ่งเป็นปกติ -> ต้องใส่เครื่องหมายลบ (-) เพื่อให้ 'ยิ่งสูงยิ่งน่าจะโจมตี'
y_pred_val_base = model_base.predict(X_val_base)
is_attack_val_base = (y_pred_val_base == -1)
score_val_base = -model_base.decision_function(X_val_base)

res_val_base = evaluate_binary(
    y_true=y_val_base,
    is_attack=is_attack_val_base,
    score=score_val_base,
    model="iforest",
    split="val",
    params=params_base,
)

# ทำนายและประเมินผลบน Test Set
X_test_base, y_test_base = load_split("test")
y_pred_test_base = model_base.predict(X_test_base)
is_attack_test_base = (y_pred_test_base == -1)
score_test_base = -model_base.decision_function(X_test_base)

res_test_base = evaluate_binary(
    y_true=y_test_base,
    is_attack=is_attack_test_base,
    score=score_test_base,
    model="iforest",
    split="test",
    params=params_base,
)

print(f"✓ บันทึกผล Base ลง reports/results_iforest.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_base['macro_f1']:.4f} | PR-AUC: {res_val_base['pr_auc']:.4f} | Recall: {res_val_base['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_base['macro_f1']:.4f} | PR-AUC: {res_test_base['pr_auc']:.4f} | Recall: {res_test_base['recall']:.4f}")

# ==============================================================================
# PART 2: ISOLATION FOREST (WITH TTL — LEAKAGE TEST)
# ==============================================================================

# %% [4] PART 2: โหลดข้อมูล และเตรียม Preprocessor แบบ With TTL (41 คอลัมน์)
print("\n" + "=" * 60)
print("PART 2: เริ่มการทำงานรอบ With TTL (Data Leakage Test)")
print("=" * 60)

# 1. โหลดข้อมูลพร้อมฟีเจอร์ TTL (extra=FEAT["check"]) รวม 41 คอลัมน์
print("กำลังโหลดข้อมูล Train และ Val (With TTL)...")
X_train_ttl, y_train_ttl = load_split("train", extra=FEAT["check"])
X_val_ttl, y_val_ttl = load_split("val", extra=FEAT["check"])

print(f"  - ขนาด X_train_ttl: {X_train_ttl.shape} (41 ฟีเจอร์: Base + TTL)")
print(f"  - ขนาด X_val_ttl  : {X_val_ttl.shape}")
print(f"  - ฟีเจอร์ TTL ที่เพิ่ม: {FEAT['check']}")

# 2. แยกประเภทคอลัมน์
cat_cols_ttl = [c for c in FEAT["categorical"] if c in X_train_ttl.columns]
num_cols_ttl = [c for c in X_train_ttl.columns if c not in cat_cols_ttl]

# 3. สร้าง Preprocessor และ Pipeline
prep_ttl = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_ttl),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_ttl),
    ]
)

params_ttl = {
    "n_estimators": 100,
    "max_samples": 256,
    "contamination": 0.13,
    "ttl": True,
}

model_ttl = Pipeline([
    ("prep", prep_ttl),
    ("clf", IsolationForest(
        n_estimators=params_ttl["n_estimators"],
        max_samples=params_ttl["max_samples"],
        contamination=params_ttl["contamination"],
        random_state=SEED,
        n_jobs=-1,
    )),
])
print("✓ ประกอบ Pipeline With TTL เรียบร้อย")

# %% [5] PART 2: ฝึกสอนและประเมินผล With TTL (Leakage Test)
print("กำลังฝึกสอนโมเดล Isolation Forest (With TTL)...")
model_ttl.fit(X_train_ttl)
print("✓ เทรน With TTL เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_ttl = model_ttl.predict(X_val_ttl)
is_attack_val_ttl = (y_pred_val_ttl == -1)
score_val_ttl = -model_ttl.decision_function(X_val_ttl)

res_val_ttl = evaluate_binary(
    y_true=y_val_ttl,
    is_attack=is_attack_val_ttl,
    score=score_val_ttl,
    model="iforest_ttl",
    split="val",
    params=params_ttl,
)

# ทำนายและประเมินผลบน Test Set
X_test_ttl, y_test_ttl = load_split("test", extra=FEAT["check"])
y_pred_test_ttl = model_ttl.predict(X_test_ttl)
is_attack_test_ttl = (y_pred_test_ttl == -1)
score_test_ttl = -model_ttl.decision_function(X_test_ttl)

res_test_ttl = evaluate_binary(
    y_true=y_test_ttl,
    is_attack=is_attack_test_ttl,
    score=score_test_ttl,
    model="iforest_ttl",
    split="test",
    params=params_ttl,
)

print(f"✓ บันทึกผล TTL ลง reports/results_iforest_ttl.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_ttl['macro_f1']:.4f} | PR-AUC: {res_val_ttl['pr_auc']:.4f} | Recall: {res_val_ttl['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_ttl['macro_f1']:.4f} | PR-AUC: {res_test_ttl['pr_auc']:.4f} | Recall: {res_test_ttl['recall']:.4f}")

# ==============================================================================
# PART 3: COMPARISON (เปรียบเทียบผลลัพธ์)
# ==============================================================================

# %% [6] PART 3: สรุปตารางเปรียบเทียบผลลัพธ์ (Without TTL vs With TTL)
summary_data = []

if "res_val_base" in locals() and "res_test_base" in locals():
    summary_data.append({
        "Model": "Without TTL (Base)",
        "Features": "38 (No TTL)",
        "Val Macro F1": round(res_val_base["macro_f1"], 4),
        "Val PR-AUC": round(res_val_base["pr_auc"], 4),
        "Val Recall": round(res_val_base["recall"], 4),
        "Test Macro F1": round(res_test_base["macro_f1"], 4),
        "Test PR-AUC": round(res_test_base["pr_auc"], 4),
    })

if "res_val_ttl" in locals() and "res_test_ttl" in locals():
    summary_data.append({
        "Model": "With TTL (Leakage)",
        "Features": "41 (Base + TTL)",
        "Val Macro F1": round(res_val_ttl["macro_f1"], 4),
        "Val PR-AUC": round(res_val_ttl["pr_auc"], 4),
        "Val Recall": round(res_val_ttl["recall"], 4),
        "Test Macro F1": round(res_test_ttl["macro_f1"], 4),
        "Test PR-AUC": round(res_test_ttl["pr_auc"], 4),
    })

df_summary = pd.DataFrame(summary_data)

print("\n" + "=" * 70)
print("ตารางเปรียบเทียบผลลัพธ์ Isolation Forest (Leakage Effect):")
print("=" * 70)
print(df_summary.to_string(index=False))
print("=" * 70)

# %%
