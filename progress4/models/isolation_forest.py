"""Progress 4 — Isolation Forest (Art)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_iforest.csv          (รอบ 1: Base — Without TTL / Without Geo)
        reports/results_iforest_ttl.csv      (รอบ 2: With TTL — Leakage Test)
        reports/results_iforest_geo.csv      (รอบ 3: Base + Geo — Shortcut Learning Test)
        reports/results_iforest_ttl_geo.csv  (รอบ 4: With TTL + Geo — Full Features)
        reports/figures/iforest_*.png        (กราฟของตัวเอง)
        reports/model_iforest.md             (สรุปผล)

รัน (จาก root ของ repo):  python progress4/models/isolation_forest.py
"""
# %% [1] Imports & Setup Path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ค้นหา Root ของ Repo (โฟลเดอร์ที่มี configs/) แบบอัตโนมัติ
current_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
repo_root = next((p for p in [current_dir, *current_dir.parents] if (p / "configs").exists()), Path.cwd())

# เพิ่มโฟลเดอร์ repo root และ progress4 เข้า sys.path
for p in [repo_root, repo_root / "progress4"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metrics import FEAT, SEED, evaluate_binary, load_split

print("✓ โหลด Libraries และตั้งค่าระบบ Isolation Forest เรียบร้อย")

# ==============================================================================
# PART 1: ISOLATION FOREST (WITHOUT TTL / WITHOUT GEO — BASE MODEL)
# ==============================================================================

# %% [2] PART 1: โหลดข้อมูล และเตรียม Preprocessor แบบ Base (38 คอลัมน์)
print("\n" + "=" * 60)
print("PART 1: เริ่มการทำงานรอบ Without TTL (Base Model)")
print("=" * 60)

# 1. โหลดข้อมูลโดยไม่ส่ง extra -> load_split จะดึงเฉพาะ FEAT["base"] (38 คอลัมน์)
print("กำลังโหลดข้อมูล Train และ Val (Base)...")
X_train_base, y_train_base = load_split("train")
X_val_base, y_val_base = load_split("val")

print(f"  - ขนาด X_train_base: {X_train_base.shape} (38 ฟีเจอร์ Flow ล้วน)")
print(f"  - ขนาด X_val_base  : {X_val_base.shape}")

# 2. แยกประเภทคอลัมน์ และแปลง Categorical เป็น string
cat_cols_base = [c for c in FEAT["categorical"] if c in X_train_base.columns]
for c in cat_cols_base:
    X_train_base[c] = X_train_base[c].astype(str)
    X_val_base[c] = X_val_base[c].astype(str)

num_cols_base = [c for c in X_train_base.columns if c not in cat_cols_base]

print(f"  - Categorical ({len(cat_cols_base)} คอลัมน์): {cat_cols_base}")
print(f"  - Numeric ({len(num_cols_base)} คอลัมน์): ฟีเจอร์ Flow พื้นฐาน")

# 3. สร้าง Preprocessor และ Pipeline
prep_base = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_base),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_base),
    ]
)

# ------------------------------------------------------------------------------
# พารามิเตอร์หลักสำหรับจูน Isolation Forest:
# - contamination: สัดส่วน anomaly ที่คาดว่าจะเจอในข้อมูล (เช่น 0.05, 0.08, 0.10, 0.13)
# - max_samples: จำนวนจุดที่สุ่มมาสร้างแต่ละต้นไม้ (128, 256, 512) ป้องกัน masking/swamping
# - max_features: สัดส่วนฟีเจอร์ที่สุ่มใช้ในแต่ละต้นไม้ (Subspace Sampling เช่น 0.8, 1.0)
# - n_estimators: จำนวนต้นไม้ใน ensemble (100, 200)
# ------------------------------------------------------------------------------
params_base = {
    "n_estimators": 100,
    "max_samples": 256,
    "contamination": 0.13,
    "max_features": 1.0,
    "ttl": False,
    "geo": False,
}

model_base = Pipeline([
    ("prep", prep_base),
    ("clf", IsolationForest(
        n_estimators=params_base["n_estimators"],
        max_samples=params_base["max_samples"],
        contamination=params_base["contamination"],
        max_features=params_base["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )),
])
print("✓ ประกอบ Pipeline Base Model เรียบร้อย")

# %% [3] PART 1: ฝึกสอนและประเมินผล Without TTL (Base Model)
print("กำลังฝึกสอนโมเดล Isolation Forest (Base Model)...")
model_base.fit(X_train_base)
print("✓ เทรน Base Model เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
# ใน scikit-learn:
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
for c in cat_cols_base:
    X_test_base[c] = X_test_base[c].astype(str)

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

# %% [3B] (เสริม) รันจูนค่า Contamination Grid อัตโนมัติบน Validation Set
# วนลูปทดลองค่า contamination ต่างๆ เพื่อดูผลต่อ Precision-Recall Trade-off
contamination_candidates = [0.03, 0.05, 0.08, 0.10, 0.13, 0.15]
grid_results_iforest = []

print("กำลังรันการทดลองจูนค่า contamination บน Validation Set...")
# คำนวณ anomaly score ล่วงหน้าครั้งเดียว (ไม่ต้อง fit ใหม่ เพราะ score ไม่เปลี่ยนตาม contamination)
for cont in contamination_candidates:
    p_cand = {**params_base, "contamination": cont}
    
    # คำนวณเกณฑ์ threshold จาก quantile ของ anomaly score
    threshold = np.percentile(score_val_base, 100 * (1 - cont))
    is_attack_cand = score_val_base >= threshold
    
    res = evaluate_binary(
        y_true=y_val_base,
        is_attack=is_attack_cand,
        score=score_val_base,
        model="iforest",
        split="val",
        params=p_cand,
    )
    grid_results_iforest.append({
        "contamination": cont,
        "Val Macro F1": round(res["macro_f1"], 4),
        "Val PR-AUC": round(res["pr_auc"], 4),
        "Val Recall": round(res["recall"], 4),
    })

print("\n" + "=" * 65)
print("ตารางเปรียบเทียบผลกระทบของค่า contamination บน Validation Set:")
print("=" * 65)
print(pd.DataFrame(grid_results_iforest).to_string(index=False))
print("=" * 65)

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
for c in cat_cols_ttl:
    X_train_ttl[c] = X_train_ttl[c].astype(str)
    X_val_ttl[c] = X_val_ttl[c].astype(str)

num_cols_ttl = [c for c in X_train_ttl.columns if c not in cat_cols_ttl]

# 3. สร้าง Preprocessor และ Pipeline
prep_ttl = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_ttl),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_ttl),
    ]
)

params_ttl = {
    **params_base,
    "ttl": True,
    "geo": False,
}

model_ttl = Pipeline([
    ("prep", prep_ttl),
    ("clf", IsolationForest(
        n_estimators=params_ttl["n_estimators"],
        max_samples=params_ttl["max_samples"],
        contamination=params_ttl["contamination"],
        max_features=params_ttl["max_features"],
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
for c in cat_cols_ttl:
    X_test_ttl[c] = X_test_ttl[c].astype(str)

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
# PART 3: ISOLATION FOREST (WITH GEO — SHORTCUT LEARNING TEST)
# ==============================================================================

# %% [6] PART 3: โหลดข้อมูล และเตรียม Preprocessor แบบ Base + Geo (42 คอลัมน์)
print("\n" + "=" * 60)
print("PART 3: เริ่มการทำงานรอบ With Geo (Shortcut Learning Test)")
print("=" * 60)

# 1. โหลดข้อมูลพร้อมฟีเจอร์ Geo (extra=FEAT["geo"]) รวม 42 คอลัมน์
print("กำลังโหลดข้อมูล Train และ Val (With Geo)...")
X_train_geo, y_train_geo = load_split("train", extra=FEAT["geo"])
X_val_geo, y_val_geo = load_split("val", extra=FEAT["geo"])

print(f"  - ขนาด X_train_geo: {X_train_geo.shape} (42 ฟีเจอร์: Base + Geo)")
print(f"  - ขนาด X_val_geo  : {X_val_geo.shape}")
print(f"  - ฟีเจอร์ Geo ที่เพิ่ม: {FEAT['geo']}")

# 2. แยกประเภทคอลัมน์ และแปลง Categorical เป็น string
cat_cols_geo = [c for c in FEAT["categorical"] if c in X_train_geo.columns]
for c in cat_cols_geo:
    X_train_geo[c] = X_train_geo[c].astype(str)
    X_val_geo[c] = X_val_geo[c].astype(str)

num_cols_geo = [c for c in X_train_geo.columns if c not in cat_cols_geo]

# 3. สร้าง Preprocessor และ Pipeline
prep_geo = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_geo),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_geo),
    ]
)

params_geo = {
    **params_base,
    "ttl": False,
    "geo": True,
}

model_geo = Pipeline([
    ("prep", prep_geo),
    ("clf", IsolationForest(
        n_estimators=params_geo["n_estimators"],
        max_samples=params_geo["max_samples"],
        contamination=params_geo["contamination"],
        max_features=params_geo["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )),
])
print("✓ ประกอบ Pipeline With Geo เรียบร้อย")

# %% [7] PART 3: ฝึกสอนและประเมินผล With Geo (Shortcut Learning Test)
print("กำลังฝึกสอนโมเดล Isolation Forest (With Geo)...")
model_geo.fit(X_train_geo)
print("✓ เทรน With Geo เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_geo = model_geo.predict(X_val_geo)
is_attack_val_geo = (y_pred_val_geo == -1)
score_val_geo = -model_geo.decision_function(X_val_geo)

res_val_geo = evaluate_binary(
    y_true=y_val_geo,
    is_attack=is_attack_val_geo,
    score=score_val_geo,
    model="iforest_geo",
    split="val",
    params=params_geo,
)

# ทำนายและประเมินผลบน Test Set
X_test_geo, y_test_geo = load_split("test", extra=FEAT["geo"])
for c in cat_cols_geo:
    X_test_geo[c] = X_test_geo[c].astype(str)

y_pred_test_geo = model_geo.predict(X_test_geo)
is_attack_test_geo = (y_pred_test_geo == -1)
score_test_geo = -model_geo.decision_function(X_test_geo)

res_test_geo = evaluate_binary(
    y_true=y_test_geo,
    is_attack=is_attack_test_geo,
    score=score_test_geo,
    model="iforest_geo",
    split="test",
    params=params_geo,
)

print(f"✓ บันทึกผล Geo ลง reports/results_iforest_geo.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_geo['macro_f1']:.4f} | PR-AUC: {res_val_geo['pr_auc']:.4f} | Recall: {res_val_geo['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_geo['macro_f1']:.4f} | PR-AUC: {res_test_geo['pr_auc']:.4f} | Recall: {res_test_geo['recall']:.4f}")

# ==============================================================================
# PART 4: ISOLATION FOREST (WITH TTL + GEO — FULL ABLATION TEST)
# ==============================================================================

# %% [8] PART 4: โหลดข้อมูล และเตรียม Preprocessor แบบ With TTL + Geo (45 คอลัมน์)
print("\n" + "=" * 60)
print("PART 4: เริ่มการทำงานรอบ With TTL + Geo (Full Features Test)")
print("=" * 60)

# 1. โหลดข้อมูลพร้อมทั้ง TTL และ Geo (extra=FEAT["check"] + FEAT["geo"]) รวม 45 คอลัมน์
print("กำลังโหลดข้อมูล Train และ Val (With TTL + Geo)...")
extra_ttl_geo = FEAT["check"] + FEAT["geo"]
X_train_ttl_geo, y_train_ttl_geo = load_split("train", extra=extra_ttl_geo)
X_val_ttl_geo, y_val_ttl_geo = load_split("val", extra=extra_ttl_geo)

print(f"  - ขนาด X_train_ttl_geo: {X_train_ttl_geo.shape} (45 ฟีเจอร์: Base + TTL + Geo)")
print(f"  - ขนาด X_val_ttl_geo  : {X_val_ttl_geo.shape}")

# 2. แยกประเภทคอลัมน์ และแปลง Categorical เป็น string
cat_cols_ttl_geo = [c for c in FEAT["categorical"] if c in X_train_ttl_geo.columns]
for c in cat_cols_ttl_geo:
    X_train_ttl_geo[c] = X_train_ttl_geo[c].astype(str)
    X_val_ttl_geo[c] = X_val_ttl_geo[c].astype(str)

num_cols_ttl_geo = [c for c in X_train_ttl_geo.columns if c not in cat_cols_ttl_geo]

# 3. สร้าง Preprocessor และ Pipeline
prep_ttl_geo = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_ttl_geo),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_ttl_geo),
    ]
)

params_ttl_geo = {
    **params_base,
    "ttl": True,
    "geo": True,
}

model_ttl_geo = Pipeline([
    ("prep", prep_ttl_geo),
    ("clf", IsolationForest(
        n_estimators=params_ttl_geo["n_estimators"],
        max_samples=params_ttl_geo["max_samples"],
        contamination=params_ttl_geo["contamination"],
        max_features=params_ttl_geo["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )),
])
print("✓ ประกอบ Pipeline With TTL + Geo เรียบร้อย")

# %% [9] PART 4: ฝึกสอนและประเมินผล With TTL + Geo (Full Features Test)
print("กำลังฝึกสอนโมเดล Isolation Forest (With TTL + Geo)...")
model_ttl_geo.fit(X_train_ttl_geo)
print("✓ เทรน With TTL + Geo เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_ttl_geo = model_ttl_geo.predict(X_val_ttl_geo)
is_attack_val_ttl_geo = (y_pred_val_ttl_geo == -1)
score_val_ttl_geo = -model_ttl_geo.decision_function(X_val_ttl_geo)

res_val_ttl_geo = evaluate_binary(
    y_true=y_val_ttl_geo,
    is_attack=is_attack_val_ttl_geo,
    score=score_val_ttl_geo,
    model="iforest_ttl_geo",
    split="val",
    params=params_ttl_geo,
)

# ทำนายและประเมินผลบน Test Set
X_test_ttl_geo, y_test_ttl_geo = load_split("test", extra=extra_ttl_geo)
for c in cat_cols_ttl_geo:
    X_test_ttl_geo[c] = X_test_ttl_geo[c].astype(str)

y_pred_test_ttl_geo = model_ttl_geo.predict(X_test_ttl_geo)
is_attack_test_ttl_geo = (y_pred_test_ttl_geo == -1)
score_test_ttl_geo = -model_ttl_geo.decision_function(X_test_ttl_geo)

res_test_ttl_geo = evaluate_binary(
    y_true=y_test_ttl_geo,
    is_attack=is_attack_test_ttl_geo,
    score=score_test_ttl_geo,
    model="iforest_ttl_geo",
    split="test",
    params=params_ttl_geo,
)

print(f"✓ บันทึกผล TTL + Geo ลง reports/results_iforest_ttl_geo.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_ttl_geo['macro_f1']:.4f} | PR-AUC: {res_val_ttl_geo['pr_auc']:.4f} | Recall: {res_val_ttl_geo['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_ttl_geo['macro_f1']:.4f} | PR-AUC: {res_test_ttl_geo['pr_auc']:.4f} | Recall: {res_test_ttl_geo['recall']:.4f}")

# ==============================================================================
# PART 5: COMPARISON (เปรียบเทียบผลลัพธ์ 4 มิติแบบ 2x2 Factorial Analysis)
# ==============================================================================

# %% [10] PART 5: สรุปตารางเปรียบเทียบผลลัพธ์ทุกรูปแบบ (Base vs TTL vs Geo vs TTL+Geo)
summary_data = []

if "res_val_base" in locals() and "res_test_base" in locals():
    summary_data.append({
        "Model": "1. Base (Clean)",
        "Features": "38 (Flow Only)",
        "TTL": "No",
        "Geo": "No",
        "Val Macro F1": round(res_val_base["macro_f1"], 4),
        "Val PR-AUC": round(res_val_base["pr_auc"], 4),
        "Val Recall": round(res_val_base["recall"], 4),
        "Test Macro F1": round(res_test_base["macro_f1"], 4),
        "Test PR-AUC": round(res_test_base["pr_auc"], 4),
    })

if "res_val_ttl" in locals() and "res_test_ttl" in locals():
    summary_data.append({
        "Model": "2. With TTL (Leakage)",
        "Features": "41 (Base + 3 TTL)",
        "TTL": "Yes",
        "Geo": "No",
        "Val Macro F1": round(res_val_ttl["macro_f1"], 4),
        "Val PR-AUC": round(res_val_ttl["pr_auc"], 4),
        "Val Recall": round(res_val_ttl["recall"], 4),
        "Test Macro F1": round(res_test_ttl["macro_f1"], 4),
        "Test PR-AUC": round(res_test_ttl["pr_auc"], 4),
    })

if "res_val_geo" in locals() and "res_test_geo" in locals():
    summary_data.append({
        "Model": "3. Base + Geo (Shortcut)",
        "Features": "42 (Base + 4 Geo)",
        "TTL": "No",
        "Geo": "Yes",
        "Val Macro F1": round(res_val_geo["macro_f1"], 4),
        "Val PR-AUC": round(res_val_geo["pr_auc"], 4),
        "Val Recall": round(res_val_geo["recall"], 4),
        "Test Macro F1": round(res_test_geo["macro_f1"], 4),
        "Test PR-AUC": round(res_test_geo["pr_auc"], 4),
    })

if "res_val_ttl_geo" in locals() and "res_test_ttl_geo" in locals():
    summary_data.append({
        "Model": "4. With TTL + Geo (Full)",
        "Features": "45 (Base + TTL + Geo)",
        "TTL": "Yes",
        "Geo": "Yes",
        "Val Macro F1": round(res_val_ttl_geo["macro_f1"], 4),
        "Val PR-AUC": round(res_val_ttl_geo["pr_auc"], 4),
        "Val Recall": round(res_val_ttl_geo["recall"], 4),
        "Test Macro F1": round(res_test_ttl_geo["macro_f1"], 4),
        "Test PR-AUC": round(res_test_ttl_geo["pr_auc"], 4),
    })

df_summary = pd.DataFrame(summary_data)

print("\n" + "=" * 95)
print("ตารางเปรียบเทียบผลลัพธ์ Isolation Forest (2x2 Factorial: Base vs TTL vs Geo vs TTL+Geo):")
print("=" * 95)
print(df_summary.to_string(index=False))
print("=" * 95)

# ==============================================================================
# PART 6: VISUALIZATIONS (สร้างรูปภาพสำหรับรายงาน Progress 4)
# ==============================================================================

# %% [11] PART 6: สร้างกราฟผลลัพธ์ลง reports/figures/
print("\n" + "=" * 60)
print("PART 6: กำลังสร้างกราฟสรุปผลลัพธ์ลง reports/figures/...")
print("=" * 60)

fig_dir = repo_root / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# กราฟที่ 1: การแจกแจงของ Anomaly Score (Normal vs Attack Distribution)
# ------------------------------------------------------------------------------
if "score_test_base" in locals() and "y_test_base" in locals():
    is_att_true = (y_test_base != "Normal")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(score_test_base[~is_att_true], label="Normal Traffic", color="#1976D2", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(score_test_base[is_att_true], label="Attack Traffic", color="#D32F2F", fill=True, alpha=0.3, ax=ax)
    ax.set_title("Isolation Forest (Base) — Anomaly Score Distribution (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Anomaly Score (-Decision Function, Higher = More Anomaly)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    score_dist_path = fig_dir / "iforest_score_distribution.png"
    plt.savefig(score_dist_path, dpi=300)
    plt.close()
    print(f"  ✓ บันทึกกราฟ Anomaly Score Distribution: {score_dist_path.name}")

# ------------------------------------------------------------------------------
# กราฟที่ 2: เปรียบเทียบผลกระทบของ 4 โมเดล (Base vs TTL vs Geo vs TTL+Geo)
# ------------------------------------------------------------------------------
if "res_val_base" in locals() and "res_val_ttl" in locals():
    metrics_names = ["Macro F1", "PR-AUC", "Recall"]
    base_scores = [res_val_base["macro_f1"], res_val_base["pr_auc"], res_val_base["recall"]]
    ttl_scores = [res_val_ttl["macro_f1"], res_val_ttl["pr_auc"], res_val_ttl["recall"]]

    x = np.arange(len(metrics_names))
    
    if "res_val_geo" in locals() and "res_val_ttl_geo" in locals():
        width = 0.2
        geo_scores = [res_val_geo["macro_f1"], res_val_geo["pr_auc"], res_val_geo["recall"]]
        ttl_geo_scores = [res_val_ttl_geo["macro_f1"], res_val_ttl_geo["pr_auc"], res_val_ttl_geo["recall"]]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - 1.5*width, base_scores, width, label="1. Base (38 feats)", color="#1976D2")
        bars2 = ax.bar(x - 0.5*width, ttl_scores, width, label="2. With TTL (41 feats)", color="#D32F2F")
        bars3 = ax.bar(x + 0.5*width, geo_scores, width, label="3. Base + Geo (42 feats)", color="#388E3C")
        bars4 = ax.bar(x + 1.5*width, ttl_geo_scores, width, label="4. With TTL + Geo (45 feats)", color="#7B1FA2")

        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 6))
        bars1 = ax.bar(x - width/2, base_scores, width, label="Without TTL (38 feats)", color="#1976D2")
        bars2 = ax.bar(x + width/2, ttl_scores, width, label="With TTL (41 feats)", color="#D32F2F")
        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Isolation Forest — Performance Comparison across Feature Sets", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Validation Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    leak_path = fig_dir / "iforest_ttl_leakage_impact.png"
    plt.savefig(leak_path, dpi=300)
    plt.close()
    print(f"  ✓ บันทึกกราฟ Comparison Impact: {leak_path.name}")

# ------------------------------------------------------------------------------
# กราฟที่ 3: Binary Confusion Matrix บน Test Set (Normal vs Attack)
# ------------------------------------------------------------------------------
if "is_attack_test_base" in locals() and "y_test_base" in locals():
    y_true_binary = (y_test_base != "Normal").astype(int)
    y_pred_binary = is_attack_test_base.astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=["Normal", "Attack (Anomaly)"],
        yticklabels=["Normal", "Attack (Anomaly)"],
        cbar=True,
        ax=ax,
    )
    ax.set_title("Isolation Forest (Base) — Binary Confusion Matrix (Normalized)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = fig_dir / "iforest_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  ✓ บันทึกกราฟ Binary Confusion Matrix: {cm_path.name}")

print("✓ จัดการกราฟทั้งหมดของ Isolation Forest เสร็จสิ้น!")

# %%