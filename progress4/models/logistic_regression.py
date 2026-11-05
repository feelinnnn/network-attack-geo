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
# %% [1] Imports & Setup Path
import sys
from pathlib import Path
import pandas as pd

# ค้นหา Root ของ Repo (โฟลเดอร์ที่มี configs/) แบบอัตโนมัติ
current_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
repo_root = next((p for p in [current_dir, *current_dir.parents] if (p / "configs").exists()), Path.cwd())

# เพิ่มโฟลเดอร์ repo root และ progress4 เข้า sys.path
for p in [repo_root, repo_root / "progress4"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metrics import FEAT, SEED, evaluate, load_split

print("✓ โหลด Libraries และตั้งค่าระบบเรียบร้อย")

# ==============================================================================
# PART 1: LOGISTIC REGRESSION (WITHOUT TTL — BASE MODEL)
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

params_base = {
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "ttl": False,
}

model_base = Pipeline([
    ("prep", prep_base),
    ("clf", LogisticRegression(
        C=params_base["C"],
        max_iter=params_base["max_iter"],
        class_weight=params_base["class_weight"],
        solver=params_base["solver"],
        random_state=SEED,
    )),
])
print("✓ ประกอบ Pipeline Base Model เรียบร้อย")

# %% [3] PART 1: ฝึกสอนและประเมินผล Without TTL (Base Model)
print("กำลังฝึกสอนโมเดล Logistic Regression (Without TTL)...")
model_base.fit(X_train_base, y_train_base)
print("✓ เทรน Base Model เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_base = model_base.predict(X_val_base)
y_score_val_base = model_base.predict_proba(X_val_base)
classes_base = model_base.classes_ if hasattr(model_base, "classes_") else model_base.named_steps["clf"].classes_

res_val_base = evaluate(
    y_true=y_val_base,
    y_pred=y_pred_val_base,
    y_score=y_score_val_base,
    score_classes=classes_base,
    model="logreg",
    split="val",
    params=params_base,
)

# ทำนายและประเมินผลบน Test Set
X_test_base, y_test_base = load_split("test")
y_pred_test_base = model_base.predict(X_test_base)
y_score_test_base = model_base.predict_proba(X_test_base)

res_test_base = evaluate(
    y_true=y_test_base,
    y_pred=y_pred_test_base,
    y_score=y_score_test_base,
    score_classes=classes_base,
    model="logreg",
    split="test",
    params=params_base,
)

print(f"✓ บันทึกผล Base ลง reports/results_logreg.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_base['macro_f1']:.4f} | PR-AUC: {res_val_base['pr_auc']:.4f} | Recall: {res_val_base['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_base['macro_f1']:.4f} | PR-AUC: {res_test_base['pr_auc']:.4f} | Recall: {res_test_base['recall']:.4f}")

# ==============================================================================
# PART 2: LOGISTIC REGRESSION (WITH TTL — LEAKAGE TEST)
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
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "ttl": True,
}

model_ttl = Pipeline([
    ("prep", prep_ttl),
    ("clf", LogisticRegression(
        C=params_ttl["C"],
        max_iter=params_ttl["max_iter"],
        class_weight=params_ttl["class_weight"],
        solver=params_ttl["solver"],
        random_state=SEED,
    )),
])
print("✓ ประกอบ Pipeline With TTL เรียบร้อย")

# %% [5] PART 2: ฝึกสอนและประเมินผล With TTL (Leakage Test)
print("กำลังฝึกสอนโมเดล Logistic Regression (With TTL)...")
model_ttl.fit(X_train_ttl, y_train_ttl)
print("✓ เทรน With TTL เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_ttl = model_ttl.predict(X_val_ttl)
y_score_val_ttl = model_ttl.predict_proba(X_val_ttl)
classes_ttl = model_ttl.classes_ if hasattr(model_ttl, "classes_") else model_ttl.named_steps["clf"].classes_

res_val_ttl = evaluate(
    y_true=y_val_ttl,
    y_pred=y_pred_val_ttl,
    y_score=y_score_val_ttl,
    score_classes=classes_ttl,
    model="logreg_ttl",
    split="val",
    params=params_ttl,
)

# ทำนายและประเมินผลบน Test Set
X_test_ttl, y_test_ttl = load_split("test", extra=FEAT["check"])
y_pred_test_ttl = model_ttl.predict(X_test_ttl)
y_score_test_ttl = model_ttl.predict_proba(X_test_ttl)

res_test_ttl = evaluate(
    y_true=y_test_ttl,
    y_pred=y_pred_test_ttl,
    y_score=y_score_test_ttl,
    score_classes=classes_ttl,
    model="logreg_ttl",
    split="test",
    params=params_ttl,
)

print(f"✓ บันทึกผล TTL ลง reports/results_logreg_ttl.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_ttl['macro_f1']:.4f} | PR-AUC: {res_val_ttl['pr_auc']:.4f} | Recall: {res_val_ttl['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_ttl['macro_f1']:.4f} | PR-AUC: {res_test_ttl['pr_auc']:.4f} | Recall: {res_test_ttl['recall']:.4f}")

# ==============================================================================
# PART 3: COMPARISON (เปรียบเทียบผลลัพธ์)
# ==============================================================================

# %% [6] PART 3: สรุปตารางเปรียบเทียบผลลัพธ์ (Without TTL vs With TTL)
# ดึงค่าล่าสุดมาแสดงเปรียบเทียบกัน
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
print("ตารางเปรียบเทียบผลลัพธ์ Logistic Regression (Leakage Effect):")
print("=" * 70)
print(df_summary.to_string(index=False))
print("=" * 70)

# %%
