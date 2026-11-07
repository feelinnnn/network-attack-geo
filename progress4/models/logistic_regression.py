"""Progress 4 — Logistic Regression (Art)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_logreg.csv          (metrics.py เขียนให้เมื่อเรียก evaluate)
        reports/results_logreg_ttl.csv      (รอบมี TTL - Data Leakage)
        reports/results_logreg_geo.csv      (รอบมี Geo - Shortcut Learning)
        reports/figures/logreg_*.png        (กราฟของตัวเอง)
        reports/model_logreg.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว ห้ามเอากลับเข้ามาตอนจูน
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="logreg_ttl"

รัน (จาก root ของ repo):  python progress4/models/logistic_regression.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from metrics import CLASSES, FEAT, SEED, evaluate, load_split

print("✓ โหลด Libraries และตั้งค่าระบบเรียบร้อย")

# ==============================================================================
# PART 1: LOGISTIC REGRESSION (WITHOUT TTL — BASE MODEL)
# ==============================================================================

# %% [2] PART 1: โหลดข้อมูล และเตรียม Preprocessor แบบ Without TTL (38 คอลัมน์)
print("\n" + "=" * 60)
print("PART 1: เริ่มการทำงานรอบ Without TTL (Base Model)")
print("=" * 60)

# ฟังก์ชันสุ่มลดเฉพาะคลาส Normal (เก็บการโจมตีครบทุกแถว 100%)
def undersample_normal(X: pd.DataFrame, y: pd.Series, n_normal: int | None = None, seed: int = SEED):
    """สุ่มลดเฉพาะแถว Normal ให้เหลือ n_normal โดยเก็บแถวโจมตีทั้ง 9 คลาสไว้ครบทั้งหมด"""
    if n_normal is None:
        return X, y
    is_normal = (y == "Normal")
    n_sample = min(n_normal, int(is_normal.sum()))
    X_normal = X[is_normal].sample(n=n_sample, random_state=seed)
    y_normal = y[is_normal].loc[X_normal.index]

    X_attack = X[~is_normal]  # เก็บการโจมตีทั้ง 9 คลาสครบ 100% (49,346 แถว)
    y_attack = y[~is_normal]

    X_resampled = pd.concat([X_normal, X_attack]).sample(frac=1, random_state=seed)
    y_resampled = y.loc[X_resampled.index]
    return X_resampled, y_resampled

# 1. โหลดข้อมูล Base (38 คอลัมน์)
print("กำลังโหลดข้อมูล Train และ Val (Base)...")
X_train_raw, y_train_raw = load_split("train")
X_val_base, y_val_base = load_split("val")

# ------------------------------------------------------------------------------
# กำหนดค่า n_normal เพื่อทดลองการปรับสัดส่วนข้อมูล (Undersampling)
# - None    : ใช้ Normal เต็มจำนวน 1,392,219 แถว (ค่ามาตรฐานเดิม)
# - 50000   : อัตราส่วน ~1:1 กับยอดโจมตี (เน้นจำแนกการโจมตีเร็วมาก)
# - 100000  : อัตราส่วน ~2:1 (จุดสมดุลความเร็วกับความแม่นยำ)
# - 200000  : อัตราส่วน ~4:1
# ------------------------------------------------------------------------------
N_NORMAL = 800000

X_train_base, y_train_base = undersample_normal(X_train_raw, y_train_raw, n_normal=N_NORMAL)

print(f"  - ขนาด X_train_base: {X_train_base.shape} (Normal: {(y_train_base == 'Normal').sum():,} แถว, Attack: {(y_train_base != 'Normal').sum():,} แถว)")
print(f"  - ขนาด X_val_base  : {X_val_base.shape} (คงสัดส่วนจริงในธรรมชาติ)")

# 2. แยกประเภทคอลัมน์ และคัดเลือกฟีเจอร์เบ้ขวา (Skewed Flow Features) ตาม EDA ของ Art (ข้อ 5)
cat_cols_base = [c for c in FEAT["categorical"] if c in X_train_base.columns]
num_cols_base = [c for c in X_train_base.columns if c not in cat_cols_base]

# รายชื่อฟีเจอร์โฟลว์เชิงปริมาณที่มีความเบ้ขวาสูงมาก (Heavy-Tailed Skewed Features)
SKEWED_FLOW_COLS = [
    "dur", "sbytes", "dbytes", "sload", "dload", "spkts", "dpkts",
    "sloss", "dloss", "sjit", "djit", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat", "res_bdy_len", "smeansz", "dmeansz",
]

skewed_cols_base = [c for c in SKEWED_FLOW_COLS if c in num_cols_base]
other_num_cols_base = [c for c in num_cols_base if c not in skewed_cols_base]

print(f"  - Categorical ({len(cat_cols_base)} คอลัมน์): {cat_cols_base}")
print(f"  - Skewed Flow ({len(skewed_cols_base)} คอลัมน์ ทำ log1p + StandardScaler): {skewed_cols_base[:4]}...")
print(f"  - Other Numeric ({len(other_num_cols_base)} คอลัมน์ ทำ StandardScaler ทั่วไป): {other_num_cols_base}")

# 3. สร้าง Preprocessor และ Pipeline (Log Transformation np.log1p + Balanced Class Weight)
log_scaler = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])

prep_base = ColumnTransformer(
    transformers=[
        ("log_num", log_scaler, skewed_cols_base),
        ("std_num", StandardScaler(), other_num_cols_base),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_base),
    ]
)

params_base = {
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",  # ปรับคืนเป็น "balanced" เพื่อความสมดุลต่อ Minority Attacks
    "solver": "lbfgs",
    "ttl": False,
    "log1p": True,
    "n_normal": N_NORMAL if N_NORMAL is not None else "full",
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
print(f"✓ ประกอบ Pipeline Base Model เรียบร้อย (n_normal = {params_base['n_normal']})")

# %% [3] PART 1: ฝึกสอนและประเมินผล Without TTL (Base Model)
print(f"กำลังฝึกสอนโมเดล Logistic Regression (Without TTL, n_normal={params_base['n_normal']})...")
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

# %% [3B] (เสริม) รันเปรียบเทียบค่า n_normal หลายๆ ค่าอัตโนมัติ (Tuning n_normal Grid)
# สามารถคลิก Run Cell นี้ เพื่อให้โมเดลทดลอง n_normal = 50,000, 100,000, และ 200,000
# แล้วนำผลลัพธ์มาเทียบในตารางเดียวให้เห็นความต่างชัดเจน
grid_results = []
n_normal_candidates = [50000, 100000, 200000, 400000, 800000, None]

print("กำลังรันการทดลองเปรียบเทียบค่า n_normal ต่างๆ บน Validation Set...")
for n_cand in n_normal_candidates:
    X_tr, y_tr = undersample_normal(X_train_raw, y_train_raw, n_normal=n_cand)
    p_cand = {**params_base, "n_normal": n_cand if n_cand is not None else "full"}
    
    m_cand = Pipeline([
        ("prep", prep_base),
        ("clf", LogisticRegression(
            C=p_cand["C"],
            max_iter=p_cand["max_iter"],
            class_weight=p_cand["class_weight"],
            solver=p_cand["solver"],
            random_state=SEED,
        )),
    ])
    m_cand.fit(X_tr, y_tr)
    
    y_p_val = m_cand.predict(X_val_base)
    y_s_val = m_cand.predict_proba(X_val_base)
    classes_cand = m_cand.classes_ if hasattr(m_cand, "classes_") else m_cand.named_steps["clf"].classes_
    
    res = evaluate(
        y_true=y_val_base,
        y_pred=y_p_val,
        y_score=y_s_val,
        score_classes=classes_cand,
        model="logreg",
        split="val",
        params=p_cand,
    )
    grid_results.append({
        "n_normal": f"{n_cand:,}" if n_cand is not None else "Full (1.44M)",
        "Train Rows": f"{len(X_tr):,}",
        "Val Macro F1": round(res["macro_f1"], 4),
        "Val PR-AUC": round(res["pr_auc"], 4),
        "Val Recall": round(res["recall"], 4),
    })

print("\n" + "=" * 68)
print("ตารางเปรียบเทียบผลกระทบของค่า n_normal บน Validation Set:")
print("=" * 68)
print(pd.DataFrame(grid_results).to_string(index=False))
print("=" * 68)

# ==============================================================================
# PART 2: LOGISTIC REGRESSION (WITH TTL — LEAKAGE TEST)
# ==============================================================================

# %% [4] PART 2: โหลดข้อมูล และเตรียม Preprocessor แบบ With TTL (41 คอลัมน์)
print("\n" + "=" * 60)
print("PART 2: เริ่มการทำงานรอบ With TTL (Data Leakage Test)")
print("=" * 60)

# 1. โหลดข้อมูลพร้อมฟีเจอร์ TTL (extra=FEAT["check"]) รวม 41 คอลัมน์
print("กำลังโหลดข้อมูล Train และ Val (With TTL)...")
X_train_ttl_raw, y_train_ttl_raw = load_split("train", extra=FEAT["check"])
X_val_ttl, y_val_ttl = load_split("val", extra=FEAT["check"])

# กำหนดค่า n_normal สำหรับโหมด With TTL (สามารถกำหนดให้ตรงกับ N_NORMAL เพื่อความยุติธรรมในการเปรียบเทียบ)
N_NORMAL_TTL = 800000

X_train_ttl, y_train_ttl = undersample_normal(X_train_ttl_raw, y_train_ttl_raw, n_normal=N_NORMAL_TTL)

print(f"  - ขนาด X_train_ttl: {X_train_ttl.shape} (Normal: {(y_train_ttl == 'Normal').sum():,} แถว, Attack: {(y_train_ttl != 'Normal').sum():,} แถว)")
print(f"  - ขนาด X_val_ttl  : {X_val_ttl.shape}")
print(f"  - ฟีเจอร์ TTL ที่เพิ่ม: {FEAT['check']}")

# 2. แยกประเภทคอลัมน์ และจัดกลุ่มฟีเจอร์เบ้ขวา (รวมถึงฟีเจอร์ TTL ที่เพิ่มเข้ามา)
cat_cols_ttl = [c for c in FEAT["categorical"] if c in X_train_ttl.columns]
num_cols_ttl = [c for c in X_train_ttl.columns if c not in cat_cols_ttl]

skewed_cols_ttl = [c for c in SKEWED_FLOW_COLS if c in num_cols_ttl]
other_num_cols_ttl = [c for c in num_cols_ttl if c not in skewed_cols_ttl]

# 3. สร้าง Preprocessor และ Pipeline ด้วย Log Transformation สำหรับ With TTL
prep_ttl = ColumnTransformer(
    transformers=[
        ("log_num", log_scaler, skewed_cols_ttl),
        ("std_num", StandardScaler(), other_num_cols_ttl),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_ttl),
    ]
)

params_ttl = {
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",  # ปรับคืนเป็น "balanced" เพื่อความสมดุลต่อ Minority Attacks
    "solver": "lbfgs",
    "ttl": True,
    "log1p": True,
    "n_normal": N_NORMAL_TTL if N_NORMAL_TTL is not None else "full",
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
print(f"✓ ประกอบ Pipeline With TTL เรียบร้อย (n_normal = {params_ttl['n_normal']})")

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

print("✓ บันทึกผล TTL ลง reports/results_logreg_ttl.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_ttl['macro_f1']:.4f} | PR-AUC: {res_val_ttl['pr_auc']:.4f} | Recall: {res_val_ttl['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_ttl['macro_f1']:.4f} | PR-AUC: {res_test_ttl['pr_auc']:.4f} | Recall: {res_test_ttl['recall']:.4f}")

# %% [5B] (เสริม) รันเปรียบเทียบค่า n_normal ทุกค่าสำหรับโหมด With TTL (Tuning With TTL Grid)
# วนลูปทดลอง n_normal = 50,000, 100,000, 200,000, 400,000, 800,000, และ None (Full) บนชุด With TTL
grid_results_ttl = []
n_normal_candidates_ttl = [50000, 100000, 200000, 400000, 800000, None]

print("กำลังรันการทดลองเปรียบเทียบค่า n_normal ต่างๆ บน Validation Set (With TTL)...")
for n_cand in n_normal_candidates_ttl:
    X_tr_ttl, y_tr_ttl = undersample_normal(X_train_ttl_raw, y_train_ttl_raw, n_normal=n_cand)
    p_cand_ttl = {**params_ttl, "n_normal": n_cand if n_cand is not None else "full"}
    
    m_cand_ttl = Pipeline([
        ("prep", prep_ttl),
        ("clf", LogisticRegression(
            C=p_cand_ttl["C"],
            max_iter=p_cand_ttl["max_iter"],
            class_weight=p_cand_ttl["class_weight"],
            solver=p_cand_ttl["solver"],
            random_state=SEED,
        )),
    ])
    m_cand_ttl.fit(X_tr_ttl, y_tr_ttl)
    
    y_p_val_ttl = m_cand_ttl.predict(X_val_ttl)
    y_s_val_ttl = m_cand_ttl.predict_proba(X_val_ttl)
    classes_cand_ttl = m_cand_ttl.classes_ if hasattr(m_cand_ttl, "classes_") else m_cand_ttl.named_steps["clf"].classes_
    
    res = evaluate(
        y_true=y_val_ttl,
        y_pred=y_p_val_ttl,
        y_score=y_s_val_ttl,
        score_classes=classes_cand_ttl,
        model="logreg_ttl",
        split="val",
        params=p_cand_ttl,
    )
    grid_results_ttl.append({
        "n_normal": f"{n_cand:,}" if n_cand is not None else "Full (1.44M)",
        "Train Rows": f"{len(X_tr_ttl):,}",
        "Val Macro F1": round(res["macro_f1"], 4),
        "Val PR-AUC": round(res["pr_auc"], 4),
        "Val Recall": round(res["recall"], 4),
    })

print("\n" + "=" * 68)
print("ตารางเปรียบเทียบผลกระทบของค่า n_normal บน Validation Set (With TTL):")
print("=" * 68)
print(pd.DataFrame(grid_results_ttl).to_string(index=False))
print("=" * 68)

# ==============================================================================
# PART 3: LOGISTIC REGRESSION (WITH GEO — SHORTCUT LEARNING TEST)
# ==============================================================================

# %% [6] PART 3: โหลดข้อมูล และเตรียม Preprocessor แบบ Base + Geo (42 คอลัมน์)
print("\n" + "=" * 60)
print("PART 3: เริ่มการทำงานรอบ With Geo (Geolocation Shortcut Test)")
print("=" * 60)

# 1. โหลดข้อมูลพร้อมฟีเจอร์ Geo (extra=FEAT["geo"]) รวม 42 คอลัมน์ (38 Base + 4 Geo)
print("กำลังโหลดข้อมูล Train และ Val (With Geo)...")
X_train_geo_raw, y_train_geo_raw = load_split("train", extra=FEAT["geo"])
X_val_geo, y_val_geo = load_split("val", extra=FEAT["geo"])

# กำหนดค่า n_normal สำหรับโหมด With Geo
N_NORMAL_GEO = 800000

X_train_geo, y_train_geo = undersample_normal(X_train_geo_raw, y_train_geo_raw, n_normal=N_NORMAL_GEO)

print(f"  - ขนาด X_train_geo: {X_train_geo.shape} (Normal: {(y_train_geo == 'Normal').sum():,} แถว, Attack: {(y_train_geo != 'Normal').sum():,} แถว)")
print(f"  - ขนาด X_val_geo  : {X_val_geo.shape}")
print(f"  - ฟีเจอร์ Geo ที่เพิ่ม: {FEAT['geo']}")

# 2. แยกประเภทคอลัมน์ และแปลงคอลัมน์ Categorical เป็น string ป้องกันปัญหา mixed types
cat_cols_geo = [c for c in FEAT["categorical"] if c in X_train_geo.columns]
for c in cat_cols_geo:
    X_train_geo[c] = X_train_geo[c].astype(str)
    X_val_geo[c] = X_val_geo[c].astype(str)

num_cols_geo = [c for c in X_train_geo.columns if c not in cat_cols_geo]

skewed_cols_geo = [c for c in SKEWED_FLOW_COLS if c in num_cols_geo]
other_num_cols_geo = [c for c in num_cols_geo if c not in skewed_cols_geo]

print(f"  - Categorical ({len(cat_cols_geo)} คอลัมน์ รวมถึง Geo): {cat_cols_geo}")
print(f"  - Skewed Flow ({len(skewed_cols_geo)} คอลัมน์ ทำ log1p + StandardScaler): {skewed_cols_geo[:4]}...")
print(f"  - Other Numeric ({len(other_num_cols_geo)} คอลัมน์ ทำ StandardScaler): {other_num_cols_geo}")

# 3. สร้าง Preprocessor และ Pipeline สำหรับ With Geo
log_scaler_geo = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])

prep_geo = ColumnTransformer(
    transformers=[
        ("log_num", log_scaler_geo, skewed_cols_geo),
        ("std_num", StandardScaler(), other_num_cols_geo),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_geo),
    ]
)

params_geo = {
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "geo": True,
    "ttl": False,
    "log1p": True,
    "n_normal": N_NORMAL_GEO if N_NORMAL_GEO is not None else "full",
}

model_geo = Pipeline([
    ("prep", prep_geo),
    ("clf", LogisticRegression(
        C=params_geo["C"],
        max_iter=params_geo["max_iter"],
        class_weight=params_geo["class_weight"],
        solver=params_geo["solver"],
        random_state=SEED,
    )),
])
print(f"✓ ประกอบ Pipeline With Geo เรียบร้อย (n_normal = {params_geo['n_normal']})")

# %% [7] PART 3: ฝึกสอนและประเมินผล With Geo (Shortcut Learning Test)
print("กำลังฝึกสอนโมเดล Logistic Regression (With Geo)...")
model_geo.fit(X_train_geo, y_train_geo)
print("✓ เทรน With Geo เสร็จเรียบร้อย!")

# ทำนายและประเมินผลบน Validation Set
y_pred_val_geo = model_geo.predict(X_val_geo)
y_score_val_geo = model_geo.predict_proba(X_val_geo)
classes_geo = model_geo.classes_ if hasattr(model_geo, "classes_") else model_geo.named_steps["clf"].classes_

res_val_geo = evaluate(
    y_true=y_val_geo,
    y_pred=y_pred_val_geo,
    y_score=y_score_val_geo,
    score_classes=classes_geo,
    model="logreg_geo",
    split="val",
    params=params_geo,
)

# ทำนายและประเมินผลบน Test Set
X_test_geo, y_test_geo = load_split("test", extra=FEAT["geo"])
for c in cat_cols_geo:
    X_test_geo[c] = X_test_geo[c].astype(str)

y_pred_test_geo = model_geo.predict(X_test_geo)
y_score_test_geo = model_geo.predict_proba(X_test_geo)

res_test_geo = evaluate(
    y_true=y_test_geo,
    y_pred=y_pred_test_geo,
    y_score=y_score_test_geo,
    score_classes=classes_geo,
    model="logreg_geo",
    split="test",
    params=params_geo,
)

print("✓ บันทึกผล Geo ลง reports/results_logreg_geo.csv เรียบร้อยแล้ว")
print(f"  [Val ] Macro F1: {res_val_geo['macro_f1']:.4f} | PR-AUC: {res_val_geo['pr_auc']:.4f} | Recall: {res_val_geo['recall']:.4f}")
print(f"  [Test] Macro F1: {res_test_geo['macro_f1']:.4f} | PR-AUC: {res_test_geo['pr_auc']:.4f} | Recall: {res_test_geo['recall']:.4f}")

# ==============================================================================
# PART 4: COMPARISON (เปรียบเทียบผลลัพธ์ 3 มิติ: Base vs TTL vs Geo)
# ==============================================================================

# %% [8] PART 4: สรุปตารางเปรียบเทียบผลลัพธ์ทุกรูปแบบ (Base vs TTL vs Geo)
# นำผลลัพธ์จากทั้งสองรอบ grid (Base vs TTL) มาประกบข้างกันถ้ามี
if "grid_results" in locals() and "grid_results_ttl" in locals() and len(grid_results) > 0 and len(grid_results_ttl) > 0:
    df_base = pd.DataFrame(grid_results)
    df_ttl = pd.DataFrame(grid_results_ttl)
    
    comp_df = pd.DataFrame({
        "n_normal": df_base["n_normal"],
        "Base F1": df_base["Val Macro F1"],
        "TTL F1": df_ttl["Val Macro F1"],
        "Diff F1": (df_ttl["Val Macro F1"] - df_base["Val Macro F1"]).round(4),
        "Base PR-AUC": df_base["Val PR-AUC"],
        "TTL PR-AUC": df_ttl["Val PR-AUC"],
        "Diff PR-AUC": (df_ttl["Val PR-AUC"] - df_base["Val PR-AUC"]).round(4),
    })
    print("\n" + "=" * 80)
    print("ตารางเปรียบเทียบ Without TTL vs With TTL ในทุกระดับ n_normal (Data Leakage Impact):")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)

# แสดงตารางสรุป 3 โมเดลหลัก
summary_data = []

if "res_val_base" in locals() and "res_test_base" in locals():
    summary_data.append({
        "Model": "1. Base (Without TTL/Geo)",
        "Features": "38 (Traffic Flow Only)",
        "Val Macro F1": round(res_val_base["macro_f1"], 4),
        "Val PR-AUC": round(res_val_base["pr_auc"], 4),
        "Val Recall": round(res_val_base["recall"], 4),
        "Test Macro F1": round(res_test_base["macro_f1"], 4),
        "Test PR-AUC": round(res_test_base["pr_auc"], 4),
    })

if "res_val_ttl" in locals() and "res_test_ttl" in locals():
    summary_data.append({
        "Model": "2. With TTL (Data Leakage)",
        "Features": "41 (Base + 3 TTL)",
        "Val Macro F1": round(res_val_ttl["macro_f1"], 4),
        "Val PR-AUC": round(res_val_ttl["pr_auc"], 4),
        "Val Recall": round(res_val_ttl["recall"], 4),
        "Test Macro F1": round(res_test_ttl["macro_f1"], 4),
        "Test PR-AUC": round(res_test_ttl["pr_auc"], 4),
    })

if "res_val_geo" in locals() and "res_test_geo" in locals():
    summary_data.append({
        "Model": "3. With Geo (Shortcut Learning)",
        "Features": "42 (Base + 4 Geo)",
        "Val Macro F1": round(res_val_geo["macro_f1"], 4),
        "Val PR-AUC": round(res_val_geo["pr_auc"], 4),
        "Val Recall": round(res_val_geo["recall"], 4),
        "Test Macro F1": round(res_test_geo["macro_f1"], 4),
        "Test PR-AUC": round(res_test_geo["pr_auc"], 4),
    })

df_summary = pd.DataFrame(summary_data)

print("\n" + "=" * 85)
print("ตารางเปรียบเทียบผลลัพธ์ Logistic Regression (Base vs TTL vs Geo):")
print("=" * 85)
print(df_summary.to_string(index=False))
print("=" * 85)

# ==============================================================================
# PART 5: VISUALIZATIONS (สร้างรูปภาพสำหรับรายงาน Progress 4)
# ==============================================================================

# %% [9] PART 5: สร้างกราฟผลลัพธ์ลง reports/figures/
print("\n" + "=" * 60)
print("PART 5: กำลังสร้างกราฟสรุปผลลัพธ์ลง reports/figures/...")
print("=" * 60)

fig_dir = repo_root / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# กราฟที่ 1: Confusion Matrix บน Test Set (Normalized per True Class)
# ------------------------------------------------------------------------------
if "y_test_base" in locals() and "y_pred_test_base" in locals():
    cm = confusion_matrix(y_test_base, y_pred_test_base, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        ax=ax,
    )
    ax.set_title("Logistic Regression (Without TTL) — Test Set Confusion Matrix (Normalized)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = fig_dir / "logreg_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  ✓ บันทึกกราฟ Confusion Matrix: {cm_path.name}")

# ------------------------------------------------------------------------------
# กราฟที่ 2: เปรียบเทียบผลกระทบของ TTL Leakage และ Geo Shortcut
# ------------------------------------------------------------------------------
if "res_val_base" in locals() and "res_val_ttl" in locals():
    metrics_names = ["Macro F1", "PR-AUC", "Recall"]
    base_scores = [res_val_base["macro_f1"], res_val_base["pr_auc"], res_val_base["recall"]]
    ttl_scores = [res_val_ttl["macro_f1"], res_val_ttl["pr_auc"], res_val_ttl["recall"]]

    x = np.arange(len(metrics_names))
    
    if "res_val_geo" in locals():
        width = 0.25
        geo_scores = [res_val_geo["macro_f1"], res_val_geo["pr_auc"], res_val_geo["recall"]]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, base_scores, width, label="1. Base (38 feats)", color="#1976D2")
        bars2 = ax.bar(x, ttl_scores, width, label="2. With TTL (41 feats - Leakage)", color="#D32F2F")
        bars3 = ax.bar(x + width, geo_scores, width, label="3. With Geo (42 feats - Shortcut)", color="#388E3C")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 6))
        bars1 = ax.bar(x - width/2, base_scores, width, label="Without TTL (38 feats)", color="#1976D2")
        bars2 = ax.bar(x + width/2, ttl_scores, width, label="With TTL (41 feats - Leakage)", color="#D32F2F")
        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Performance Comparison: Base vs TTL Leakage vs Geo Shortcut", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Validation Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    leak_path = fig_dir / "logreg_ttl_leakage_impact.png"
    plt.savefig(leak_path, dpi=300)
    plt.close()
    print(f"  ✓ บันทึกกราฟ Comparison Impact: {leak_path.name}")

# ------------------------------------------------------------------------------
# กราฟที่ 3: Top Feature Coefficients (สัมประสิทธิ์ฟีเจอร์สำคัญ)
# ------------------------------------------------------------------------------
target_model = model_geo if "model_geo" in locals() and hasattr(model_geo.named_steps["clf"], "coef_") else (model_base if "model_base" in locals() and hasattr(model_base.named_steps["clf"], "coef_") else None)
if target_model is not None:
    try:
        clf_step = target_model.named_steps["clf"]
        prep_step = target_model.named_steps["prep"]
        feature_names = prep_step.get_feature_names_out()
        
        # คำนวณความสำคัญรวม (Mean Absolute Coefficient ข้ามทุกคลาส)
        mean_abs_coef = np.mean(np.abs(clf_step.coef_), axis=0)
        coef_series = pd.Series(mean_abs_coef, index=feature_names).sort_values(ascending=False).head(15)

        # ทำความสะอาดชื่อฟีเจอร์ให้อ่านง่าย
        clean_names = [f.replace("log_num__", "log1p_").replace("std_num__", "").replace("cat__", "") for f in coef_series.index]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=coef_series.values, y=clean_names, palette="viridis", ax=ax)
        model_title = "Base + Geo" if target_model == model_geo else "Base Model"
        ax.set_title(f"Top 15 Feature Importances ({model_title}) — Mean |Coefficient|", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Absolute Weight (|β|)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        feat_path = fig_dir / "logreg_feature_importance.png"
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"  ✓ บันทึกกราฟ Feature Importance: {feat_path.name}")
    except Exception as e:
        print(f"  ! ไม่สามารถสร้างกราฟ Feature Importance: {e}")

print("✓ จัดการกราฟทั้งหมดเสร็จสิ้น!")

# %%
