"""Progress 4 - การทดลอง Random Forest ของ Pimmy

วัตถุประสงค์ของไฟล์นี้
----------------------
การทดลองรอบแรกจะสร้างโมเดล ``RF-base`` โดยใช้เฉพาะ feature ที่อธิบาย
พฤติกรรมของ traffic จาก ``features.yaml`` ในส่วน ``base``

รอบนี้ตั้งใจ *ไม่ใช้* feature ต่อไปนี้
    - IP และ port ซึ่งเป็นตัวระบุของเครื่อง (โมเดลอาจจำค่าแทนพฤติกรรม)
    - Geo เช่น country และ ASN (เสี่ยงเกิด geo shortcut)
    - TTL ได้แก่ sttl, dttl และ ct_state_ttl (เสี่ยงเกิด lab leakage)

ขั้นตอนการทดลอง
----------------
1. อ่าน train/val/test ที่ถูกแบ่งตามเวลาไว้แล้วจาก ``split.py``
2. แปลงข้อมูลประเภทข้อความด้วย One-Hot Encoding
3. ทดลองค่า hyperparameter หลายชุด โดยฝึกจาก train และวัดจาก val
4. เลือกชุดที่มี Macro F1 บน val สูงที่สุด
5. ฝึกโมเดลสุดท้ายใหม่ด้วย train + val
6. วัด test เพียงครั้งเดียวด้วยค่า hyperparameter ที่เลือกแล้ว

สิ่งที่ไฟล์นี้สร้าง
------------------
    - reports/results_rf.csv
        ผลของทุก configuration ที่ลองบน val และผลสุดท้ายบน test
    - reports/rf_base_class_recall.csv
        Recall แยกตาม class ของผล test
    - reports/rf_base_confusion_matrix.csv
        Confusion matrix ของผล test
    - reports/figures/rf_*.png
        กราฟ tuning, metric comparison, leakage gap, recall และ confusion matrix
    - reports/model_rf.md
        สรุปขั้นตอนและผลการทดลองสำหรับนำไปอธิบายงาน

หมายเหตุ
--------
ไฟล์ ``metrics.py`` และ ``split.py`` เป็นไฟล์กลางของทีมและห้ามแก้
ดังนั้นการเตรียมข้อมูลและการเลือกโมเดลทั้งหมดจึงทำในไฟล์นี้เท่านั้น

โหมดการรันจาก root ของ repository:
    /opt/anaconda3/bin/python progress4/models/random_forest.py --experiment base
    /opt/anaconda3/bin/python progress4/models/random_forest.py --experiment ablation
    /opt/anaconda3/bin/python progress4/models/random_forest.py --experiment figures

``base`` ใช้สำหรับสร้าง baseline ส่วน ``ablation`` จะรันการเปรียบเทียบ
RF-base + TTL และ RF-base + Geo โดยใช้ hyperparameter เดียวกับ baseline
ที่เลือกไว้แล้ว

``figures`` อ่านผลจาก CSV ที่มีอยู่แล้วเพื่อสร้างกราฟ โดยไม่ train โมเดลและ
ไม่เรียก evaluate จึงไม่เพิ่มผล test ซ้ำ
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ไฟล์ metrics.py อยู่ใน progress4/ แต่ไฟล์นี้อยู่ใน progress4/models/
# จึงเพิ่ม progress4 เข้า sys.path เพื่อให้ import metrics ได้เมื่อรันไฟล์โดยตรง
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import CLASSES, FEAT, SEED, evaluate, load_split


# ------------------------------------------------------------
# การตั้งค่าการทดลอง
# ------------------------------------------------------------

# ============================================================
# รอบที่ 1: RF-base
#
# ส่วนนี้เป็นโค้ดของการทดลองครั้งแรก เก็บไว้เพื่อย้อนรันและตรวจสอบผลเดิมได้
# ห้ามลบเมื่อเริ่มการทดลองรอบใหม่
# ============================================================

# ใช้ค่าหลายชุดเพื่อหาค่าที่เหมาะสมบน validation set
# ทุกชุดใช้ class_weight เพื่อไม่ให้ Normal ซึ่งมีจำนวนมากครอบงำโมเดล
#
# ข้อมูลมี train มากกว่า 1.4 ล้านแถว จึงใช้ต้นไม้จำนวน 30 ต้นในช่วงค้นหา
# เพื่อให้เปรียบเทียบ configuration ได้ครบในเวลาที่เหมาะสม แล้วค่อยเพิ่ม
# จำนวนต้นไม้ในโมเดลสุดท้ายด้านล่าง
SEARCH_PARAMS = [
    {
        "n_estimators": 30,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 30,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 30,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 30,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "class_weight": "balanced_subsample",
    },
]

# ใช้ค่าเดียวกับ configuration ที่ถูกเลือกบน validation ในรอบ final
# เพื่อให้ตรงตามกติกา "จูนบน val แล้ว train ใหม่ด้วยค่าเดิม"
FINAL_N_ESTIMATORS = 30

# Hyperparameter ที่เลือกได้จาก RF-base รอบแรก
# ใช้ค่าชุดเดียวกันกับ TTL และ Geo เพื่อให้การเปรียบเทียบเป็น ablation study
# ที่วัดผลจากการเพิ่ม feature เป็นหลัก ไม่ใช่เปรียบเทียบจากการจูนคนละแบบ
ABLATION_PARAMS = {
    "n_estimators": FINAL_N_ESTIMATORS,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """สร้าง OneHotEncoder ที่รองรับ scikit-learn หลายเวอร์ชัน"""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # scikit-learn รุ่นเก่าใช้ชื่อ argument ว่า sparse
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_model(feature_columns: list[str], params: dict) -> Pipeline:
    """สร้าง preprocessing และ Random Forest เป็น Pipeline เดียวกัน

    Pipeline ทำให้ preprocessing ถูก fit จากข้อมูลที่ส่งเข้า fit เท่านั้น
    จึงไม่เกิดการนำข้อมูล val/test ไปคำนวณ encoder หรือค่าเติม missing ล่วงหน้า
    """

    # เลือกเฉพาะ categorical columns ที่อยู่ใน base features จริง ๆ
    categorical = [
        column for column in FEAT["categorical"] if column in feature_columns
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    # ตัวเลข: เติม missing ด้วย median ที่คำนวณจาก train
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # ข้อความ: เติม missing ด้วยค่าที่พบบ่อย แล้วแปลงเป็น 0/1 columns
    # handle_unknown ช่วยให้ category ที่พบครั้งแรกใน val/test ไม่ทำให้รันพัง
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    # ใช้ seed เดียวกับ split และโมเดลอื่น ๆ ตามข้อตกลงของทีม
    classifier = RandomForestClassifier(
        **params,
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def load_splits(
    extra: list[str] | None = None,
    train_sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """โหลด train/val/test และเลือก feature เพิ่มสำหรับ ablation study

    เมื่อ ``extra`` เป็น None จะได้ RF-base ตาม features.yaml ส่วน base
    เมื่อส่ง FEAT["check"] จะได้ base + TTL และเมื่อส่ง FEAT["geo"]
    จะได้ base + Geo
    """

    # load_split ใช้ FEAT["base"] จาก features.yaml เป็นชุด feature กลาง
    # และจะต่อคอลัมน์ใน extra ให้โดยไม่แก้ metrics.py
    x_train, y_train = load_split("train", extra=extra)
    x_val, y_val = load_split("val", extra=extra)
    x_test, y_test = load_split("test", extra=extra)

    # สำคัญ: สุ่มเฉพาะ train เท่านั้น
    # validation และ test ต้องใช้ข้อมูลเต็มตาม time split เดิม เพื่อให้คะแนน
    # เปรียบเทียบกับการทดลองรอบก่อนหน้าได้อย่างยุติธรรม
    if train_sample_size is not None:
        x_train, y_train = stratified_sample_train(
            x_train,
            y_train,
            max_rows=train_sample_size,
        )
    return x_train, y_train, x_val, y_val, x_test, y_test


def stratified_sample_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_rows: int,
    min_per_class: int = 500,
) -> tuple[pd.DataFrame, pd.Series]:
    """สุ่ม train แบบ stratified พร้อมกันพื้นที่ให้คลาสเล็ก

    การสุ่มแบบ random ธรรมดาจะอิงสัดส่วนเดิมของข้อมูล ซึ่งทำให้คลาสอย่าง
    Worms ที่มีเพียง 88 แถวใน train อาจเหลือเพียงไม่กี่แถวหรือไม่ถูกเลือกเลย
    ส่งผลให้โมเดลเรียนรู้คลาสนั้นไม่ได้และ Macro F1 ลดลง

    วิธีนี้จึงทำตามกติกา:
        1. เก็บอย่างน้อย min_per_class แถวต่อคลาสเท่าที่มีอยู่จริง
        2. แจกจำนวนแถวที่เหลือตามสัดส่วนของข้อมูลที่ยังไม่ได้เลือก
        3. สุ่มด้วย SEED เดิม เพื่อให้รันซ้ำแล้วได้ชุดข้อมูลเดิม

    จะไม่สร้างข้อมูลปลอมและจะไม่สุ่มซ้ำ (ไม่มี oversampling) ดังนั้นจำนวนแถว
    ของคลาส Worms จะยังคงเป็น 88 แถวทั้งหมดที่มีใน train
    """

    if max_rows <= 0:
        raise ValueError("max_rows ต้องมากกว่า 0")
    if max_rows >= len(x_train):
        print("ไม่สุ่ม train เพราะจำนวนข้อมูลไม่เกิน train_sample_size")
        return x_train.reset_index(drop=True), y_train.reset_index(drop=True)

    labels = pd.Series(y_train).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    grouped_positions = {
        label: positions.to_numpy()
        for label, positions in labels.groupby(labels, sort=True).groups.items()
    }

    # สำรองจำนวนขั้นต่ำให้ทุกคลาสก่อน
    mandatory_counts = {
        label: min(len(positions), min_per_class)
        for label, positions in grouped_positions.items()
    }
    mandatory_total = sum(mandatory_counts.values())
    if mandatory_total > max_rows:
        raise ValueError(
            "train_sample_size น้อยกว่าจำนวนขั้นต่ำรวมของทุกคลาส "
            f"({mandatory_total:,} > {max_rows:,})"
        )

    selected_counts = mandatory_counts.copy()
    remaining_budget = max_rows - mandatory_total
    remaining_capacity = {
        label: len(positions) - mandatory_counts[label]
        for label, positions in grouped_positions.items()
    }
    total_capacity = sum(remaining_capacity.values())

    # แจก budget ที่เหลือตามจำนวนข้อมูลของแต่ละคลาส
    if remaining_budget and total_capacity:
        raw_allocations = {
            label: remaining_budget * capacity / total_capacity
            for label, capacity in remaining_capacity.items()
        }
        allocations = {
            label: min(remaining_capacity[label], int(np.floor(raw_allocations[label])))
            for label in grouped_positions
        }
        leftover = remaining_budget - sum(allocations.values())

        # แจกเศษที่เหลือให้คลาสที่มี fractional remainder สูงสุด
        # และใช้ label เป็น tie-breaker เพื่อให้ผล deterministic
        order = sorted(
            grouped_positions,
            key=lambda label: (
                -(raw_allocations[label] - np.floor(raw_allocations[label])),
                label,
            ),
        )
        for label in order:
            if leftover == 0:
                break
            if allocations[label] < remaining_capacity[label]:
                allocations[label] += 1
                leftover -= 1

        for label, amount in allocations.items():
            selected_counts[label] += amount

    selected_positions = []
    for label, positions in grouped_positions.items():
        # สุ่มภายในแต่ละ class หลังจากกำหนดจำนวนแล้ว
        chosen = rng.choice(positions, size=selected_counts[label], replace=False)
        selected_positions.extend(chosen.tolist())

    # สุ่มลำดับของแถวอีกครั้ง ไม่ให้ข้อมูลเรียงตาม class
    selected_positions = np.asarray(selected_positions, dtype=int)
    rng.shuffle(selected_positions)

    sampled_x = x_train.iloc[selected_positions].reset_index(drop=True)
    sampled_y = labels.iloc[selected_positions].reset_index(drop=True)

    distribution = sampled_y.value_counts().sort_index()
    print(
        f"สุ่ม train แบบ stratified: {len(x_train):,} -> {len(sampled_x):,} แถว"
    )
    print("จำนวนตัวอย่างต่อ class หลังสุ่ม:")
    print(distribution.to_string())
    return sampled_x, sampled_y


def load_base_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """โหลด train/val/test ของ RF-base เพื่อคงชื่อเดิมที่ใช้ใน baseline"""

    return load_splits()


def save_test_details(
    y_test: pd.Series,
    y_pred,
    report_dir: Path,
    report_prefix: str = "rf_base",
) -> None:
    """บันทึก Recall ราย class และ confusion matrix ของ test

    Macro F1 เป็น metric หลักสำหรับเลือกโมเดล แต่ Recall ราย class จะช่วยชี้ว่า
    attack class ใดถูกพลาด จึงบันทึกเป็นผลประกอบที่ตรวจสอบย้อนหลังได้
    """

    report_dir.mkdir(parents=True, exist_ok=True)

    recall_values = recall_score(
        y_test,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )
    support = pd.Series(y_test).value_counts().reindex(CLASSES, fill_value=0)
    recall_report = pd.DataFrame(
        {
            "class": CLASSES,
            "recall": recall_values,
            "support": support.to_numpy(),
        }
    )
    recall_report.to_csv(
        report_dir / f"{report_prefix}_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(y_test, y_pred, labels=CLASSES)
    confusion_report = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    confusion_report.index.name = "actual"
    confusion_report.to_csv(
        report_dir / f"{report_prefix}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def _latest_result(csv_path: Path, split: str) -> pd.Series | None:
    """อ่านผลล่าสุดของ split ที่ต้องการจากไฟล์ผลลัพธ์"""

    if not csv_path.exists():
        return None
    results = pd.read_csv(csv_path)
    rows = results[results["split"] == split]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _save_figure(fig: plt.Figure, figure_path: Path) -> None:
    """บันทึกกราฟด้วยรูปแบบเดียวกัน เพื่อให้ใช้ในสไลด์ได้ทันที"""

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(report_dir: Path) -> None:
    """สร้างกราฟสรุปผล Random Forest จาก CSV ที่มีอยู่แล้ว

    ฟังก์ชันนี้ไม่ train โมเดลและไม่เรียก evaluate จึงไม่เพิ่มแถวใหม่ใน results CSV
    และไม่ทำให้ test ถูกวัดซ้ำ กราฟจะถูกสร้างเท่าที่มีไฟล์ผลลัพธ์รองรับ
    """

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1) กราฟเปรียบเทียบ metrics ของ RF-base, TTL และ Geo
    # --------------------------------------------------------
    model_files = {
        "RF-base": report_dir / "results_rf.csv",
        "RF-base + TTL": report_dir / "results_rf_ttl.csv",
        "RF-base + Geo": report_dir / "results_rf_geo.csv",
    }
    comparison_rows = []
    for label, csv_path in model_files.items():
        row = _latest_result(csv_path, "test")
        if row is not None:
            comparison_rows.append(
                {
                    "model": label,
                    "Macro F1": float(row["macro_f1"]),
                    "PR-AUC": float(row["pr_auc"]),
                    "Macro Recall": float(row["recall"]),
                }
            )

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows).set_index("model")
        ax = comparison.plot(
            kind="bar",
            figsize=(11, 6.5),
            color=["#2563eb", "#f59e0b", "#10b981"],
            width=0.78,
            edgecolor="white",
        )
        ax.set_title("Random Forest: Test Metrics Comparison", fontsize=16, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(title="Metric", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
        _save_figure(fig=ax.get_figure(), figure_path=figure_dir / "rf_metrics_comparison.png")

    # --------------------------------------------------------
    # 2) กราฟผล tuning ทั้ง 4 configurations บน validation
    # --------------------------------------------------------
    base_results_path = report_dir / "results_rf.csv"
    if base_results_path.exists():
        base_results = pd.read_csv(base_results_path)
        validation_rows = base_results[base_results["split"] == "val"].tail(len(SEARCH_PARAMS))
        if len(validation_rows) == len(SEARCH_PARAMS):
            tuning = pd.DataFrame(
                {
                    "Configuration": [f"Config {i}" for i in range(1, len(validation_rows) + 1)],
                    "Macro F1": validation_rows["macro_f1"].astype(float).to_numpy(),
                    "PR-AUC": validation_rows["pr_auc"].astype(float).to_numpy(),
                    "Macro Recall": validation_rows["recall"].astype(float).to_numpy(),
                }
            ).set_index("Configuration")
            ax = tuning.plot(
                kind="bar",
                figsize=(11, 6.5),
                color=["#2563eb", "#8b5cf6", "#10b981"],
                width=0.78,
                edgecolor="white",
            )
            ax.set_title("RF-base Hyperparameter Tuning on Validation", fontsize=16, weight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Validation score", fontsize=11)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(title="Metric", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
            _save_figure(fig=ax.get_figure(), figure_path=figure_dir / "rf_tuning_validation.png")

    # --------------------------------------------------------
    # 3) Recall ราย class พร้อมจำนวนตัวอย่างใน test
    # --------------------------------------------------------
    recall_path = report_dir / "rf_base_class_recall.csv"
    if recall_path.exists():
        recall = pd.read_csv(recall_path).sort_values("recall", ascending=True)
        colors = ["#dc2626" if value < 0.5 else "#f59e0b" if value < 0.7 else "#2563eb" for value in recall["recall"]]
        fig, ax = plt.subplots(figsize=(11, 7.5))
        bars = ax.barh(recall["class"], recall["recall"], color=colors, edgecolor="white")
        ax.set_title("RF-base Recall by Class on Test", fontsize=16, weight="bold")
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value, support in zip(bars, recall["recall"], recall["support"]):
            ax.text(
                min(value + 0.015, 0.84),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}  (n={int(support):,})",
                va="center",
                fontsize=9,
            )
        _save_figure(fig, figure_dir / "rf_class_recall.png")

    # --------------------------------------------------------
    # 4) Confusion matrix แบบ normalize ต่อแถว
    # --------------------------------------------------------
    matrix_path = report_dir / "rf_base_confusion_matrix.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        counts = matrix.to_numpy(dtype=float)
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)
        labels = list(matrix.index)

        fig, ax = plt.subplots(figsize=(10.5, 9))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title("RF-base Confusion Matrix (Row-normalized Test Recall)", fontsize=16, weight="bold")
        ax.set_xlabel("Predicted class", fontsize=11)
        ax.set_ylabel("Actual class", fontsize=11)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
        threshold = normalized.max() / 2
        for row_index in range(normalized.shape[0]):
            for col_index in range(normalized.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    f"{normalized[row_index, col_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if normalized[row_index, col_index] > threshold else "black",
                )
        _save_figure(fig, figure_dir / "rf_confusion_matrix.png")

        # เก็บ count matrix อีกภาพสำหรับอธิบายจำนวนแถวจริงใน confusion matrix
        fig, ax = plt.subplots(figsize=(10.5, 9))
        image = ax.imshow(counts, cmap="Oranges", aspect="auto")
        ax.set_title("RF-base Confusion Matrix (Raw Counts)", fontsize=16, weight="bold")
        ax.set_xlabel("Predicted class", fontsize=11)
        ax.set_ylabel("Actual class", fontsize=11)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of samples")
        for row_index in range(counts.shape[0]):
            for col_index in range(counts.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    f"{int(counts[row_index, col_index]):,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if counts[row_index, col_index] > counts.max() / 2 else "black",
                )
        _save_figure(fig, figure_dir / "rf_confusion_matrix_counts.png")

    # --------------------------------------------------------
    # 5) กราฟ gap ของ TTL และ Geo เทียบกับ RF-base
    # --------------------------------------------------------
    if len(comparison_rows) >= 2 and "RF-base" in comparison.index:
        gaps = comparison.loc[[label for label in comparison.index if label != "RF-base"]].copy()
        gaps = gaps - comparison.loc["RF-base"]
        ax = gaps.plot(
            kind="bar",
            figsize=(10, 6.2),
            color=["#2563eb", "#f59e0b", "#10b981"],
            width=0.72,
            edgecolor="white",
        )
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title("Feature Ablation Gap Compared with RF-base", fontsize=16, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Change in score", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(title="Metric", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        for container in ax.containers:
            ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=9)
        _save_figure(fig=ax.get_figure(), figure_path=figure_dir / "rf_ablation_gap.png")

    print(f"สร้างกราฟ Random Forest แล้วที่ {figure_dir}")


def write_summary(
    report_dir: Path,
    selected_params: dict,
    final_params: dict,
    val_result: dict,
    test_result: dict,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """เขียนสรุปผลเป็น Markdown สำหรับอธิบายงาน"""

    base_features = ", ".join(FEAT["base"])
    summary = f"""# Random Forest - RF-base

## วัตถุประสงค์

การทดลองนี้เป็น baseline รอบแรก ใช้ feature พฤติกรรมของ network traffic
โดยไม่ใช้ IP, Geo และ TTL เพื่อหลีกเลี่ยงการที่โมเดลจำลักษณะเฉพาะของห้องแล็บ

## การแบ่งข้อมูล

- แบ่งตามเวลาใน `progress4/split.py`
- Train: {n_train:,} แถว
- Validation: {n_val:,} แถว
- Test: {n_test:,} แถว
- Random seed: `{SEED}`
- เลือกโมเดลด้วย Macro F1 บน validation เท่านั้น
- Test ใช้วัดหลังจากเลือก configuration แล้ว

## Feature ที่ใช้

ใช้เฉพาะ feature ใน `features.yaml` ส่วน `base`:

`{base_features}`

ไม่ได้ใช้:

- Geo: `src_country`, `dst_country`, `src_asn`, `dst_asn`
- TTL: `sttl`, `dttl`, `ct_state_ttl`
- IP, port และ timestamp

## Configuration ที่เลือกจาก validation

```text
{selected_params}
```

โมเดลรอบ final ใช้ configuration เดียวกับที่เลือกจาก validation:

```text
{final_params}
```

## ผลบน validation

- Macro F1: **{val_result['macro_f1']:.4f}**
- Macro PR-AUC: **{val_result['pr_auc']:.4f}**
- Macro Recall: **{val_result['recall']:.4f}**

## ผลบน test

- Macro F1: **{test_result['macro_f1']:.4f}**
- Macro PR-AUC: **{test_result['pr_auc']:.4f}**
- Macro Recall: **{test_result['recall']:.4f}**

## การอ่านผล

Macro F1 เป็น metric หลักที่ใช้เลือก configuration เพราะข้อมูลมีความไม่สมดุล
รุนแรงและมีทั้งหมด 10 classes ส่วน Macro PR-AUC และ Recall ราย class ใช้เป็น
หลักฐานประกอบเพื่อดูว่าโมเดลพลาด attack class ใดบ้าง

การทดลองนี้ยังไม่ใช่การพิสูจน์ว่าโมเดลไม่มี leakage ทั้งหมด เพราะต้องทำรอบ
เปรียบเทียบกับโมเดลที่เพิ่ม TTL และ Geo ในขั้นตอนถัดไป

รายละเอียด Recall ราย class อยู่ที่ `reports/rf_base_class_recall.csv`
และ confusion matrix อยู่ที่ `reports/rf_base_confusion_matrix.csv`
"""
    (report_dir / "model_rf.md").write_text(summary, encoding="utf-8")


def run_base() -> dict:
    # รอบที่ 1: RF-base เดิมของ Pimmy
    # เก็บ function นี้ไว้ ไม่รวมกับ ablation เพื่อให้รายงานแต่ละรอบแยกกัน
    """รัน RF-base ตั้งแต่ validation จนถึง final test"""

    report_dir = Path(__file__).resolve().parents[2] / "reports"
    x_train, y_train, x_val, y_val, x_test, y_test = load_base_splits()
    feature_columns = list(x_train.columns)

    print("RF-base: ใช้ feature จำนวน", len(feature_columns), "คอลัมน์")
    print("RF-base: train/val/test =", len(x_train), len(x_val), len(x_test))
    print("RF-base: เริ่มจูนด้วย Macro F1 บน validation")

    best_params = None
    best_val_result = None

    # ทดลองทีละ configuration โดย fit เฉพาะ train และประเมินบน val
    for trial, params in enumerate(SEARCH_PARAMS, start=1):
        print(f"\nทดลอง configuration {trial}/{len(SEARCH_PARAMS)}")
        print(params)

        model = make_model(feature_columns, params)
        model.fit(x_train, y_train)

        # evaluate บันทึกผลลง reports/results_rf.csv โดยไม่แก้ metrics.py
        val_result = evaluate(
            y_val,
            model.predict(x_val),
            model.predict_proba(x_val),
            model.named_steps["model"].classes_,
            model="rf",
            split="val",
            params=params,
        )

        print(
            f"ผล val: Macro F1={val_result['macro_f1']:.4f}, "
            f"PR-AUC={val_result['pr_auc']:.4f}, "
            f"Recall={val_result['recall']:.4f}"
        )

        # เกณฑ์หลักคือ Macro F1; ถ้าเท่ากันใช้ PR-AUC เป็นตัวตัดสินรอง
        is_better = best_val_result is None or (
            val_result["macro_f1"], val_result["pr_auc"]
        ) > (
            best_val_result["macro_f1"], best_val_result["pr_auc"]
        )
        if is_better:
            best_params = params.copy()
            best_val_result = val_result

    if best_params is None or best_val_result is None:
        raise RuntimeError("ไม่สามารถเลือก configuration จาก validation ได้")

    # ใช้ hyperparameter เดิมที่เลือกจาก validation โดยไม่เปลี่ยนค่า
    # เพื่อให้ผล test ตรงกับกติกาของการทดลอง
    final_params = best_params.copy()

    print("\nเลือก configuration จาก val:", best_params)
    print("ฝึกโมเดลสุดท้ายด้วย train + val และวัด test ครั้งเดียว")

    # รวม train กับ val หลังจบการเลือก configuration แล้วเท่านั้น
    x_train_val = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    final_model = make_model(feature_columns, final_params)
    final_model.fit(x_train_val, y_train_val)

    y_test_pred = final_model.predict(x_test)
    y_test_score = final_model.predict_proba(x_test)
    test_result = evaluate(
        y_test,
        y_test_pred,
        y_test_score,
        final_model.named_steps["model"].classes_,
        model="rf",
        split="test",
        params=final_params,
    )

    save_test_details(y_test, y_test_pred, report_dir)
    write_summary(
        report_dir,
        best_params,
        final_params,
        best_val_result,
        test_result,
        len(x_train),
        len(x_val),
        len(x_test),
    )
    generate_figures(report_dir)

    print("\nผล test:")
    print(f"Macro F1    = {test_result['macro_f1']:.4f}")
    print(f"Macro PR-AUC = {test_result['pr_auc']:.4f}")
    print(f"Macro Recall = {test_result['recall']:.4f}")
    print(f"บันทึกผลแล้วที่ {report_dir}")
    # ส่งค่าที่ชนะจาก validation ให้ run_all() ใช้กับ TTL/Geo ต่อโดยตรง
    return final_params


# ============================================================
# รอบที่ 2: Ablation study
#
# ส่วนนี้เป็นการทดลองต่อจาก RF-base โดยเพิ่ม TTL หรือ Geo ทีละชุด
# โค้ดรอบที่ 1 ด้านบนยังคงอยู่และไม่ได้ถูกลบ
# ============================================================


def run_fixed_ablation(
    experiment_name: str,
    extra_features: list[str],
    feature_description: str,
    report_dir: Path,
    train_sample_size: int | None = None,
    model_name_override: str | None = None,
    selected_params: dict | None = None,
) -> tuple[dict, dict]:
    """รันหนึ่งโมเดลของ ablation study ด้วย hyperparameter ที่ล็อกไว้

    การทดลองนี้ไม่ได้จูน parameter ใหม่ เพราะต้องการตอบคำถามเฉพาะว่า
    การเพิ่ม TTL หรือ Geo ทำให้คะแนนเปลี่ยนจาก RF-base เท่าไร

    ขั้นตอนยังคงแยก train/val/test ตามเดิม:
        1. fit train และวัด val
        2. ใช้ configuration เดิม fit train + val
        3. วัด test เพียงครั้งเดียว
    """

    model_name = model_name_override or f"rf_{experiment_name}"
    # run_all() จะส่งค่าที่ชนะจาก RF-base มาให้โดยตรง
    # ถ้ารัน ablation แยก จะใช้ configuration ล่าสุดที่เก็บไว้ใน ABLATION_PARAMS
    experiment_params = (selected_params or ABLATION_PARAMS).copy()
    report_prefix = model_name
    x_train, y_train, x_val, y_val, x_test, y_test = load_splits(
        extra_features,
        train_sample_size=train_sample_size,
    )
    feature_columns = list(x_train.columns)

    print(f"\nเริ่ม {model_name}: {feature_description}")
    print("จำนวน feature:", len(feature_columns))
    print("train/val/test =", len(x_train), len(x_val), len(x_test))
    print("ใช้ hyperparameter เดียวกับ RF-base:", experiment_params)

    # วัด validation ก่อน เพื่อบันทึกผลของชุด feature นี้
    validation_model = make_model(feature_columns, experiment_params)
    validation_model.fit(x_train, y_train)
    val_result = evaluate(
        y_val,
        validation_model.predict(x_val),
        validation_model.predict_proba(x_val),
        validation_model.named_steps["model"].classes_,
        model=model_name,
        split="val",
        params=experiment_params,
    )
    print(
        f"ผล {model_name} val: Macro F1={val_result['macro_f1']:.4f}, "
        f"PR-AUC={val_result['pr_auc']:.4f}, "
        f"Recall={val_result['recall']:.4f}"
    )

    # หลังจากวัด validation แล้ว จึงรวม train กับ val เพื่อฝึกโมเดลสุดท้าย
    x_train_val = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    final_model = make_model(feature_columns, experiment_params)
    final_model.fit(x_train_val, y_train_val)

    y_test_pred = final_model.predict(x_test)
    y_test_score = final_model.predict_proba(x_test)
    test_result = evaluate(
        y_test,
        y_test_pred,
        y_test_score,
        final_model.named_steps["model"].classes_,
        model=model_name,
        split="test",
        params=experiment_params,
    )
    save_test_details(y_test, y_test_pred, report_dir, report_prefix)

    print(
        f"ผล {model_name} test: Macro F1={test_result['macro_f1']:.4f}, "
        f"PR-AUC={test_result['pr_auc']:.4f}, "
        f"Recall={test_result['recall']:.4f}"
    )
    return val_result, test_result


def write_sampled_summary(
    report_dir: Path,
    val_result: dict,
    test_result: dict,
    sample_size: int,
) -> None:
    """เขียนรายงานรอบที่ 3 สำหรับการลดทรัพยากรด้วย sampled training"""

    base_test = latest_test_result(report_dir / "results_rf.csv")
    summary = f"""# รอบที่ 3: Resource-aware Random Forest

## วัตถุประสงค์

ตรวจสอบว่าสามารถลดจำนวนข้อมูลที่ใช้ train เหลือประมาณ {sample_size:,} แถว
โดยยังรักษาประสิทธิภาพของโมเดลไว้ได้หรือไม่

การสุ่มมีเงื่อนไขแบบ stratified:

- สุ่มเฉพาะ train เท่านั้น
- validation และ test ใช้ข้อมูลเต็มตาม time split เดิม
- ใช้ seed `{SEED}`
- กันจำนวนขั้นต่ำให้ทุก class ก่อนแจกจำนวนที่เหลือตามสัดส่วน
- ไม่ oversampling และไม่สร้างข้อมูลปลอม

## ผลลัพธ์

| Model | Macro F1 | Macro PR-AUC | Macro Recall |
|---|---:|---:|---:|
| RF-base เต็มข้อมูล | {float(base_test['macro_f1']):.4f} | {float(base_test['pr_auc']):.4f} | {float(base_test['recall']):.4f} |
| RF-base sampled | {float(test_result['macro_f1']):.4f} | {float(test_result['pr_auc']):.4f} | {float(test_result['recall']):.4f} |

## ผลบน validation

- Macro F1: **{float(val_result['macro_f1']):.4f}**
- Macro PR-AUC: **{float(val_result['pr_auc']):.4f}**
- Macro Recall: **{float(val_result['recall']):.4f}**

## การตีความ

ผลรอบนี้ใช้ตรวจความเหมาะสมของวิธีลดทรัพยากร ไม่ได้แทนที่ผลรอบที่ 1
โดยอัตโนมัติ หาก Macro F1 และ Recall ใกล้เคียงกับ RF-base เต็มข้อมูล
สามารถใช้ sampled training ในการทดลองต่อไปได้ โดยต้องเก็บวิธีสุ่มและ seed
ไว้ในรายงานทุกครั้ง

รายละเอียดผลลัพธ์อยู่ที่:

- `reports/results_rf_sampled.csv`
- `reports/rf_sampled_class_recall.csv`
- `reports/rf_sampled_confusion_matrix.csv`
"""
    # โครงการกำหนดให้เหลือรายงาน Random Forest เพียงไฟล์เดียวคือ model_rf.md
    # จึงไม่สร้างรายงาน .md แยกสำหรับ sampled แม้โหมดนี้จะยังเก็บไว้เป็นทางเลือกสำรอง
    # ผลคะแนนของ sampled ยังคงอยู่ใน CSV ที่ metrics.py เขียนให้ตรวจสอบได้
    print("ไม่ได้สร้างรายงาน sampled .md แยก เพื่อคงรายงาน Random Forest ไว้ไฟล์เดียว")


def run_sampled_base(sample_size: int = 100_000) -> None:
    """รัน RF-base ด้วย train แบบ stratified sample เพื่อประเมินข้อจำกัดทรัพยากร"""

    report_dir = Path(__file__).resolve().parents[2] / "reports"
    val_result, test_result = run_fixed_ablation(
        experiment_name="sampled",
        extra_features=[],
        feature_description=f"RF-base sampled ({sample_size:,} แถว)",
        report_dir=report_dir,
        train_sample_size=sample_size,
        model_name_override="rf_sampled",
    )
    write_sampled_summary(report_dir, val_result, test_result, sample_size)
    print(f"\nผล sampled ถูกเก็บใน CSV ภายใต้ {report_dir} โดยไม่สร้าง .md เพิ่ม")


def latest_test_result(csv_path: Path) -> dict:
    """อ่านผล test ล่าสุดของโมเดลจาก CSV ที่ metrics.py สร้างไว้"""

    if not csv_path.exists():
        raise FileNotFoundError(f"ไม่พบผลลัพธ์ที่ต้องใช้เปรียบเทียบ: {csv_path}")
    results = pd.read_csv(csv_path)
    test_rows = results[results["split"] == "test"]
    if test_rows.empty:
        raise ValueError(f"ยังไม่มีผล test ใน {csv_path}")
    return test_rows.iloc[-1].to_dict()


def write_ablation_summary(
    report_dir: Path,
    ttl_test: dict,
    geo_test: dict,
) -> None:
    """เขียนตารางเปรียบเทียบ Base, Base+TTL และ Base+Geo

    leakage gap คำนวณจากผล test ของ Base และ Base+TTL
    ส่วน geo gap ใช้ดูผลของ metadata จาก IP ต่อคะแนนโมเดล
    """

    base_test = latest_test_result(report_dir / "results_rf.csv")
    ttl_gap = float(ttl_test["macro_f1"]) - float(base_test["macro_f1"])
    geo_gap = float(geo_test["macro_f1"]) - float(base_test["macro_f1"])

    summary = f"""# Random Forest Ablation Study

## วัตถุประสงค์

เปรียบเทียบผลของการเพิ่ม feature ที่มีความเสี่ยงเป็น leakage หรือ shortcut
โดยใช้ Random Forest hyperparameter เดียวกันกับ RF-base:

- RF-base: traffic behavior เท่านั้น
- RF-base + TTL: เพิ่ม `sttl`, `dttl`, `ct_state_ttl`
- RF-base + Geo: เพิ่ม `src_country`, `dst_country`, `src_asn`, `dst_asn`

การเปรียบเทียบนี้ใช้ผล test หลังจากแต่ละโมเดล train ด้วย train + val แล้ว
โดยไม่ได้ใช้ test ในการเลือก configuration

## ผลลัพธ์

| Model | Macro F1 | Macro PR-AUC | Macro Recall |
|---|---:|---:|---:|
| RF-base | {float(base_test['macro_f1']):.4f} | {float(base_test['pr_auc']):.4f} | {float(base_test['recall']):.4f} |
| RF-base + TTL | {float(ttl_test['macro_f1']):.4f} | {float(ttl_test['pr_auc']):.4f} | {float(ttl_test['recall']):.4f} |
| RF-base + Geo | {float(geo_test['macro_f1']):.4f} | {float(geo_test['pr_auc']):.4f} | {float(geo_test['recall']):.4f} |

## ช่องว่างของคะแนน

- TTL leakage gap (Macro F1): **{ttl_gap:+.4f}**
- Geo shortcut gap (Macro F1): **{geo_gap:+.4f}**

การเพิ่ม TTL ทำให้ Macro F1 เปลี่ยนจาก Base ตามค่า TTL leakage gap ด้านบน
คะแนนที่เพิ่มขึ้นอาจมาจากลักษณะเฉพาะของเครื่องหรือสภาพแวดล้อมในแลป
จึงไม่ควรสรุปว่าโมเดลจะทำงานดีในระบบจริงโดยอัตโนมัติ

การเพิ่ม Geo ต้องอ่านตามเครื่องหมายของ Geo shortcut gap:
ถ้าค่าเป็นบวก แปลว่า Geo อาจช่วยจากความสัมพันธ์ระหว่างประเทศ/ASN กับ label
แต่ถ้าค่าเป็นศูนย์หรือติดลบ เช่นผลรอบนี้ แปลว่ายังไม่พบประโยชน์ของ Geo
ภายใต้ time split และ hyperparameter ที่ใช้

ผล Recall ราย class และ confusion matrix แยกตามโมเดลอยู่ในไฟล์:

- `reports/rf_base_class_recall.csv`
- `reports/rf_ttl_class_recall.csv`
- `reports/rf_geo_class_recall.csv`
- `reports/rf_base_confusion_matrix.csv`
- `reports/rf_ttl_confusion_matrix.csv`
- `reports/rf_geo_confusion_matrix.csv`
"""
    # โครงการกำหนดให้รวมคำอธิบายทั้งหมดไว้ใน reports/model_rf.md ไฟล์เดียว
    # จึงไม่เขียน rf_ablation_summary.md กลับขึ้นมาใหม่เมื่อรันโค้ดซ้ำ
    print("ไม่ได้สร้าง rf_ablation_summary.md แยก เพราะใช้ reports/model_rf.md เป็นรายงานรวม")


def run_ablation(selected_params: dict | None = None) -> None:
    """รัน Base+TTL และ Base+Geo เพื่อวัด leakage/shortcut gap"""

    report_dir = Path(__file__).resolve().parents[2] / "reports"

    # ทำ TTL ก่อนตามลำดับที่วางแผนไว้ เพราะเป็น leakage ที่ EDA พบชัดที่สุด
    _, ttl_test = run_fixed_ablation(
        experiment_name="ttl",
        extra_features=FEAT["check"],
        feature_description="RF-base + TTL (ตรวจ lab leakage)",
        report_dir=report_dir,
        selected_params=selected_params,
    )

    # ทำ Geo แยกจาก TTL เพื่อวัด geo shortcut โดยไม่ปนกับผล TTL
    _, geo_test = run_fixed_ablation(
        experiment_name="geo",
        extra_features=FEAT["geo"],
        feature_description="RF-base + Geo (ตรวจ IP-derived shortcut)",
        report_dir=report_dir,
        selected_params=selected_params,
    )

    write_ablation_summary(report_dir, ttl_test, geo_test)
    generate_figures(report_dir)
    print(f"\nผล ablation ถูกเก็บใน CSV ภายใต้ {report_dir} และอธิบายรวมใน model_rf.md")


def run_all() -> None:
    """รัน workflow ตาม README: RF-base แล้วจึงทำ TTL/Geo ablation"""

    # ให้ค่าที่ชนะจาก RF-base ถูกส่งต่อไปยัง TTL/Geo โดยตรง
    selected_params = run_base()
    run_ablation(selected_params)


def main() -> None:
    """เลือกโหมดการทดลองจาก command line โดย default เป็น workflow ทั้งหมด"""

    parser = argparse.ArgumentParser(description="Run Pimmy Random Forest experiments")
    parser.add_argument(
        "--experiment",
        choices=["all", "base", "ablation", "sampled", "figures"],
        default="all",
        help=(
            "all = base + ablation; base = RF-base tuning; "
            "ablation = RF-base+TTL และ RF-base+Geo; sampled = ทางเลือกสำรอง; "
            "figures = สร้างกราฟจากผล CSV โดยไม่ train ใหม่"
        ),
    )
    args = parser.parse_args()

    if args.experiment == "all":
        run_all()
    elif args.experiment == "base":
        run_base()
    elif args.experiment == "ablation":
        run_ablation()
    elif args.experiment == "figures":
        generate_figures(Path(__file__).resolve().parents[2] / "reports")
    else:
        run_sampled_base()


if __name__ == "__main__":
    main()
