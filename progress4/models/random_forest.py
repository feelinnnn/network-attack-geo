"""Progress 4: Random Forest ของ Pimmy."""

import argparse
import ast
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import CLASSES, FEAT, SEED, evaluate, load_split


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
    {
        "n_estimators": 60,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 60,
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 45,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 60,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
]

ABLATION_PARAMS = {
    "n_estimators": 30,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """สร้าง OneHotEncoder ที่รองรับหลายเวอร์ชัน"""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_model(feature_columns: list[str], params: dict) -> Pipeline:
    """สร้าง pipeline ของ preprocessing และ Random Forest"""

    categorical = [
        column for column in FEAT["categorical"] if column in feature_columns
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """โหลด train/val/test พร้อม feature เพิ่มถ้ามี"""

    x_train, y_train = load_split("train", extra=extra)
    x_val, y_val = load_split("val", extra=extra)
    x_test, y_test = load_split("test", extra=extra)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_base_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """โหลดข้อมูลของ RF-base"""

    return load_splits()


def save_test_details(
    y_test: pd.Series,
    y_pred,
    report_dir: Path,
    report_prefix: str = "rf_base",
) -> None:
    """บันทึก recall ราย class และ confusion matrix"""

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
    """อ่านผลล่าสุดของ split"""

    if not csv_path.exists():
        return None
    results = pd.read_csv(csv_path)
    rows = results[results["split"] == split]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _saved_validation_result(report_dir: Path, params: dict) -> dict | None:
    """อ่านผล validation เดิมของ configuration"""

    result_path = report_dir / "results_rf.csv"
    if not result_path.exists():
        return None

    results = pd.read_csv(result_path)
    rows = results[(results["model"] == "rf") & (results["split"] == "val")]
    for _, row in rows.iloc[::-1].iterrows():
        try:
            saved_params = ast.literal_eval(row["params"])
        except (ValueError, SyntaxError):
            continue
        if saved_params == params:
            return {
                "macro_f1": float(row["macro_f1"]),
                "pr_auc": float(row["pr_auc"]),
                "recall": float(row["recall"]),
            }
    return None


def _best_saved_params(report_dir: Path) -> dict | None:
    """อ่าน configuration ที่ดีที่สุดจาก validation"""

    result_path = report_dir / "results_rf.csv"
    if not result_path.exists():
        return None
    results = pd.read_csv(result_path)
    rows = results[(results["model"] == "rf") & (results["split"] == "val")]
    if rows.empty:
        return None
    best_row = rows.sort_values(["macro_f1", "pr_auc"], ascending=False).iloc[0]
    return ast.literal_eval(best_row["params"])


def _save_figure(fig: plt.Figure, figure_path: Path) -> None:
    """บันทึกกราฟ"""

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(report_dir: Path) -> None:
    """สร้างกราฟจาก CSV"""

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

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


def run_base() -> dict:
    """จูน RF-base และวัดผล test"""

    report_dir = Path(__file__).resolve().parents[2] / "reports"
    x_train, y_train, x_val, y_val, x_test, y_test = load_base_splits()
    feature_columns = list(x_train.columns)

    print("RF-base: ใช้ feature จำนวน", len(feature_columns), "คอลัมน์")
    print("RF-base: train/val/test =", len(x_train), len(x_val), len(x_test))
    print("RF-base: เริ่มจูนด้วย Macro F1 บน validation")

    best_params = None
    best_val_result = None

    for trial, params in enumerate(SEARCH_PARAMS, start=1):
        print(f"\nทดลอง configuration {trial}/{len(SEARCH_PARAMS)}")
        print(params)

        val_result = _saved_validation_result(report_dir, params)
        if val_result is not None:
            print("ใช้ผล validation ที่บันทึกไว้แล้วสำหรับ configuration นี้")
        else:
            model = make_model(feature_columns, params)
            model.fit(x_train, y_train)
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

    final_params = best_params.copy()

    print("\nเลือก configuration จาก val:", best_params)
    print("ฝึกโมเดลสุดท้ายด้วย train + val และวัด test ครั้งเดียว")

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
    generate_figures(report_dir)

    print("\nผล test:")
    print(f"Macro F1    = {test_result['macro_f1']:.4f}")
    print(f"Macro PR-AUC = {test_result['pr_auc']:.4f}")
    print(f"Macro Recall = {test_result['recall']:.4f}")
    print(f"บันทึกผลแล้วที่ {report_dir}")
    return final_params


def run_fixed_ablation(
    experiment_name: str,
    extra_features: list[str],
    feature_description: str,
    report_dir: Path,
    model_name_override: str | None = None,
    selected_params: dict | None = None,
) -> tuple[dict, dict]:
    """รันหนึ่งรอบของ ablation"""

    model_name = model_name_override or f"rf_{experiment_name}"
    experiment_params = (selected_params or ABLATION_PARAMS).copy()
    report_prefix = model_name
    x_train, y_train, x_val, y_val, x_test, y_test = load_splits(extra_features)
    feature_columns = list(x_train.columns)

    print(f"\nเริ่ม {model_name}: {feature_description}")
    print("จำนวน feature:", len(feature_columns))
    print("train/val/test =", len(x_train), len(x_val), len(x_test))
    print("ใช้ hyperparameter เดียวกับ RF-base:", experiment_params)

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


def run_ablation(selected_params: dict | None = None) -> None:
    """รัน TTL และ Geo ablation"""

    report_dir = Path(__file__).resolve().parents[2] / "reports"
    if selected_params is None:
        selected_params = _best_saved_params(report_dir)
        if selected_params is None:
            raise RuntimeError("ไม่พบ hyperparameter ที่เลือกจาก RF-base validation")
        print("ใช้ hyperparameter ล่าสุดที่ชนะจาก RF-base validation:", selected_params)

    _, ttl_test = run_fixed_ablation(
        experiment_name="ttl",
        extra_features=FEAT["check"],
        feature_description="RF-base + TTL (ตรวจ lab leakage)",
        report_dir=report_dir,
        selected_params=selected_params,
    )

    _, geo_test = run_fixed_ablation(
        experiment_name="geo",
        extra_features=FEAT["geo"],
        feature_description="RF-base + Geo (ตรวจ IP-derived shortcut)",
        report_dir=report_dir,
        selected_params=selected_params,
    )

    generate_figures(report_dir)
    print(f"\nผล ablation ถูกเก็บใน CSV ภายใต้ {report_dir}")


def run_all() -> None:
    """รัน workflow ทั้งหมด"""

    selected_params = run_base()
    run_ablation(selected_params)


def main() -> None:
    """เลือกโหมดการรัน"""

    parser = argparse.ArgumentParser(description="Run Pimmy Random Forest experiments")
    parser.add_argument(
        "--experiment",
        choices=["all", "base", "ablation", "figures"],
        default="all",
        help=(
            "all = base + ablation; base = RF-base tuning; "
            "ablation = RF-base+TTL และ RF-base+Geo; "
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


if __name__ == "__main__":
    main()
