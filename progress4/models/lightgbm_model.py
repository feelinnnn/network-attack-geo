"""Progress 4 — LightGBM (Keen)

Input : data/processed/train.parquet, val.parquet, test.parquet  (จาก split.py)
Output: reports/results_lgbm.csv          (Base Model)
        reports/results_lgbm_geo.csv      (Base + Geo features)
        reports/results_lgbm_ttl.csv      (รอบมี TTL เพื่อวัด leakage)
        reports/figures/lgbm_*.png        (กราฟ Feature Importance และ Confusion Matrix)
        reports/model_lgbm.md             (สรุปผล)

TTL: load_split ตัดออกให้แล้ว ห้ามเอากลับเข้ามาตอนจูน
     จูนเสร็จแล้วค่อยเทรนอีกรอบด้วยค่าเดิม + extra=FEAT["check"] และ model="lgbm_ttl"

รัน (จาก root ของ repo):  python progress4/models/lightgbm_model.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import metrics.py ได้
from metrics import CLASSES, FEAT, REP, SEED, evaluate, load_split


def encode_categoricals(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """แปลงคอลัมน์หมวดหมู่เป็น pd.Categorical โดยยึด Categories จากชุด Train

    ช่วยให้:
    1. Integer codes ของหมวดหมู่ใน Train, Val, Test ตรงกันเสมอ
    2. ค่าที่ไม่เคยเจอใน Train จะกลายเป็น NaN ซึ่ง LightGBM จัดการได้อัตโนมัติ
    """
    train_out, val_out, test_out = train_df.copy(), val_df.copy(), test_df.copy()
    cat_cols = [c for c in FEAT["categorical"] if c in train_df.columns]

    for col in cat_cols:
        categories = train_df[col].dropna().astype(str).unique().tolist()
        train_out[col] = pd.Categorical(train_out[col].astype(str), categories=categories)
        val_out[col] = pd.Categorical(val_out[col].astype(str), categories=categories)
        test_out[col] = pd.Categorical(test_out[col].astype(str), categories=categories)

    return train_out, val_out, test_out


def engineer_geo_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ปรับปรุงฟีเจอร์ Geo:
    1. สร้าง Binary flags: ตรวจสอบ Private IP และการรับส่งข้ามประเทศ
    2. ยุบกลุ่ม ASN หางยาว: เก็บ Top 50 ASNs จาก Train นอกนั้นแปลงเป็น 'OTHER' เพื่อลด Noise
    """
    train_out, val_out, test_out = train_df.copy(), val_df.copy(), test_df.copy()

    # 1. สร้าง Flags
    for df in [train_out, val_out, test_out]:
        df["is_private_src"] = (df["src_country"].astype(str) == "PRIVATE").astype(int)
        df["is_private_dst"] = (df["dst_country"].astype(str) == "PRIVATE").astype(int)
        df["is_same_country"] = (df["src_country"].astype(str) == df["dst_country"].astype(str)).astype(int)

    # 2. ยุบ ASN หางยาว
    top_asns = set(train_df["src_asn"].dropna().astype(str).value_counts().head(50).index) | set(
        train_df["dst_asn"].dropna().astype(str).value_counts().head(50).index
    )

    for df in [train_out, val_out, test_out]:
        df["src_asn"] = df["src_asn"].astype(str).apply(lambda x: x if x in top_asns else "OTHER")
        df["dst_asn"] = df["dst_asn"].astype(str).apply(lambda x: x if x in top_asns else "OTHER")

    return train_out, val_out, test_out


def plot_feature_importance(model: lgb.LGBMClassifier, feature_names: list[str], out_path: Path) -> None:
    """สร้างกราฟ Top 20 Feature Importance"""
    importances = model.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="viridis")
    plt.title("LightGBM — Top 20 Feature Importance (Gain / Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved figure -> {out_path}")


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, out_path: Path) -> None:
    """สร้างกราฟ Confusion Matrix (Normalized ตาม True Class)"""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
    )
    plt.title("LightGBM — Confusion Matrix on Test Set (Normalized)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved figure -> {out_path}")


def main() -> None:
    fig_dir = REP / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: โหลดข้อมูล Base Features (ตัด TTL ออกแล้ว)")
    print("=" * 60)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test)

    # ---------------------------------------------------------
    # Step 2: จูนโมเดลบนชุด Val (Hyperparameter Tuning)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: เริ่มการจูน Hyperparameters บนชุด Val")
    print("=" * 60)

    # กำหนดชุดพารามิเตอร์ที่ต้องการทดลอง (10 รูปแบบครอบคลุมมิติต่างๆ)
    param_grid = [
        # 1. Baseline: ไม่ถ่วงน้ำหนักคลาส (Unweighted Control)
        {"n_estimators": 100, "learning_rate": 0.1, "class_weight": None},
        # 2. Baseline Balanced: ถ่วงน้ำหนักคลาส (Balanced Baseline)
        {"n_estimators": 100, "learning_rate": 0.1, "class_weight": "balanced"},
        # 3. Tuning Step: ลด learning rate 0.05 + ต้นไม้ 150
        {"n_estimators": 150, "learning_rate": 0.05, "class_weight": "balanced"},
        # 4. Deeper Trees: num_leaves=63
        {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63, "class_weight": "balanced"},
        # 5. Feature Subsampling 80%: colsample_bytree=0.8 + min_child=50 (คะแนนสูงสุดรอบก่อน)
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "class_weight": "balanced",
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
        },
        # 6. Feature Subsampling 75%: ปรับจุดกึ่งกลางของ Feature Sampling
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "class_weight": "balanced",
            "colsample_bytree": 0.75,
            "min_child_samples": 50,
        },
        # 7. Slower Learning + More Trees: lr=0.03, n_estimators=250 (เรียนรู้อย่างละเอียดขึ้น)
        {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "class_weight": "balanced",
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
        },
        # 8. Regularization Focus: เพิ่ม L1 (reg_alpha=0.1) และ L2 (reg_lambda=1.0) ลด Overfit
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "class_weight": "balanced",
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 50,
        },
        # 9. Controlled Depth: max_depth=8 + num_leaves=45 (คุมไม่ให้กิ่งลึกเกินไป)
        {
            "n_estimators": 200,
            "learning_rate": 0.04,
            "max_depth": 8,
            "num_leaves": 45,
            "class_weight": "balanced",
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
        },
        # 10. High-Capacity Model: num_leaves=95 + max_depth=10 + min_child=70
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 95,
            "class_weight": "balanced",
            "colsample_bytree": 0.8,
            "min_child_samples": 70,
        },
    ]

    best_macro_f1 = -1.0
    best_params: dict = param_grid[0]
    best_model: lgb.LGBMClassifier | None = None

    for i, params in enumerate(param_grid, 1):
        print(f"\n[Trial {i}/{len(param_grid)}] Parameters: {params}")
        clf = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)

        # วัดผลบนชุด val
        y_val_pred = clf.predict(X_val)
        y_val_score = clf.predict_proba(X_val)
        res = evaluate(
            y_true=y_val,
            y_pred=y_val_pred,
            y_score=y_val_score,
            score_classes=clf.classes_,
            model="lgbm",
            split="val",
            params=params,
        )

        if res["macro_f1"] > best_macro_f1:
            best_macro_f1 = res["macro_f1"]
            best_params = params
            best_model = clf

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    print("\n" + "*" * 60)
    print(f"Best Val Macro F1: {best_macro_f1:.4f}")
    print(f"Best Parameters  : {best_params}")
    print("*" * 60)

    # ---------------------------------------------------------
    # Step 3: วัดผลบนชุด Test ครั้งเดียวด้วย Best Model
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: วัดผลกับชุด Test ครั้งเดียว (Final Evaluation)")
    print("=" * 60)
    y_test_pred = best_model.predict(X_test)
    y_test_score = best_model.predict_proba(X_test)
    evaluate(
        y_true=y_test,
        y_pred=y_test_pred,
        y_score=y_test_score,
        score_classes=best_model.classes_,
        model="lgbm",
        split="test",
        params=best_params,
    )

    # สร้างรูปกราฟ Base Model
    plot_feature_importance(best_model, X_train.columns.tolist(), fig_dir / "lgbm_importance.png")
    plot_confusion_matrix(y_test, y_test_pred, fig_dir / "lgbm_confusion_matrix.png")

    # ---------------------------------------------------------
    # Step 4: ทดลองเปรียบเทียบ Geo Features (Base + Geo)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: ทดลองรอบ Geo Features (Base + Geo: country & ASN)")
    print("=" * 60)
    X_train_geo, _ = load_split("train", extra=FEAT["geo"])
    X_val_geo, _ = load_split("val", extra=FEAT["geo"])
    X_test_geo, _ = load_split("test", extra=FEAT["geo"])
    # สร้าง Flags และยุบกลุ่ม Top-50 ASNs เพื่อลด Noise
    X_train_geo, X_val_geo, X_test_geo = engineer_geo_features(X_train_geo, X_val_geo, X_test_geo)
    X_train_geo, X_val_geo, X_test_geo = encode_categoricals(X_train_geo, X_val_geo, X_test_geo)

    clf_geo = lgb.LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
    clf_geo.fit(X_train_geo, y_train)

    # วัดผล Geo บน Val
    y_val_geo_pred = clf_geo.predict(X_val_geo)
    y_val_geo_score = clf_geo.predict_proba(X_val_geo)
    evaluate(
        y_true=y_val,
        y_pred=y_val_geo_pred,
        y_score=y_val_geo_score,
        score_classes=clf_geo.classes_,
        model="lgbm_geo",
        split="val",
        params=best_params,
    )

    # วัดผล Geo บน Test
    y_test_geo_pred = clf_geo.predict(X_test_geo)
    y_test_geo_score = clf_geo.predict_proba(X_test_geo)
    evaluate(
        y_true=y_test,
        y_pred=y_test_geo_pred,
        y_score=y_test_geo_score,
        score_classes=clf_geo.classes_,
        model="lgbm_geo",
        split="test",
        params=best_params,
    )

    # สร้างรูปกราฟของ Geo Model
    plot_feature_importance(clf_geo, X_train_geo.columns.tolist(), fig_dir / "lgbm_geo_importance.png")
    plot_confusion_matrix(y_test, y_test_geo_pred, fig_dir / "lgbm_geo_confusion_matrix.png")

    # ---------------------------------------------------------
    # Step 5: ทดลองเปรียบเทียบ TTL Leakage (Base + TTL)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: ทดลองรอบ TTL Leakage (Base + TTL features)")
    print("=" * 60)
    X_train_ttl, _ = load_split("train", extra=FEAT["check"])
    X_val_ttl, _ = load_split("val", extra=FEAT["check"])
    X_test_ttl, _ = load_split("test", extra=FEAT["check"])
    X_train_ttl, X_val_ttl, X_test_ttl = encode_categoricals(X_train_ttl, X_val_ttl, X_test_ttl)

    clf_ttl = lgb.LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
    clf_ttl.fit(X_train_ttl, y_train)

    # วัดผล TTL บน Val
    y_val_ttl_pred = clf_ttl.predict(X_val_ttl)
    y_val_ttl_score = clf_ttl.predict_proba(X_val_ttl)
    evaluate(
        y_true=y_val,
        y_pred=y_val_ttl_pred,
        y_score=y_val_ttl_score,
        score_classes=clf_ttl.classes_,
        model="lgbm_ttl",
        split="val",
        params=best_params,
    )

    # วัดผล TTL บน Test
    y_test_ttl_pred = clf_ttl.predict(X_test_ttl)
    y_test_ttl_score = clf_ttl.predict_proba(X_test_ttl)
    evaluate(
        y_true=y_test,
        y_pred=y_test_ttl_pred,
        y_score=y_test_ttl_score,
        score_classes=clf_ttl.classes_,
        model="lgbm_ttl",
        split="test",
        params=best_params,
    )

    print("\nเสร็จสิ้นการทดลองทั้งหมด ผลลัพธ์ถูกบันทึกลง reports/ เรียบร้อยแล้ว")


if __name__ == "__main__":
    main()
