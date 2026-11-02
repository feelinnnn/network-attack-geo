"""Progress 4 — โหลดข้อมูล + วัดผล (ทุกคนใช้ไฟล์นี้ตัวเดียวกัน ห้ามแก้)

load_split("train" / "val" / "test") -> X (base 38 คอลัมน์), y (attack_cat)
evaluate(...)        -> Macro F1, PR-AUC, Recall  เขียนลง reports/results_<model>.csv
evaluate_binary(...) -> แบบเดียวกันสำหรับ Isolation Forest (ทายได้แค่ ปกติ / โจมตี)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo

CLASSES = ["Normal", "Exploits", "Generic", "Fuzzers", "Reconnaissance",
           "DoS", "Analysis", "Backdoor", "Shellcode", "Worms"]  # 10 คลาส

CFG, FEAT = load_config(), load_config("features.yaml")
PROC, REP = p(CFG["paths"]["processed"]), p(CFG["paths"]["reports"])
SEED = CFG["seed"]  # 42


# อ่าน 1 ชุด -> (X, y)  extra = คอลัมน์ที่เพิ่ม เช่น FEAT["check"] ตอนทดลอง TTL leak
def load_split(name: str, extra: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    path = PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"not found: {path} (run python progress4/split.py first)")
    cols = FEAT["base"] + (extra or [])
    df = pd.read_parquet(path, columns=cols + [FEAT["target"]])
    return df[cols], df[FEAT["target"]].astype(str)


# เขียน 1 แถวลง results_<model>.csv  รันซ้ำ (model, split) เดิม -> ทับแถวเดิม
def save(row: dict) -> None:
    out = REP / f"results_{row['model']}.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[old["split"] != row["split"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).round(4).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [{row['model']} | {row['split']}] Macro F1 {row['macro_f1']:.3f}  PR-AUC {row['pr_auc']:.3f}  Recall {row['recall']:.3f}")


# y_score = model.predict_proba(X), score_classes = model.classes_
def evaluate(y_true, y_pred, y_score, score_classes, model: str, split: str) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    score = np.asarray(y_score)[:, [list(score_classes).index(c) for c in CLASSES]]  # เรียงคอลัมน์ตาม CLASSES
    row = {"model": model, "split": split,
           "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
           "pr_auc": np.mean([average_precision_score(y_true == c, score[:, k]) for k, c in enumerate(CLASSES)]),
           "recall": recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)}
    save(row)
    return row


# is_attack = True/False ต่อแถว, score = ยิ่งสูงยิ่งน่าจะโจมตี
def evaluate_binary(y_true, is_attack, score, model: str, split: str) -> dict:
    true_att = np.asarray(y_true).astype(str) != "Normal"
    row = {"model": model, "split": split,
           "macro_f1": f1_score(true_att, is_attack, average="macro", zero_division=0),
           "pr_auc": average_precision_score(true_att, score),
           "recall": recall_score(true_att, is_attack, average="macro", zero_division=0)}
    save(row)
    return row
