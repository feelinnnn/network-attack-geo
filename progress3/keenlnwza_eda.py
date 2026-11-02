"""Progress 3 -- EDA (Exploratory Data Analysis)

Input : data/processed/features.parquet  (ผลจาก Progress 2)
Output: reports/figures/<ชื่อ>_*.png    (กราฟ ขึ้นต้นด้วยชื่อตัวเองจะได้ไม่ซ้ำ)
        reports/eda_<ชื่อ>.md           (สรุปข้อค้นพบของตัวเอง)

ทำอะไรก็ได้ ขั้นต่ำคนละ 5 อย่าง
แต่ละอย่างต้องตอบได้ว่า "เห็นอะไร -> เลยตัดสินใจ / เข้าใจอะไร"

รัน (จาก root ของ repo):  python progress3/eda.py

7 แนวทาง:
  1. Class Imbalance & Rare Classes
  2. Geo-distribution & Shortcut Risk
  3. TTL Leakage Verification
  4. Heavy-tailed Distributions & Log Transforms
  5. Protocol & Service Attack Profiling
  6. Multicollinearity & Correlation Analysis
  7. Label Conflict Deep-Dive
"""
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

# ป้องกัน [WinError 5] Access is denied ของ Matplotlib บน Windows
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")  # บังคับใช้ backend แบบไม่เปิด GUI (เซฟเป็นไฟล์อย่างเดียว)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, p

# ----------------------------------------------------------------------
# ตั้งค่าพาธและตัวแปรหลัก (ประกาศระดับโมดูล ไม่ให้ Linter เตือน)
# ----------------------------------------------------------------------
NAME = "keen"                 # ชื่อ -> ขึ้นต้นไฟล์กราฟ
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

ATTACK_ORDER = ["Normal", "Generic", "Exploits", "Fuzzers", "Reconnaissance",
                "DoS", "Analysis", "Backdoor", "Shellcode", "Worms"]

findings: list[dict] = []     # เก็บข้อค้นพบ -> สร้าง eda_keen.md ท้ายสุด


def save(fig: plt.Figure, tag: str) -> Path:
    """บันทึกกราฟลง reports/figures/keen_<tag>.png"""
    out = FIG_DIR / f"{NAME}_{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")
    return out


def note(title: str, see: str, decide: str) -> None:
    """บันทึกข้อค้นพบ (เห็นอะไร -> ตัดสินใจอะไร)"""
    findings.append({"title": title, "see": see, "decide": decide})


# ======================================================================
# EDA 1: Class Imbalance & Rare Classes
# ======================================================================
def eda1_class_imbalance(df: pd.DataFrame) -> None:
    print("\n[1/7] Class Imbalance & Rare Classes")

    counts = df["attack_cat"].value_counts().reindex(ATTACK_ORDER)
    total = len(df)

    # --- กราฟ: bar chart (log scale) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71" if c == "Normal" else "#e74c3c" for c in counts.index]
    bars = ax1.barh(counts.index[::-1], counts.values[::-1],
                    color=colors[::-1], edgecolor="white")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of rows (log scale)")
    ax1.set_title("1a. Class Distribution (log scale)")
    for bar, val in zip(bars, counts.values[::-1]):
        pct = val / total * 100
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}  ({pct:.2f}%)", va="center", fontsize=8)

    # --- กราฟ: pie chart เฉพาะ Attack (ไม่รวม Normal) ---
    atk = counts.drop("Normal")
    ax2.pie(atk.values, labels=atk.index, autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 8})
    ax2.set_title("1b. Attack-only breakdown (excl. Normal)")

    fig.suptitle("EDA 1 -- Class Imbalance & Rare Classes", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "1_class_imbalance")

    # --- สรุป ---
    rarest = counts.idxmin()
    note("Class Imbalance & Rare Classes",
         f"Normal มี {counts['Normal']:,} แถว ({counts['Normal']/total*100:.1f}%) "
         f"คลาสน้อยสุดคือ {rarest} มีแค่ {counts[rarest]:,} แถว ({counts[rarest]/total*100:.3f}%)",
         "ต้องใช้ Stratified Split เสมอ และวัดผลด้วย Macro F1-Score ไม่ใช่ Accuracy "
         "เพราะโมเดลที่ทายว่า Normal ทุกแถวจะได้ Accuracy 95% แต่ F1 ของคลาสโจมตี = 0")


# ======================================================================
# EDA 2: Geo-distribution & Shortcut Risk
# ======================================================================
def eda2_geo_shortcut(df: pd.DataFrame) -> None:
    print("\n[2/7] Geo-distribution & Shortcut Risk")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- กราฟ 2a: src_country vs label ---
    ct_src = pd.crosstab(df["src_country"], df["label"])
    ct_src.columns = ["Normal", "Attack"]
    ct_src = ct_src.sort_values("Attack", ascending=True)
    ct_src.plot.barh(stacked=True, ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("2a. src_country vs label")
    axes[0].set_xlabel("Number of rows")

    # --- กราฟ 2b: dst_country vs label ---
    ct_dst = pd.crosstab(df["dst_country"], df["label"])
    ct_dst.columns = ["Normal", "Attack"]
    ct_dst = ct_dst.sort_values("Attack", ascending=True)
    ct_dst.plot.barh(stacked=True, ax=axes[1], color=["#2ecc71", "#e74c3c"])
    axes[1].set_title("2b. dst_country vs label")
    axes[1].set_xlabel("Number of rows")

    fig.suptitle("EDA 2 -- Geo-distribution & Shortcut Risk", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "2_geo_shortcut")

    # --- ตาราง src_asn vs attack_cat ---
    ct_asn = pd.crosstab(df["src_asn"], df["attack_cat"])
    print("  src_asn vs attack_cat:\n", ct_asn.to_string())

    # --- สรุป ---
    total_attack = int((df["label"] == 1).sum())
    note("Geo Shortcut Risk",
         f"การโจมตีทั้ง {total_attack:,} แถว มาจาก KP (ASN 131279) ทั้งหมด 100% "
         f"ส่วน JP และ AU เป็น Normal ล้วน -> Geo = Perfect Predictor ในชุดนี้",
         "ต้องสร้าง Baseline Model แบบ Base-only (ไม่มี Geo) เทียบกับ Base+Geo "
         "ถ้า Geo ดีขึ้นมาก แปลว่าโมเดลอาศัย Shortcut -> ใช้จริงไม่ได้ "
         "(โลกจริงการโจมตีมาจากทุกประเทศ ไม่ใช่แค่ KP)")


# ======================================================================
# EDA 3: TTL Leakage Verification
# ======================================================================
def eda3_ttl_leakage(df: pd.DataFrame) -> None:
    print("\n[3/7] TTL Leakage Verification")

    ttl_cols = ["sttl", "dttl", "ct_state_ttl"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col in zip(axes, ttl_cols):
        # สุ่ม sample เพราะข้อมูล 2 ล้านแถวจะช้า
        sample = df[[col, "label"]].sample(n=min(50_000, len(df)), random_state=42)
        sample["label_str"] = sample["label"].map({0: "Normal (0)", 1: "Attack (1)"})
        sns.boxplot(data=sample, x="label_str", y=col, hue="label_str", ax=ax,
                    palette={"Normal (0)": "#2ecc71", "Attack (1)": "#e74c3c"},
                    fliersize=1, legend=False)
        ax.set_xlabel("label")
        ax.set_title(col)

    fig.suptitle("EDA 3 -- TTL Leakage Verification (label 0 vs 1)", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "3_ttl_leakage")

    # --- คำนวณ attack-rate เมื่อแบ่งตาม sttl ---
    bins = pd.cut(df["sttl"], bins=[0, 30, 60, 128, 255], include_lowest=True)
    ttl_rate = df.groupby(bins, observed=False)["label"].agg(["mean", "count"])
    ttl_rate.columns = ["attack_rate", "n_rows"]
    print("  sttl bins -> attack rate:")
    print(ttl_rate.to_string())

    # --- สรุป ---
    note("TTL Leakage",
         "sttl/dttl แยก Normal กับ Attack ได้ชัดเจนผิดปกติ "
         "เพราะ UNSW-NB15 ใช้เครื่องยิงคนละ OS (TTL ต่างกัน) "
         "ค่า ct_state_ttl ที่สูง >= 2 มักเป็น Attack",
         "ฟีเจอร์ TTL เป็น Data Leakage จาก Lab Setup -> ต้องทดลองเทรนทั้งมีและไม่มี TTL "
         "สอดคล้องกับ features.yaml ที่แยก check: [sttl, dttl, ct_state_ttl] ไว้")


# ======================================================================
# EDA 4: Heavy-tailed Distributions & Log Transforms
# ======================================================================
def eda4_heavy_tailed(df: pd.DataFrame) -> None:
    print("\n[4/7] Heavy-tailed Distributions & Log Transforms")

    cols = ["dur", "sbytes", "dbytes", "sload", "dload", "spkts"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        vals = df[col].dropna()
        vals_log = np.log10(vals + 1)

        ax.hist(vals_log, bins=80, color="#3498db", edgecolor="white",
                alpha=0.8, density=True)
        ax.set_title(f"{col}\nraw: skew={vals.skew():.1f} -> log10: skew={vals_log.skew():.1f}",
                     fontsize=9)
        ax.set_xlabel(f"log10({col} + 1)")
        ax.set_ylabel("Density")

    fig.suptitle("EDA 4 -- Heavy-tailed Distributions (after log10 transform)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "4_heavy_tailed")

    # --- สรุปตาราง skewness ---
    skew_raw = df[cols].skew()
    skew_log = np.log10(df[cols] + 1).skew()
    skew_df = pd.DataFrame({"raw_skew": skew_raw, "log10_skew": skew_log})
    print("  Skewness comparison:")
    print(skew_df.to_string())

    note("Heavy-tailed Distributions",
         f"ฟีเจอร์ volume/rate (dur, sbytes, sload ฯลฯ) มี skewness สูงมาก "
         f"(เช่น sbytes skew={skew_raw['sbytes']:.0f}) ค่าส่วนใหญ่เล็กมาก "
         f"แต่มี outlier สูงหลักล้าน",
         "ใช้ log10(x+1) transform ก่อนเข้าโมเดล Linear/NN เพื่อลด skewness "
         "หรือเลือกใช้ Tree-based model (XGBoost/LightGBM) ที่ไม่ไวต่อ scale")


# ======================================================================
# EDA 5: Protocol & Service Attack Profiling
# ======================================================================
def eda5_proto_service(df: pd.DataFrame) -> None:
    print("\n[5/7] Protocol & Service Attack Profiling")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 5a: service vs attack_cat heatmap ---
    ct_svc = pd.crosstab(df["attack_cat"], df["service"])
    ct_svc = ct_svc.reindex(ATTACK_ORDER)
    # แปลงเป็น row-normalized (%) เพื่อให้เทียบสัดส่วนได้
    ct_svc_pct = ct_svc.div(ct_svc.sum(axis=1), axis=0) * 100
    sns.heatmap(ct_svc_pct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax1,
                linewidths=0.5, cbar_kws={"label": "% within class"})
    ax1.set_title("5a. Service distribution per attack class (%)")
    ax1.set_ylabel("")

    # --- 5b: proto (top 5 + other) vs label ---
    top_proto = df["proto"].value_counts().head(5).index.tolist()
    proto_group = df["proto"].where(df["proto"].isin(top_proto), "other")
    ct_proto = pd.crosstab(proto_group, df["label"])
    ct_proto.columns = ["Normal", "Attack"]
    ct_proto_pct = ct_proto.div(ct_proto.sum(axis=1), axis=0) * 100
    ct_proto_pct.plot.barh(stacked=True, ax=ax2, color=["#2ecc71", "#e74c3c"])
    ax2.set_title("5b. Protocol groups -> Attack rate (%)")
    ax2.set_xlabel("% of rows")
    ax2.legend(loc="lower right")

    fig.suptitle("EDA 5 -- Protocol & Service Attack Profiling", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "5_proto_service")

    # --- สรุป ---
    n_proto = df["proto"].nunique()
    pct_tcp_udp = (df["proto"].isin(["tcp", "udp"]).sum() / len(df)) * 100
    note("Protocol & Service Profiling",
         f"proto มี {n_proto} ค่าย่อย แต่ TCP+UDP คิดเป็น {pct_tcp_udp:.1f}% "
         f"ของข้อมูลทั้งหมด -- service '-' (ไม่ระบุ) คิดเป็นกว่าครึ่ง "
         f"การโจมตีแต่ละประเภทมี service profile ต่างกัน เช่น Fuzzers/DoS มักเกิดบน '-' (no service)",
         "จัดกลุ่ม proto ที่พบน้อยรวมเป็น 'other' เพื่อลดมิติ One-Hot Encoding "
         "ฟีเจอร์ service มีประโยชน์ในการแยก attack type ควรเก็บไว้")


# ======================================================================
# EDA 6: Multicollinearity & Correlation Analysis
# ======================================================================
def eda6_correlation(df: pd.DataFrame, feat: dict) -> None:
    print("\n[6/7] Multicollinearity & Correlation Analysis")
    print("  calculating Spearman correlation (this takes ~1-2 min)...")

    num_cols = [c for c in feat["base"] if c not in feat.get("categorical", [])
                and c in df.columns]
    corr = df[num_cols].corr(method="spearman")

    # --- กราฟ: correlation heatmap ---
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", ax=ax, linewidths=0.3,
                annot_kws={"fontsize": 6}, square=True)
    ax.set_title("EDA 6 -- Spearman Correlation Matrix (numeric base features)")
    fig.tight_layout()
    save(fig, "6_correlation")

    # --- หาคู่ที่ correlation สูง ---
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.85:
                high_corr.append((corr.index[i], corr.columns[j], round(r, 3)))
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    if high_corr:
        print("  High correlation pairs (|r| >= 0.85):")
        for a, b, r in high_corr:
            print(f"    {a:25s} <-> {b:25s}  r = {r:+.3f}")

    note("Multicollinearity",
         f"พบ {len(high_corr)} คู่ฟีเจอร์ที่มี |Spearman r| >= 0.85 "
         f"เช่น " + (", ".join(f"{a}<->{b} ({r:+.3f})" for a, b, r in high_corr[:5])
                     if high_corr else "ไม่มี"),
         "คู่ที่ซ้ำซ้อนมาก (เช่น ct_srv_src <-> ct_srv_dst) สามารถตัดตัวหนึ่งออกได้ "
         "เพื่อลดมิติ หรือปล่อยให้ Tree-based model จัดการเอง แต่ต้องระวังตอน interpret")


# ======================================================================
# EDA 7: Label Conflict Deep-Dive
# ======================================================================
def eda7_label_conflicts(df: pd.DataFrame, feat: dict) -> None:
    print("\n[7/7] Label Conflict Deep-Dive")

    X_cols = feat["base"] + feat["check"]
    key = pd.util.hash_pandas_object(df[X_cols], index=False)
    n_cls = df["attack_cat"].groupby(key.values).transform("nunique")
    conflict_mask = n_cls.values > 1

    conf_rate = (df.groupby("attack_cat")
                 .apply(lambda g: pd.Series({
                     "total": len(g),
                     "conflict": int(conflict_mask[g.index].sum()),
                 }), include_groups=False))
    conf_rate["pct"] = (conf_rate["conflict"] / conf_rate["total"] * 100).round(1)
    conf_rate = conf_rate.sort_values("pct", ascending=False)
    print("  Label conflict rates:")
    print(conf_rate.to_string())

    # --- กราฟ: bar chart conflict % ---
    fig, ax = plt.subplots(figsize=(10, 5))
    order = conf_rate.index.tolist()
    colors = ["#e74c3c" if p > 50 else "#f39c12" if p > 10 else "#2ecc71"
              for p in conf_rate["pct"]]
    bars = ax.barh(order[::-1], conf_rate["pct"].values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Conflict rate (%)")
    ax.set_title("EDA 7 -- Label Conflicts: identical features, different attack_cat")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    for bar, val in zip(bars, conf_rate["pct"].values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "7_label_conflicts")

    # --- สรุป ---
    worst = conf_rate.index[0]
    note("Label Conflicts",
         f"คลาส {worst} มี conflict rate สูงสุดที่ {conf_rate.loc[worst, 'pct']:.1f}% "
         f"หมายถึง {conf_rate.loc[worst, 'conflict']:,} แถวมีฟีเจอร์เหมือนกันทุกตัว "
         f"แต่ถูกติดป้ายคนละ attack_cat",
         "โมเดลไม่มีทางทำนาย Backdoor/Analysis ได้ถูก 100% เพราะมี Noise/Ambiguity "
         "ในตัวข้อมูล ควรรายงานขีดจำกัดนี้ใน report และใช้เป็น upper bound สำหรับ evaluation")


# ======================================================================
# สร้าง eda_keen.md
# ======================================================================
def write_report(out_path: Path) -> None:
    buf = StringIO()
    buf.write("# EDA Findings -- keen\n\n")
    buf.write("รันจาก `python progress3/eda.py` ทั้ง 7 แนวทาง\n\n")
    for i, f in enumerate(findings, 1):
        buf.write(f"## {i}. {f['title']}\n\n")
        buf.write(f"**เห็นอะไร:** {f['see']}\n\n")
        buf.write(f"**ตัดสินใจ/เข้าใจอะไร:** {f['decide']}\n\n")
        buf.write(f"![{f['title']}](figures/{NAME}_{i}_*.png)\n\n")
    buf.write("---\n*สร้างอัตโนมัติจาก progress3/eda.py*\n")
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  saved -> {out_path.relative_to(ROOT)}")


# ======================================================================
# main
# ======================================================================
def main() -> None:
    cfg = load_config()
    feat = load_config("features.yaml")
    proc = p(cfg["paths"]["processed"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("loading features.parquet")
    df = pd.read_parquet(proc / "features.parquet")
    df["attack_cat"] = df["attack_cat"].astype(str)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # 7 แนวทาง EDA
    eda1_class_imbalance(df)
    eda2_geo_shortcut(df)
    eda3_ttl_leakage(df)
    eda4_heavy_tailed(df)
    eda5_proto_service(df)
    eda6_correlation(df, feat)
    eda7_label_conflicts(df, feat)

    # สร้างรายงานสรุป
    print("\nwriting eda report")
    rep_dir = p(cfg["paths"]["reports"])
    write_report(rep_dir / f"eda_{NAME}.md")

    print(f"\nDONE -- {len(findings)} findings, "
          f"{len(list(FIG_DIR.glob(f'{NAME}_*.png')))} figures saved")


if __name__ == "__main__":
    main()
