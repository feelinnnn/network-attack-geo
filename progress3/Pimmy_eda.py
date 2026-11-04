"""Reproducible exploratory analysis for the Progress 2 feature dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
FIGURES_DIR = ROOT / "reports" / "figures" / "Pimmy"
REPORT_PATH = ROOT / "reports" / "eda_Pimmy.md"
PREFIX = "Pimmy"
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(name: str) -> None:
    path = FIGURES_DIR / f"{PREFIX}_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# คลาสที่มีจำนวนแถวมากที่สุด = Normal, น้อยที่สุด = Worms (171 แถว) -> ต้อง stratified split
def class_balance(df: pd.DataFrame, findings: list[str]) -> None:
    counts = df["attack_cat"].value_counts().sort_values()
    normal_share = counts["Normal"] / len(df)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=counts.values, y=counts.index, color="#4472c4")
    ax.set_xscale("log")
    ax.set(xlabel="Rows (log scale)", ylabel="Attack category")
    ax.set_title("Class balance after cleaning")
    for value, label in zip(counts.values, counts.index):
        ax.text(value * 1.08, label, f"{value:,}", va="center", fontsize=8) # ไม่นับ Normal เพราะมันเยอะมาก
    save_figure("class_balance")
    findings.append(
        f"- **Class balance:** `Normal` is {normal_share:.1%} of {len(df):,} rows, "
        f"while `Worms` has only {counts['Worms']:,}. Use stratified splits and "
        "class-aware metrics (macro F1 and per-class recall); accuracy alone is misleading."
    )

# กราฟแสดงสัดส่วนการโจมตีของแต่ละประเทศต้นทาง (src_country) และประเภทการโจมตี (attack_cat) โดยใช้ heatmap เพื่อให้เห็นว่าประเทศใดมีอัตราการโจมตีสูงสุด และควรเก็บคุณสมบัติทางภูมิศาสตร์ไว้ในการวิเคราะห์ต่อไป
def geography(df: pd.DataFrame, findings: list[str]) -> None:
    table = pd.crosstab(df["src_country"], df["attack_cat"])
    rates = table.div(table.sum(axis=1), axis=0).drop(columns="Normal", errors="ignore")
    attack_rate = df.groupby("src_country", observed=True)["label"].mean()
    rates["Any attack"] = attack_rate
    rates = rates.sort_values("Any attack", ascending=False)
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        rates.drop(columns="Any attack"),
        cmap="YlOrRd",
        vmin=0,
        vmax=max(float(rates.drop(columns="Any attack").to_numpy().max()), 0.01),
        annot=True,
        fmt=".2%",
        cbar_kws={"label": "Share within source country"},
    )
    plt.title("Attack-category composition by source country")
    plt.xlabel("Attack category")
    plt.ylabel("Source country / address type")
    save_figure("geo_attack_rate")
    country = rates.index[0]
    findings.append(
        f"- **Geography:** `{country}` has the highest source-country attack rate "
        f"({rates.loc[country, 'Any attack']:.2%}), while traffic is dominated by "
        f"`{df['src_country'].value_counts().index[0]}`. Keep geo features, but "
        "evaluate by country/address type and avoid causal interpretations."
    )

def protocol(df: pd.DataFrame, findings: list[str]) -> None:
    top = df["proto"].value_counts().head(12).index
    summary = (
        df[df["proto"].isin(top)]
        .groupby("proto", observed=True)["label"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=summary.reset_index(),
        x="mean",
        y="proto",
        order=summary.index,
        color="#ed7d31",
    )
    ax.set(xlabel="Attack rate", ylabel="Protocol")
    ax.set_xlim(0, min(1, max(0.01, summary["mean"].max() * 1.15)))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Attack rate for the 12 most common protocols")
    for i, (rate, count) in enumerate(zip(summary["mean"], summary["count"])):
        ax.text(rate + 0.005, i, f"n={count:,}", va="center", fontsize=8)
    save_figure("protocol_attack_rate")
    findings.append(
        f"- **Protocol:** `{summary.index[0]}` has the highest attack rate among the "
        f"12 most common protocols ({summary.iloc[0]['mean']:.2%}; n={summary.iloc[0]['count']:,}). "
        "Encode `proto` categorically and retain support counts because rare levels are unstable."
    )

# ตัวเลขบางคอลัมน์มีค่าที่สูงมากเมื่อเทียบกับค่ากลาง (median) เช่น dur, sbytes, dbytes, sload -> ควรใช้ log1p หรือ robust scaler เพื่อไม่ให้ extreme flows ครอบงำการฝึกโมเดล
def numeric_scales(df: pd.DataFrame, findings: list[str]) -> None:
    columns = ["dur", "sbytes", "dbytes", "sload"]
    sample = df.sample(n=min(120_000, len(df)), random_state=RANDOM_STATE)
    long = sample[["label", *columns]].melt(
        id_vars="label", var_name="feature", value_name="value"
    )
    long["value"] = long["value"].clip(lower=0) + 1
    grid = sns.FacetGrid(
        long,
        col="feature",
        col_wrap=2,
        hue="label",
        sharex=False,
        sharey=False,
        height=3.2,
        aspect=1.35,
    )
    grid.map_dataframe(sns.histplot, x="value", bins=60, stat="density", common_norm=False)
    grid.set(xscale="log", xlabel="Value + 1 (log scale)", ylabel="Density")
    grid.add_legend(title="label")
    grid.fig.suptitle("Heavy-tailed numeric features (deterministic sample)", y=1.02)
    grid.savefig(FIGURES_DIR / f"{PREFIX}_numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    ratios = {column: df[column].max() / max(df[column].median(), 1e-12) for column in columns}
    skewed = max(ratios, key=ratios.get)
    findings.append(
        f"- **Numeric scale:** `{skewed}` has the largest max-to-median ratio "
        f"({ratios[skewed]:,.0f}x). Apply `log1p` or a robust scaler to non-negative "
        "volume/rate features so extreme flows do not dominate training."
    )

# ดาต้า TTL (Time To Live) มีความแตกต่างกันระหว่างแถวปกติและแถวโจมตี โดย median ของ sttl, dttl, และ ct_state_ttl ของแถวโจมตีมีค่าที่สูงกว่าแถวปกติ -> ควรตรวจสอบคุณสมบัติ TTL แยกจากการฝึกโมเดลเพื่อหลีกเลี่ยงการรั่วไหลของข้อมูล
def ttl_check(df: pd.DataFrame, findings: list[str]) -> None:
    summary = (
        df.groupby("attack_cat", observed=True)[["sttl", "dttl", "ct_state_ttl"]]
        .median()
        .sort_values("sttl")
    )
    scaled = summary.copy()
    for column in scaled:
        spread = scaled[column].max() - scaled[column].min()
        scaled[column] = (scaled[column] - scaled[column].min()) / (spread or 1)
    plt.figure(figsize=(10, 5))
    sns.heatmap(scaled, cmap="viridis", annot=summary, fmt=".1f", cbar=False)
    plt.title("Median TTL-related values by attack category")
    plt.xlabel("Color normalized per feature; annotations are medians")
    plt.ylabel("Attack category")
    save_figure("ttl_leakage_check")
    by_label = df.groupby("label", observed=True)["sttl"].median()
    findings.append(
        f"- **Potential leakage:** median `sttl` is {by_label.get(0, float('nan')):.1f} "
        f"for normal rows versus {by_label.get(1, float('nan')):.1f} for attacks. "
        "Audit `sttl`, `dttl`, and `ct_state_ttl` in a separate ablation before claiming generalization."
    )

# คอลัมน์ state มีค่าที่แตกต่างกันมากกว่า 12 ค่า แต่มีเพียง 12 ค่าที่พบบ่อยที่สุด -> ควร encode เป็น categorical และ regularize/group rare levels เพื่อลด overfitting
def connection_state(df: pd.DataFrame, findings: list[str]) -> None:
    counts = df["state"].value_counts()
    states = counts.head(12).index
    summary = (
        df[df["state"].isin(states)]
        .groupby("state", observed=True)["label"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=summary.reset_index(), x="mean", y="state", order=summary.index, color="#70ad47"
    )
    ax.set(xlabel="Attack rate", ylabel="Connection state")
    ax.set_xlim(0, min(1, max(0.01, summary["mean"].max() * 1.15)))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Attack rate for the 12 most common connection states")
    save_figure("state_attack_rate")
    rare = int((counts < 100).sum())
    findings.append(
        f"- **Connection state:** {len(counts)} states occur, including {rare} with fewer "
        "than 100 rows. Encode `state` categorically and regularize/group rare levels "
        "during validation to reduce overfitting."
    )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Run Progress 2 first; missing {FEATURES_PATH}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    findings: list[str] = []
    class_balance(df, findings)
    geography(df, findings)
    protocol(df, findings)
    numeric_scales(df, findings)
    ttl_check(df, findings)
    connection_state(df, findings)
    figure_names = [
        "class_balance",
        "geo_attack_rate",
        "protocol_attack_rate",
        "numeric_distributions",
        "ttl_leakage_check",
        "state_attack_rate",
    ]
    report = [
        "# EDA - Pimmy",
        "",
        f"Dataset: `data/processed/features.parquet` ({len(df):,} rows, {len(df.columns)} columns).",
        "",
        "## Findings and decisions",
        "",
        *findings,
        "",
        "## Generated figures",
        "",
        *[f"- `reports/figures/Pimmy/{PREFIX}_{name}.png`" for name in figure_names],
        "",
        "Plots use deterministic ordering or a fixed random seed where sampling is required.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved -> {REPORT_PATH}")
    print(f"saved figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
