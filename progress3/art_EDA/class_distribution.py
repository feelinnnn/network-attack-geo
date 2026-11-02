"""EDA 1: Target Class Imbalance & Distribution
วิเคราะห์การกระจายตัวของคลาสเป้าหมาย attack_cat (10 คลาส)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "Art"


def run_class_distribution(df: pd.DataFrame = None, fig_dir: Path = None):
    """วิเคราะห์การกระจายตัวของ attack_cat และสร้างกราฟ log-scale bar chart"""
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=["attack_cat"])
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 1] Running Target Class Distribution Analysis...")
    counts = df["attack_cat"].value_counts()
    pcts = 100 * counts / len(df)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(
        counts.index,
        counts.values,
        color=sns.color_palette("deep", len(counts)),
        edgecolor="black",
        alpha=0.85
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Number of Flows (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Attack Category (attack_cat)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Target Class Imbalance (Total: 2,059,414 flows)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right")
    
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        yval = bar.get_height()
        label = f"{count:,}\n({pct:.2f}%)" if pct >= 0.05 else f"{count:,}\n({pct:.3f}%)"
        ax1.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, label, ha="center", va="bottom", fontsize=8)
    
    ax1.set_ylim(bottom=10, top=counts.max() * 6)
    plt.tight_layout()
    
    save_path = out_dir / "01_class_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")
    return counts, pcts


if __name__ == "__main__":
    run_class_distribution()

