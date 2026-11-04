"""EDA 2: TTL Leakage Investigation
วิเคราะห์การรั่วไหลของข้อมูลฟีเจอร์ sttl และ dttl แยกตาม attack_cat
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "Art"


def run_ttl_leakage(df: pd.DataFrame = None, fig_dir: Path = None):
    """วิเคราะห์การแจกแจงของ sttl และ dttl ด้วย boxplot"""
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=["attack_cat", "sttl", "dttl"])
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 2] Running TTL Leakage Analysis...")
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    order = df["attack_cat"].value_counts().index
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    sns.boxplot(data=sample_df, x="attack_cat", y="sttl", order=order, ax=axes[0], palette="Set2")
    axes[0].set_title("Source TTL (sttl) by Attack Category", fontweight="bold")
    axes[0].set_xlabel("Attack Category", fontweight="bold")
    axes[0].set_ylabel("Source TTL (sttl)", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(y=64, color="r", linestyle="--", alpha=0.7, label="Default Linux TTL (64)")
    axes[0].axhline(y=254, color="purple", linestyle="--", alpha=0.7, label="Synthetic / Testbed TTL (254)")
    axes[0].legend(loc="upper right", fontsize=8)
    
    sns.boxplot(data=sample_df, x="attack_cat", y="dttl", order=order, ax=axes[1], palette="Set2")
    axes[1].set_title("Destination TTL (dttl) by Attack Category", fontweight="bold")
    axes[1].set_xlabel("Attack Category", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    
    fig.suptitle("2. TTL Feature Leakage: Normal (Median=31) vs Attacks (Median=254)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = out_dir / "02_ttl_leakage_distribution.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_ttl_leakage()

