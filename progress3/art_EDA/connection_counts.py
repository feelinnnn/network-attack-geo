"""EDA 7: Connection Count Features Analysis (ct_*)
วิเคราะห์พฤติกรรมความถี่การเชื่อมต่อ (Connection Frequency & Density) แยกตามประเภทการโจมตี
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "art"

CT_COLS = [
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]


def run_connection_counts(df: pd.DataFrame = None, fig_dir: Path = None):
    """วิเคราะห์และพล็อตฟีเจอร์กลุ่ม ct_* เพื่อตรวจจับพฤติกรรมการเชื่อมต่อซ้ำซ้อนและการโจมตีแบบ Flood/Stealth"""
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=["attack_cat"] + CT_COLS)
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 7] Running Connection Count Features (ct_*) Analysis...")
    
    # 1. คำนวณค่าเฉลี่ยของแต่ละ ct_* ฟีเจอร์แยกตาม attack_cat
    ct_mean = df.groupby("attack_cat")[CT_COLS].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    
    # พล็อตที่ 1: Heatmap ค่าเฉลี่ย Connection Counts ในแต่ละคลาส
    sns.heatmap(
        ct_mean,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Mean Connection Count"},
        ax=ax1,
        linewidths=0.5,
    )
    ax1.set_title("Mean Connection Counts (ct_*) per Attack Category", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Connection Count Features", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attack Category (attack_cat)", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=35)
    
    # พล็อตที่ 2: Boxplot เปรียบเทียบ ct_srv_src (จำนวนการเชื่อมต่อบริการเดียวกันจากต้นทางเดิม)
    order = ["Generic", "Normal", "Fuzzers", "Backdoor", "Analysis", "DoS", "Reconnaissance", "Exploits", "Shellcode", "Worms"]
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    
    sns.boxplot(
        data=sample_df,
        x="attack_cat",
        y="ct_srv_src",
        hue="attack_cat",
        order=order,
        palette="Spectral",
        ax=ax2,
        showfliers=False,  # ซ่อน outlier จุดไข่ปลา เพื่อให้เห็น IQR กล่องชัดเจน
        legend=False,
    )
    ax2.set_title("Distribution of ct_srv_src (Excluding Extreme Outliers)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Attack Category", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ct_srv_src Count", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", rotation=40)
    ax2.axhline(y=4.0, color="gray", linestyle="--", alpha=0.7, label="Normal Median (4.0)")
    ax2.legend(loc="upper right", fontsize=9)
    
    fig.suptitle("7. Connection Count Features: Flooding vs Stealth Single-Shot Attack Patterns", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = out_dir / "07_connection_counts.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_connection_counts()
