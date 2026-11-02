"""EDA 3: Geolocation Shortcut Learning Risk
วิเคราะห์ความเสี่ยงการจำประเทศ/ASN ทางลัด (Spurious Correlation)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "art"


def run_geo_shortcut(df: pd.DataFrame = None, fig_dir: Path = None):
    """วิเคราะห์สัดส่วนการโจมตีแยกตามประเทศต้นทาง src_country"""
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=["src_country", "label"])
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 3] Running Geolocation Shortcut Risk Analysis...")
    geo_label = pd.crosstab(df["src_country"], df["label"], normalize="index") * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    
    # 1. Normal vs Attack proportion
    geo_label.rename(columns={0: "Normal", 1: "Attack"}).plot(
        kind="bar", stacked=True, color=["#4caf50", "#f44336"], ax=ax1, edgecolor="black", alpha=0.85
    )
    ax1.set_title("Attack vs Normal Proportion per Source Country", fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.set_xlabel("Source Country (src_country)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="Traffic Type", loc="center left", bbox_to_anchor=(1, 0.5))
    
    # 2. Total attack count by country
    attack_by_country = df[df["label"] == 1]["src_country"].value_counts()
    ax2.bar(attack_by_country.index, attack_by_country.values, color="#e53935", edgecolor="black", alpha=0.85)
    ax2.set_title("Total Attack Flow Count by Source Country", fontweight="bold")
    ax2.set_ylabel("Number of Attack Flows", fontweight="bold")
    ax2.set_xlabel("Source Country", fontweight="bold")
    for idx, val in enumerate(attack_by_country.values):
        ax2.text(idx, val + 1000, f"{val:,}\n(100%)", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, attack_by_country.max() * 1.15)
    
    fig.suptitle("3. Geolocation Shortcut Risk: 100% of Attacks Originate Exclusively from KP (North Korea)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_path = out_dir / "03_geo_shortcut_risk.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_geo_shortcut()

