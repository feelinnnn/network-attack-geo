"""EDA 6: Protocol & Service Attack Affinity
วิเคราะห์ความสัมพันธ์ระหว่างประเภทการโจมตีกับเครือข่ายบริการ (service & proto)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "art"


def run_service_protocol(df: pd.DataFrame = None, fig_dir: Path = None):
    """วิเคราะห์ความสัมพันธ์ของ service กับ attack_cat ด้วย heatmap"""
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=["service", "attack_cat"])
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 6] Running Protocol & Service Attack Affinity Analysis...")
    top_services = df["service"].value_counts().head(7).index
    service_sub = df[df["service"].isin(top_services)]
    ct = pd.crosstab(service_sub["attack_cat"], service_sub["service"], normalize="index") * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        ct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "% Within Attack Category"},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title("6. Attack Category Affinity with Network Services (% Row Normalization)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Network Service (service)", fontweight="bold")
    ax.set_ylabel("Attack Category (attack_cat)", fontweight="bold")
    plt.tight_layout()
    
    save_path = out_dir / "06_service_proto_heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_service_protocol()

