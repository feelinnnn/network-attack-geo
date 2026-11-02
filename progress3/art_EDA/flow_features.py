"""EDA 5: Numerical Flow Features Distribution & Skewness
วิเคราะห์ความเบ้ (Skewness) ของตัวแปรสถิติการไหล และการแปลง np.log1p()
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "Art"


def run_flow_features(df: pd.DataFrame = None, fig_dir: Path = None):
    """เปรียบเทียบการแจกแจงแบบ Raw vs np.log1p() ของฟีเจอร์ dur, sbytes, dbytes, sload"""
    features = ["dur", "sbytes", "dbytes", "sload"]
    if df is None:
        print(f"Loading data from: {DEFAULT_DATA_PATH}")
        df = pd.read_parquet(DEFAULT_DATA_PATH, columns=features)
        
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 5] Running Flow Features Distribution & Skewness Analysis...")
    sample_df = df[features].sample(n=min(100000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    
    for i, col in enumerate(features):
        # แถวบน: Raw scale
        sns.histplot(sample_df[col], bins=30, ax=axes[0, i], color="#1e88e5", kde=False)
        axes[0, i].set_title(f"Raw: {col}", fontweight="bold")
        axes[0, i].set_ylabel("Frequency" if i == 0 else "")
        axes[0, i].tick_params(axis="x", rotation=30)
        
        # แถวล่าง: Log1p scale
        log_data = np.log1p(sample_df[col])
        sns.histplot(log_data, bins=30, ax=axes[1, i], color="#43a047", kde=True)
        axes[1, i].set_title(f"Log1p: {col}", fontweight="bold")
        axes[1, i].set_xlabel(f"log1p({col})")
        axes[1, i].set_ylabel("Frequency" if i == 0 else "")
    
    fig.suptitle("5. Heavy-Tailed Flow Features: Raw Distribution vs np.log1p() Transformation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_path = out_dir / "05_flow_features_log_dist.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_flow_features()

