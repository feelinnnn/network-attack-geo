"""EDA 4: Label Conflicts Severity
วิเคราะห์อัตราแถวที่ฟีเจอร์ซ้ำซ้อนกันแต่มีคำตอบคนละคลาส (Label Conflicts)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFLICT_PATH = BASE_DIR / "reports" / "label_conflicts.csv"
DEFAULT_FIG_DIR = BASE_DIR / "reports" / "figures" / "art"


def run_label_conflicts(df: pd.DataFrame = None, fig_dir: Path = None, conflict_path: Path = None):
    """วิเคราะห์และพล็อตกราฟเปอร์เซ็นต์ Label Conflict แยกตามคลาส"""
    c_path = Path(conflict_path) if conflict_path else DEFAULT_CONFLICT_PATH
    out_dir = Path(fig_dir) if fig_dir else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA 4] Running Label Conflicts Severity Analysis...")
    if not c_path.exists():
        print(f"[-] Conflict report not found at {c_path}. Skipping.")
        return
    
    conf_df = pd.read_csv(c_path).sort_values("conflict_pct", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e53935" if pct >= 50 else "#fb8c00" if pct >= 10 else "#43a047" for pct in conf_df["conflict_pct"]]
    bars = ax.barh(conf_df["attack_cat"], conf_df["conflict_pct"], color=colors, edgecolor="black", alpha=0.85)
    
    ax.set_xlabel("Conflict Percentage (% of rows sharing identical features with other classes)", fontweight="bold")
    ax.set_ylabel("Attack Category", fontweight="bold")
    ax.set_title("4. Label Conflict: Identical Feature Rows with Conflicting Classes", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    
    for bar, pct, rows, conf in zip(bars, conf_df["conflict_pct"], conf_df["rows"], conf_df["conflict_rows"]):
        ax.text(pct + 1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}% ({conf:,}/{rows:,})", va="center", fontsize=9)
        
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    save_path = out_dir / "04_label_conflicts_severity.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


if __name__ == "__main__":
    run_label_conflicts()

