"""อ่าน config กลางของทีม"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]   # root ของ repo


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(relative: str) -> Path:
    """แปลง path ใน config ให้เป็น path เต็มจาก root ของ repo"""
    return ROOT / relative
