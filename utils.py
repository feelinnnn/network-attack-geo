"""ของใช้ร่วมกันทุก progress: อ่าน config และแปลง path

ไฟล์นี้อยู่ที่ root ของ repo ทุกสคริปต์ใน progressN/ เรียกใช้ได้ด้วย
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import load_config, p
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent   # utils.py อยู่ที่ root ของ repo


def load_config(name: str = "config.yaml") -> dict:
    """อ่านไฟล์ใน configs/ (ค่าเริ่มต้น config.yaml, ใช้ 'features.yaml' ได้)"""
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(relative: str) -> Path:
    """แปลง path ใน config ให้เป็น path เต็มจาก root ของ repo"""
    return ROOT / relative
