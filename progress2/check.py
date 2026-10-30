"""Progress 2 — ตรวจเกณฑ์ผ่าน

Input : ไฟล์ output ทั้งหมดของ clean.py และ join.py

ต้องตรวจ
1. attack_cat เหลือ 10 ค่า ไม่มีค่าว่าง
2. ไม่มีแถวซ้ำ
3. ชนิดข้อมูลถูก และไม่มีค่าว่างในคอลัมน์ที่ห้ามว่าง
4. จำนวนแถวก่อนและหลัง join เท่ากัน
5. ip_lookup.csv มี IP ครบทุกตัว

ตัวเลขสรุป
"""
import sys
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo
 
CLASSES = {"Normal", "Generic", "Exploits", "Fuzzers", "DoS",
           "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"}  # 10 คลาสที่ต้องได้
NULLABLE = {"sport", "dsport"}  # port ว่างได้ (hex เกิน 65535, "-") คอลัมน์อื่นห้ามว่าง
TEXT_COLS = {"srcip", "dstip", "proto", "state", "service", "attack_cat",
             "src_country", "dst_country", "src_asn", "dst_asn"}  # คอลัมน์ข้อความ (geo เป็นรหัส เช่น KP / 131279)
 
# ค่าอ้างอิง (UNSW-NB15 ชุดเต็ม + ProdIPData 2026-10) ถ้าไม่ตรง = ไฟล์ดิบหรือโค้ดต่างจากทีม
EXPECTED = {
    "rows_after_clean": 2_059_414,
    "unique_ips": 49,
    "ip_type": {"PUBLIC": 35, "PRIVATE": 11, "MULTICAST": 2, "LOOPBACK": 1},
    "attack_cat": {"Normal": 1_959_771, "Exploits": 27_599, "Generic": 25_378, "Fuzzers": 21_795,
                   "Reconnaissance": 13_357, "DoS": 5_665, "Analysis": 2_184, "Backdoor": 1_983,
                   "Shellcode": 1_511, "Worms": 171},
}
 
 
def require_file(path: Path) -> None:  # หยุดพร้อมบอกชื่อไฟล์ ถ้ายังไม่ได้รันขั้นก่อนหน้า
    if not path.exists():
        raise FileNotFoundError(f"not found: {path} (run clean.py and join.py first)")
 
 
# label conflict: แถวที่ feature เหมือนกันทุกตัวแต่ attack_cat ต่างกัน แยกตามคลาส
def label_conflicts(df: pd.DataFrame, feat: dict) -> pd.DataFrame:
    X = feat["base"] + feat["check"]  # feature ที่โมเดลจะเห็น
    key = pd.util.hash_pandas_object(df[X], index=False)  # 1 แถว -> 1 ตัวเลข
    n_cls = df["attack_cat"].groupby(key.values).transform("nunique")  # แถวหน้าตาเดียวกันมีกี่คลาส
    conflict = n_cls.values > 1  # มากกว่า 1 คลาส = ทับกัน
    out = pd.DataFrame({
        "rows": df["attack_cat"].value_counts(),
        "conflict_rows": df.loc[conflict, "attack_cat"].value_counts(),
    }).fillna(0).astype("int64")
    out["conflict_pct"] = (100 * out["conflict_rows"] / out["rows"]).round(1)
    return out.sort_values("conflict_pct", ascending=False).rename_axis("attack_cat")
 
 
def main() -> None:
    cfg, feat = load_config(), load_config("features.yaml")
    proc, rep = p(cfg["paths"]["processed"]), p(cfg["paths"]["reports"])
    for f in [proc / "clean_flows.parquet", proc / "features.parquet", rep / "ip_lookup.csv"]:
        require_file(f)
 
    clean = pd.read_parquet(proc / "clean_flows.parquet", columns=["label"])  # ใช้แค่นับแถว
    df = pd.read_parquet(proc / "features.parquet")
    lk = pd.read_csv(rep / "ip_lookup.csv", dtype={"asn": str})
    df["attack_cat"] = df["attack_cat"].astype(str)
    results: list[dict] = []  # -> check_report.csv
 
    def check(name: str, ok: bool, detail: str = "") -> None:  # บันทึกผล + พิมพ์ PASS / FAIL
        results.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")) 
 
    # คำตอบ (attack_cat)
    print("Checks")
    cats = set(df["attack_cat"].unique())
    detail = f"{len(cats)} classes"
    if cats != CLASSES:  # บอกว่าคลาสไหนขาด / เกิน จะได้หาสาเหตุเร็ว
        detail += f", missing={sorted(CLASSES - cats)}, extra={sorted(cats - CLASSES)}"
    check("attack_cat: 10 classes, no missing", cats == CLASSES, detail)
    check("attack_cat: label 0 = Normal on every row",
          ((df["label"] == 0) == (df["attack_cat"] == "Normal")).all())
 
    # แถวซ้ำ
    n_dup = int(df.drop(columns=["source_file"] + feat["geo"]).duplicated().sum())  # ไม่นับ source_file และ geo ที่เพิ่มทีหลัง
    check("duplicates: none", n_dup == 0, f"{n_dup:,} duplicated")
 
    # ค่าว่าง / ชนิดข้อมูล / คอลัมน์
    has_null = [c for c in df.columns if c not in NULLABLE and df[c].isna().any()]
    check("missing: none in required columns", not has_null, ", ".join(has_null))
    not_num = [c for c in df.columns
               if c not in TEXT_COLS and not pd.api.types.is_numeric_dtype(df[c])]  # ที่ไม่ใช่ข้อความต้องเป็นตัวเลข
    check("dtype: numeric columns are numeric", not not_num, ", ".join(not_num))
    check("dtype: is_ftp_login is 0/1 only", df["is_ftp_login"].isin([0, 1]).all())
    check("dtype: ports within 0-65535",
          all(df[c].dropna().between(0, 65535).all() for c in ["sport", "dsport"]))
    need = set(feat["drop"]) | set(feat["geo"]) | set(feat["check"]) | set(feat["base"]) | {feat["target"]}
    missing = need - set(df.columns)  # ทุกคอลัมน์ใน features.yaml ต้องมีจริง
    check("columns: match features.yaml", not missing, ", ".join(sorted(missing)))
 
    # join
    check("join: same row count before/after", len(clean) == len(df), f"{len(clean):,} -> {len(df):,}")
    ips = set(df["srcip"]) | set(df["dstip"])
    lost = sorted(ips - set(lk["ip"]))  # IP ที่มีในข้อมูลแต่ไม่มีใน ip_lookup
    check("join: ip_lookup.csv covers every IP", not lost,
          f"{len(lk)} IPs" + (f", missing e.g. {lost[:10]}" if lost else ""))
 
    # ตัวเลขอ้างอิง
    check("expected: numbers match the team reference",
          len(df) == EXPECTED["rows_after_clean"]
          and len(lk) == EXPECTED["unique_ips"]
          and lk["ip_type"].value_counts().to_dict() == EXPECTED["ip_type"]
          and df["attack_cat"].value_counts().to_dict() == EXPECTED["attack_cat"],
          f"{len(df):,} rows (expected {EXPECTED['rows_after_clean']:,})")
 
    pd.DataFrame(results).to_csv(rep / "check_report.csv", index=False, encoding="utf-8-sig")
    print("saved -> reports/check_report.csv")
 
    print("\nSummary")
    print(f"  rows after clean : {len(df):,}")
    print(f"  unique IPs       : {len(lk)}  {lk['ip_type'].value_counts().to_dict()}")
    print(f"\n  attacks by src_country (label 0 / 1):\n{pd.crosstab(df['src_country'], df['label']).to_string()}")
 
    print("\nData quality (info only, does not fail)")
    conf = label_conflicts(df, feat)
    conf.to_csv(rep / "label_conflicts.csv", encoding="utf-8-sig")
    print("  rows with identical features but different attack_cat -> reports/label_conflicts.csv")
    print(conf.to_string())
 
    n_fail = sum(not r["passed"] for r in results)
    if n_fail:
        sys.exit(f"\nRESULT: {n_fail} check(s) FAILED, see above")
    print("\nRESULT: ALL CHECKS PASSED")
 
 
if __name__ == "__main__":
    main()