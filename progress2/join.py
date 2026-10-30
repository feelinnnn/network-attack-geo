"""Progress 2 — Join กับ ProdIPData

Input : data/processed/clean_flows.parquet, data/staging/prodipdata.parquet
Output: data/processed/features.parquet  (clean + คอลัมน์ geo)
        reports/ip_lookup.csv            (IP ทุกตัวใน UNSW → ประเภท / ประเทศ / ASN)
        reports/join_report.csv          (จำนวน IP ที่หาเจอ / หาไม่เจอ / private ฯลฯ)

ต้องทำ
1. แยกประเภท IP: PUBLIC / PRIVATE / LOOPBACK / MULTICAST (ระวัง 127.x ถูกนับเป็น private ด้วย)
2. IP แบบ PUBLIC → แปลงเป็นรูปแบบเดียวกับ IP24Prefix (เช่น 175.45.176.3 → 175.45.176.0/24) แล้วหาในตาราง
3. หาไม่เจอ → UNKNOWN, ไม่ใช่ PUBLIC → ใส่ชื่อประเภทแทน
4. ทำทั้ง srcip และ dstip ได้คอลัมน์ประเทศและ ASN ของทั้งสองฝั่ง
5. จำนวนแถวก่อนและหลัง join ต้องเท่ากัน
6. ProdIPData ใช้ "-" แทนค่าว่าง

"""
import ipaddress
import sys
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo
 
STATUSES = ["FOUND", "UNKNOWN", "PRIVATE", "LOOPBACK", "MULTICAST", "INVALID"]  # สถานะทั้งหมดใน join_report
 
 
def require_file(path: Path) -> None:  # หยุดพร้อมบอกชื่อไฟล์ ถ้ายังไม่ได้รันขั้นก่อนหน้า
    if not path.exists():
        raise FileNotFoundError(f"not found: {path} (run the previous step first)")
 
 
# IP -> ประเภท / บล็อก /24
def ip_type(ip: str) -> str:  # ProdIPData มีแค่ PUBLIC ประเภทอื่นไม่ต้องหา ใส่ชื่อประเภทแทน
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return "INVALID"  # รูปแบบ IP ผิด
    if addr.is_loopback:
        return "LOOPBACK"  # 127.x ต้องเช็กก่อน private เพราะ Python นับ 127.x เป็น private ด้วย
    if addr.is_multicast:
        return "MULTICAST"  # 224.x ส่งหาหลายเครื่องพร้อมกัน
    if addr.is_private:
        return "PRIVATE"  # 10.x, 192.168.x วงแลน
    return "PUBLIC"  # IP บนอินเทอร์เน็ต -> ไปหาใน ProdIPData
 
 
def to_prefix24(ip: str) -> str:  # 175.45.176.3 -> 175.45.176.0/24 (รูปเดียวกับ IP24Prefix)
    return ip.strip().rsplit(".", 1)[0] + ".0/24"
 
 
# ProdIPData
def load_prodip(cfg: dict, prefixes: list[str]) -> pd.DataFrame:  # อ่านเฉพาะบล็อกที่ใช้ (4 จาก 14 ล้าน) ประหยัด RAM
    path = p(cfg["paths"]["staging"]) / "prodipdata.parquet"
    require_file(path)
    cols = [c for c in cfg["prodipdata"]["columns"] if c != "snapshot_month"]  # snapshot_month ค่าเดียวทุกแถว ไม่ต้องใช้
    ip = pd.read_parquet(path, columns=cols, filters=[("IP24Prefix", "in", prefixes)])  # filters = อ่านเฉพาะแถวที่ตรงรายการ
    ip = ip.replace(cfg["prodipdata"]["missing_value"], pd.NA)  # "-" = ไม่มีข้อมูล -> ค่าว่าง (จะกลายเป็น UNKNOWN ทีหลัง)
    if ip["IP24Prefix"].duplicated().any():  # 1 บล็อกต้องมี 1 แถว ไม่งั้น join แล้วแถวเพิ่ม
        raise ValueError("IP24Prefix duplicated in prodipdata")
    return ip.set_index("IP24Prefix")  # ค้นหาด้วยชื่อบล็อกได้ทันที
 
 
# ตาราง IP -> ประเทศ / ASN (1 แถวต่อ 1 IP)
def build_ip_lookup(flows: pd.DataFrame, cfg: dict) -> pd.DataFrame:  # 2 ล้านแถวมี IP ไม่ซ้ำแค่ 49 ตัว -> หาครั้งเดียวต่อ IP
    src_n, dst_n = flows["srcip"].value_counts(), flows["dstip"].value_counts()  # IP นี้เป็นผู้ส่ง / ผู้รับกี่แถว
    lk = pd.DataFrame({"ip": sorted(set(src_n.index) | set(dst_n.index))})  # รวม IP ทั้ง 2 ฝั่งแบบไม่ซ้ำ
    lk["ip_type"] = lk["ip"].map(ip_type)
    is_pub = lk["ip_type"] == "PUBLIC"
    lk["ip24prefix"] = lk["ip"].where(is_pub).map(to_prefix24, na_action="ignore")  # แปลงเฉพาะ PUBLIC
 
    prodip = load_prodip(cfg, sorted(lk["ip24prefix"].dropna().unique()))  # ส่งรายการบล็อกที่ต้องหาไป filters
    lk = lk.join(prodip, on="ip24prefix")  # ต่อคอลัมน์ของ ProdIPData ตามบล็อก
    found = is_pub & lk["ip24prefix"].isin(prodip.index)  # หาบล็อกเจอในตารางไหม
 
    # PUBLIC เจอ -> ค่าจากตาราง ("-" -> UNKNOWN), PUBLIC ไม่เจอ -> UNKNOWN, อื่นๆ -> ชื่อประเภท
    lk["status"] = lk["ip_type"].where(~is_pub, "UNKNOWN").mask(found, "FOUND")
    for new, col in [("country", "CountryCode"), ("asn", "ASNID")]:
        lk[new] = lk["ip_type"].where(~is_pub, lk[col].astype("string").fillna("UNKNOWN"))
    lk = lk.drop(columns=["CountryCode", "ASNID"])  # ได้เป็น country / asn แล้ว
 
    lk["src_rows"] = lk["ip"].map(src_n).fillna(0).astype("int64")  # จำนวนแถวที่ IP นี้เป็นผู้ส่ง
    lk["dst_rows"] = lk["ip"].map(dst_n).fillna(0).astype("int64")  # จำนวนแถวที่ IP นี้เป็นผู้รับ
    first = ["ip", "ip_type", "status", "ip24prefix", "country", "asn"]  # คอลัมน์สำคัญไว้หน้าสุด
    return lk[first + [c for c in lk.columns if c not in first]]
 
 
# สรุปผลการ join (all = ภาพรวม IP, src / dst = แยกฝั่ง พร้อมจำนวนแถว)
def join_report(lk: pd.DataFrame, rows_before: int, rows_after: int) -> pd.DataFrame:
    out = [("all", "unique_ips", len(lk), None)]
    all_n = lk["status"].value_counts()
    out += [("all", status, int(all_n.get(status, 0)), None) for status in STATUSES]  # กี่ IP ในแต่ละสถานะ
    for side in ["src", "dst"]:
        g = lk[lk[f"{side}_rows"] > 0].groupby("status")  # เฉพาะ IP ที่เคยอยู่ฝั่งนี้
        for status in STATUSES:
            n_ip = int(g.size().get(status, 0))  # กี่ IP
            n_rows = int(g[f"{side}_rows"].sum().get(status, 0))  # รวมกี่แถว
            out.append((side, status, n_ip, n_rows))
    out.append(("all", "rows_before_join", None, rows_before))  # ก่อน/หลังต้องเท่ากัน
    out.append(("all", "rows_after_join", None, rows_after))
    return pd.DataFrame(out, columns=["side", "status", "ips", "rows"]).astype({"ips": "Int64", "rows": "Int64"})
 
 
def main() -> None:
    cfg = load_config()
    feat_cfg = load_config("features.yaml")
    proc, rep_dir = p(cfg["paths"]["processed"]), p(cfg["paths"]["reports"])
    require_file(proc / "clean_flows.parquet")
    rep_dir.mkdir(parents=True, exist_ok=True)
 
    print("load clean_flows + build IP -> country / ASN")
    flows = pd.read_parquet(proc / "clean_flows.parquet")
    rows_before = len(flows)
    lk = build_ip_lookup(flows, cfg)
    print(f"  {rows_before:,} rows, unique IPs {len(lk)}: {lk['ip_type'].value_counts().to_dict()}")
 
    print("put country / ASN into flows")
    by_ip = lk.set_index("ip")
    for side in ["src", "dst"]:  # เติมกลับทุกแถวตาม IP ของแต่ละฝั่ง
        flows[f"{side}_country"] = flows[f"{side}ip"].map(by_ip["country"])
        flows[f"{side}_asn"] = flows[f"{side}ip"].map(by_ip["asn"])
    geo = feat_cfg["geo"]  # ชื่อคอลัมน์ geo ต้องตรงกับ features.yaml
    flows = flows[[c for c in flows.columns if c not in geo] + geo]  # ย้ายคอลัมน์ geo ไปท้ายตาราง
 
    if len(flows) != rows_before:  # แถวต้องไม่เพิ่มไม่หาย
        raise AssertionError("number of rows after join is not the same as before")
    if flows[geo].isna().any().any():  # ทุกแถวต้องได้ค่า geo
        raise AssertionError("there are rows without geo values")
 
    print("save features.parquet + ip_lookup.csv + join_report.csv")
    flows.to_parquet(proc / "features.parquet", index=False)
    lk.to_csv(rep_dir / "ip_lookup.csv", index=False, encoding="utf-8-sig")
    report = join_report(lk, rows_before, len(flows))
    report.to_csv(rep_dir / "join_report.csv", index=False, encoding="utf-8-sig")
    print("saved -> data/processed/features.parquet")
    print("saved -> reports/ip_lookup.csv, reports/join_report.csv\n")
    print(report[(report["rows"].fillna(1) > 0) & (report["ips"].fillna(1) > 0)].to_string(index=False))  # ซ่อนแถวที่เป็น 0
    print("\npublic blocks:")
    print(lk[lk["status"] == "FOUND"][["ip24prefix", "country", "asn", "ASNName"]]
          .drop_duplicates().to_string(index=False))
 
 
if __name__ == "__main__":
    main()