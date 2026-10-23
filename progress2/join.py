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
