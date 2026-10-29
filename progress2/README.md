# Progress 2 — Clean + Join

**เป้าหมาย:** ทุกคนได้ข้อมูลที่ clean แล้วชุดเดียวกัน และรู้ว่าแต่ละ IP อยู่ประเทศ / ASN ไหน
**ตำแหน่งใน pipeline:** ขั้นที่ 3 Clean → ขั้นที่ 4 Join (ยังไม่ทำโมเดล)

## งานที่ต้องทำ

1. `git pull origin main` แล้ว merge เข้า branch ตัวเอง
2. รัน Progress 1 ใหม่ (ProdIPData เปลี่ยนเป็นไฟล์จาก prodipdata.com และเลือกคอลัมน์เพิ่ม)
3. เขียน `clean.py` → `join.py` → `check.py` (รายละเอียดอยู่ในหัวไฟล์แต่ละไฟล์)
4. ดู `configs/features.yaml` (กำหนดแล้ว) ให้ชื่อคอลัมน์ geo ใน features.parquet ตรงกับไฟล์นี้
5. Push branch → รีวิวร่วมกัน → เลือกโค้ดที่ดีที่สุดเข้า `main`

## ไฟล์

| ไฟล์ | หน้าที่ |
|------|--------|
| `clean.py` | staging → `clean_flows.parquet` + `cleaning_report.csv` |
| `join.py` | clean + ProdIPData → `features.parquet` + `ip_lookup.csv` + `join_report.csv` |
| `check.py` | ตรวจเกณฑ์ผ่าน และพิมพ์ตัวเลขไว้เทียบกัน |

ไฟล์ parquet เก็บใน `data/processed/` (ไม่ commit), ไฟล์ csv เก็บใน `reports/` (commit ได้ ใช้เทียบผลกัน)

## กฎที่ตกลงแล้ว

**Clean** — แก้ค่าทั้งหมดก่อน แล้วค่อยลบแถวซ้ำเป็นขั้นสุดท้าย

| ปัญหา | วิธีจัดการ |
|-------|-----------|
| ค่าว่างใน `ct_ftp_cmd`, `is_ftp_login`, `ct_flw_http_mthd` (`ct_ftp_cmd` ว่างเป็น `" "`) | เติม 0 |
| `is_ftp_login` มีค่า 2 (30 แถว) และ 4 (156 แถว) | แปลงเป็น 1 |
| port เป็น hex ไม่เกิน 65535 เช่น `0xcc09` | แปลงด้วย `int(x, 16)` |
| port เป็น hex เกิน 65535 เช่น `0x20205321` | ให้เป็นค่าว่าง |
| port เป็น `-` | ให้เป็นค่าว่าง |
| `attack_cat` ว่างตรงที่ `label = 0` | เติม `Normal` |
| `attack_cat` มีช่องว่างหน้า/หลัง | ตัดช่องว่าง |
| `Backdoors` | เปลี่ยนเป็น `Backdoor` |
| แถวซ้ำ | ลบแถว (ขั้นสุดท้าย) |

**Join กับ ProdIPData**

1. แปลง IP ให้อยู่ในรูปแบบ `IP24Prefix`: เปลี่ยนเลขชุดสุดท้ายเป็น 0 แล้วต่อ `/24` เช่น `175.45.176.3` → `175.45.176.0/24`
2. เช็กประเภท IP ก่อน: `PUBLIC` → นำไปหาในตาราง, `PRIVATE` / `LOOPBACK` / `MULTICAST` → ใส่ชื่อประเภทแทน
   (เช็ก `LOOPBACK` ก่อน `PRIVATE` เพราะ Python นับ 127.x เป็น private ด้วย)
3. หาใน `IP24Prefix` เจอ → ได้ `CountryCode` และ `ASNID`, ไม่เจอ → `UNKNOWN`
4. ทำทั้ง `srcip` และ `dstip` ได้ 4 คอลัมน์: `src_country`, `dst_country`, `src_asn`, `dst_asn`
5. ค่า `"-"` ใน ProdIPData คือค่าว่าง
6. จำนวนแถวหลัง join ต้องเท่าเดิม

## เกณฑ์ผ่าน (ทุกคนต้องได้ตรงกัน)

- `attack_cat` เหลือ 10 ค่า
- จำนวนแถวหลังลบแถวซ้ำตรงกันทุกคน
- จำนวนแถวก่อนและหลัง join เท่ากัน
- `ip_lookup.csv` ตรงกันทุกคน
- schema ผ่าน (ชนิดข้อมูลถูก ไม่มีค่าว่างในคอลัมน์ที่ห้ามว่าง)