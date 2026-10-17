# Progress 1 — Raw Data + Ingest

**เป้าหมาย:** ทุกคนมีข้อมูลชุดเดียวกัน และโหลดเข้า Python ได้
**ตำแหน่งใน pipeline:** ขั้นที่ 1 Raw Data → ขั้นที่ 2 Ingest (ยังไม่ clean ยังไม่ทำโมเดล)

## งานที่ต้องทำ

1. Clone repo แล้วสร้าง branch `<ชื่อ>`
2. ดาวน์โหลดข้อมูล (ห้ามแก้ไฟล์ต้นฉบับ)
   - UNSW-NB15: `UNSW-NB15_1.csv` – `_4.csv` + `NUSW-NB15_features.csv` → `data/raw/unsw/`
   - ProdIPData release 2026-10: `prefix_attributes_2026-10_iso-ALL_parquet.zip` จาก
     https://prodipdata.com/downloads-parquet-2026-10.html (zip มีรหัสผ่าน ขอได้ที่ contact@prodipdata.com
     หรือขอจากหัวหน้าทีม) แตกไฟล์ให้ได้ `prefix_attributes_2026-10_iso-ALL.parquet` → `data/raw/prodipdata/`
3. รัน `progress1/ingest.py` → ได้ `data/staging/unsw_flows.parquet` และ `data/staging/prodipdata.parquet`
4. Push branch แล้วนำมารีวิวร่วมกัน

## ตัวเลขที่ทุกคนต้องได้ (สคริปต์ตรวจให้)

| ข้อมูล | ค่า |
|---|---|
| UNSW แถว / คอลัมน์ | 2,540,047 / 49 (+ `source_file`) |
| label 0 / 1 | 2,218,764 / 321,283 |
| ProdIPData แถว | 14,349,493 (1 แถว = 1 บล็อก /24, snapshot 2026-10, prefix ไม่ซ้ำ) |
| ProdIPData คอลัมน์ที่เลือก | 11 จาก 45 (ดู `configs/config.yaml`) |

## วิธีรัน (จาก root ของ repo)

```bash
pip install -r requirements.txt
python progress1/ingest.py
```
