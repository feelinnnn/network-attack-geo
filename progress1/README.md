# Progress 1 — Raw Data + Ingest

**เป้าหมาย:** ทุกคนมีข้อมูลชุดเดียวกัน และโหลดเข้า Python ได้
**รีวิวโค้ด:** 14 ต.ค.

## งานที่ต้องทำ

1. Clone repo แล้วสร้าง branch `p1/<ชื่อ>`
2. ดาวน์โหลด UNSW-NB15 (CSV 4 ไฟล์ + ไฟล์ features) ไว้ที่ `data/raw/unsw/` และ ProdIPData ไว้ที่ `data/raw/prodipdata/`
3. เขียน `progress1/ingest.py` ให้โหลด CSV ใส่ชื่อคอลัมน์ครบ 49 ตัว แล้วบันทึกเป็น Parquet ใน `data/staging/`
4. Push branch แล้วนำมารีวิวร่วมกัน

## วิธีรัน

```bash
pip install -r requirements.txt
python progress1/ingest.py
```