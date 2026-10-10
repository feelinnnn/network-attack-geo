"""Progress 1 — Ingest: Raw CSV -> Parquet

รัน (จาก root ของ repo):  python progress1/ingest.py

สิ่งที่ทำ
1. อ่านชื่อคอลัมน์จาก NUSW-NB15_features.csv (ไฟล์ข้อมูลไม่มี header)
2. โหลด UNSW-NB15_1.csv ถึง _4.csv แล้วต่อกัน พร้อมคอลัมน์ source_file
3. บันทึกเป็น data/staging/unsw_flows.parquet

ยังไม่ทำความสะอาดข้อมูล (เป็นงานของ Progress 2)
"""
