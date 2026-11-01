# Progress 3 — EDA

**เป้าหมาย:** สำรวจข้อมูลก่อนเทรน รู้ว่าข้อมูลเป็นยังไง มีอะไร และควรเทรนแบบไหน
**ตำแหน่งใน pipeline:** ขั้นที่ 5 EDA

## งานที่ต้องทำ

1. ใช้ branch เดิมของตัวเอง แล้วดึง main ล่าสุดเข้ามา
   ```bash
   git fetch origin
   git switch ชื่อ-branch-ตัวเอง
   git merge origin/main
   ```
   ถ้าขึ้น CONFLICT ใน `progress2/` หรือ `reports/` (เพราะเคยเขียน Progress 2 เอง) ให้ใช้ของ main:
   ```bash
   git checkout origin/main -- progress2 reports
   git commit -m "merge main: use team Progress 2"
   ```
2. รัน Progress 2 ให้ได้ `data/processed/features.parquet` (ต้องผ่าน `python progress2/check.py`)
3. ทำ EDA อิสระ **ขั้นต่ำคนละ 5 อย่าง** ใน `eda.py`
   แต่ละอย่างต้องตอบได้ว่า "เห็นอะไร → ตัดสินใจ / เข้าใจอะไร"

## ที่เก็บไฟล์

| อะไร | เก็บที่ |
|------|--------|
| โค้ด | `progress3/eda.py` หรือ `progress3/eda_<ชื่อ>.ipynb` |
| กราฟ | `reports/figures/<ชื่อ>_*.png` (ขึ้นต้นด้วยชื่อตัวเอง กันทับกัน) |
| สรุปข้อค้นพบ | `reports/eda_<ชื่อ>.md` |

## ติดตั้งเพิ่ม

```bash
pip install matplotlib seaborn
```