"""Progress 2 — Clean

Input : data/staging/unsw_flows.parquet  (ผลจาก Progress 1)
Output: data/processed/clean_flows.parquet
        reports/cleaning_report.csv      (กฎแต่ละข้อแก้ไปกี่แถว)

ต้องทำ
1. attack_cat: ตัดช่องว่างหน้า/หลัง, Backdoors → Backdoor, ค่าว่างที่ label = 0 → Normal (ต้องเหลือ 10 ค่า)
2. ค่าว่างใน ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd → เติม 0 (ct_ftp_cmd ว่างเป็นช่องว่าง " " ไม่ใช่ NaN)
3. is_ftp_login ค่า 2 และ 4 → 1
4. sport, dsport: hex ไม่เกิน 65535 → int(x, 16), hex เกิน 65535 และ "-" → ค่าว่าง
5. ลบแถวซ้ำ เป็นขั้นสุดท้าย
6. บันทึกจำนวนแถวที่ถูกแก้ของทุกกฎลง cleaning_report.csv

"""
