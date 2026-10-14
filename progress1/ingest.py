"""Progress 1 — Ingest: Raw CSV -> Parquet

รัน (จาก root ของ repo):  python progress1/ingest.py

สิ่งที่ทำ
1. อ่านชื่อคอลัมน์จาก NUSW-NB15_features.csv (ไฟล์ข้อมูลไม่มี header)
2. โหลด UNSW-NB15_1.csv ถึง _4.csv แล้วต่อกัน พร้อมคอลัมน์ source_file
3. บันทึกเป็น data/staging/unsw_flows.parquet

ยังไม่ทำความสะอาดข้อมูล (เป็นงานของ Progress 2)
"""
import os
import glob
import pandas as pd

def get_column_names(features_path):
    try:
        df_feat = pd.read_csv(features_path, encoding='latin1')
    except Exception:
        df_feat = pd.read_csv(features_path, encoding='utf-8')
    
    if 'Name' in df_feat.columns:
        col_names = df_feat['Name'].astype(str).str.strip().tolist()
    else:
        col_names = df_feat.iloc[:, 1].astype(str).str.strip().tolist()
        
    return col_names

def main():
    unsw_dir = os.path.join('data', 'raw', 'unsw')
    staging_dir = os.path.join('data', 'staging')
    os.makedirs(staging_dir, exist_ok=True)

    features_files = glob.glob(os.path.join(unsw_dir, '*features*.csv')) + glob.glob(os.path.join(unsw_dir, '*Features*.csv'))
    if not features_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ features ใน {unsw_dir}")
    
    features_path = features_files[0]
    print(f"📖 อ่านชื่อคอลัมน์จาก: {features_path}")
    column_names = get_column_names(features_path)
    print(f"พบจำนวนคอลัมน์ทั้งหมด: {len(column_names)} คอลัมน์")

    csv_files = sorted([
        f for f in glob.glob(os.path.join(unsw_dir, '*.csv'))
        if 'features' not in f.lower()
    ])

    if not csv_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล UNSW-NB15 CSV ใน {unsw_dir}")

    df_list = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"กำลังโหลด: {filename}")
        
        # อ่านโดยระบุ dtype=str ชั่วคราวเพื่อป้องกัน Data Type ผสมกันในคอลัมน์ เช่น sport/dsport
        df = pd.read_csv(file_path, header=None, names=column_names, dtype=str, low_memory=False)
        df['source_file'] = filename
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)

    # เซฟด้วย pyarrow engine
    output_parquet = os.path.join(staging_dir, 'unsw_flows.parquet')
    full_df.to_parquet(output_parquet, index=False, engine='pyarrow')
    
    print(f"✅ บันทึกไฟล์สำเร็จ: {output_parquet}")
    print(f"📊 จำนวนแถวทั้งหมด: {len(full_df):,} แถว | คอลัมน์: {len(full_df.columns)} คอลัมน์")

if __name__ == '__main__':
    main()