# รายงานผลการทดลองโมเดล Isolation Forest (Progress 4)

**ผู้จัดทำ:** Art (`branch: Art`)  
**โมเดลที่รับผิดชอบ:** Isolation Forest (Unsupervised Tree-based Anomaly Detection, Binary Classification)  
**ชุดข้อมูล:** `data/processed/` (`train.parquet`, `val.parquet`, `test.parquet` แบ่งตามลำดับเวลา 70/10/20)  
**เป้าหมายการประเมิน:** ตรวจจับความผิดปกติแบบ Binary (`Normal` vs `Attack/Anomaly`)  
**ไฟล์บันทึกผลการทดลอง:**
- `reports/results_iforest.csv` (รอบ Base Model — Without TTL / Without Geo พร้อม Contamination Grid)
- `reports/results_iforest_ttl.csv` (รอบ With TTL — Data Leakage Test)
- `reports/results_iforest_geo.csv` (รอบ Base + Geo — Shortcut Learning Test)
- `reports/results_iforest_ttl_geo.csv` (รอบ With TTL + Geo — Full Features Test)

---

## บทสรุปภาพรวม (Executive Summary)

โมเดล **Isolation Forest** แตกต่างจาก Logistic Regression อย่างมีนัยสำคัญ:
1. **Unsupervised Learning:** โมเดลไม่เห็นป้ายกำกับ (`attack_cat`) ในขณะเทรน เรียนรู้เฉพาะโครงสร้างความหนาแน่นและการกระจายตัวของข้อมูลเท่านั้น
2. **Binary Classification:** ทายผลจำแนกระหว่าง "ทราฟฟิกปกติ (Normal)" และ "ความผิดปกติ/การโจมตี (Anomaly)" ผ่านฟังก์ชัน `evaluate_binary()`
3. **Isolation Principle:** ตรวจจับ Anomaly โดยอาศัยหลักการว่าจุดที่ผิดปกติจะมีค่าคุณลักษณะแปลกแยก ทำให้ถูก "ตัดแยก (Isolate)" ออกมาด้วยจำนวนกิ่ง (Path Length) ที่สั้นกว่าจุดปกติ

### สรุปตัวเลขผลการทดลองหลักบนชุดทดสอบ (Test Set Summary):
- **Base Model (คลีน - Traffic Flow ล้วน 38 คอลัมน์):** Macro F1 = **0.7206**, PR-AUC = **0.3977**, Recall = **0.8667**
- **With TTL (Data Leakage 41 คอลัมน์):** Macro F1 = **0.7590** (+0.0384), PR-AUC = **0.4919** (+0.0942), Recall = **0.9362**
- **Base + Geo (Shortcut Learning 42 คอลัมน์):** Macro F1 = **0.7632** (+0.0426), PR-AUC = **0.4457** (+0.0480), Recall = **0.9385**
- **With TTL + Geo (รวมทุกฟีเจอร์ 45 คอลัมน์):** Macro F1 = **0.7606** (+0.0400), PR-AUC = **0.4537** (+0.0560), Recall = **0.9380**

---

## 1. การจูนไฮเปอร์พารามิเตอร์ (Hyperparameter Tuning)

### 1.1 ตารางผลการจูนค่า `contamination` บน Validation Set
ในการทดลอง Cell `[3B]` ได้ทำการทดสอบค่า `contamination` (สัดส่วนความผิดปกติที่คาดหวัง) ตั้งแต่ 0.03 ถึง 0.15:

| `contamination` | Val Macro F1 | Val PR-AUC | Val Recall | พฤติกรรมการตรวจจับ |
| :---: | :---: | :---: | :---: | :--- |
| **0.03** | 0.6147 | 0.4435 | 0.5850 | ตรวจจับเฉพาะ Anomaly ที่รุนแรงสุดโต่ง (Recall ต่ำ) |
| **0.05** | 0.6302 | 0.4435 | 0.6083 | ใกล้เคียงสัดส่วนการโจมตีเฉลี่ยทั้งก้อน |
| **0.08** | 0.6718 | 0.4435 | 0.6699 | จุดสมดุลในกรณีต้องการคุม False Positive |
| **0.10** | 0.7000 | 0.4435 | 0.7208 | Macro F1 แตะระดับ 0.70 |
| **0.13 (ค่าเริ่มต้น)** | **0.7331** | **0.4435** | **0.7974** | **จุดสมดุลที่ดีที่สุดระหว่าง F1 และ Recall** |
| **0.15** | **0.7343** | **0.4435** | **0.8269** | **Recall พุ่งสูงสุด โดย F1 สูงสุดสูสีกับ 0.13** |

---

### 1.2 การวิเคราะห์ทางสถิติของการจูนพารามิเตอร์:

1. **ทำไม PR-AUC คงที่ 0.4435 ในทุกระดับ `contamination`?**
   - นี่คือคุณสมบัติทางทฤษฎีที่ถูกต้องของ **Isolation Forest**: ค่า `decision_function(X)` (Anomaly Score) ถูกคำนวณจากโครงสร้างต้นไม้ที่สร้างเสร็จแล้ว ซึ่งค่าความผิดปกติของแต่ละแถวไม่ได้เปลี่ยนแปลงเลย
   - ค่า `contamination` ทำหน้าที่เพียงแค่กำหนด **"จุดตัด Threshold (Offset Quantile)"** เพื่อแปลง Score ต่อเนื่องให้กลายเป็น Binary Label (+1 หรือ -1) 
   - ดังนั้น PR-AUC ซึ่งวัดความสามารถในการจัดลำดับสัมพัทธ์ (Ranking) จึงคงที่เท่าเดิม 100% แต่ค่า `Macro F1` และ `Recall` จะขยับเปลี่ยนไปตามจุดตัด
2. **ทำไมเลือกช่วง `contamination = 0.13 - 0.15`?**
   - แม้สัดส่วนการโจมตีใน Train Set จะอยู่ที่ประมาณ ~3.4% แต่ใน Validation Set ที่แบ่งตามเวลา (Time-based Split) มีอัตราการโจมตีขยับขึ้นมาอยู่ที่ **8.2%** (เกิด Concept / Traffic Drift ตามที่พบใน EDA)
   - การกำหนด `contamination = 0.13` เผื่อไว้เล็กน้อย ช่วยให้โมเดลมี Threshold ที่ยืดหยุ่นพอจะดักจับทราฟฟิกโจมตีได้ถึง **79.7% - 82.7% บน Val Set** และพุ่งไปถึง **86.67% บน Test Set** โดยไม่ทำให้ False Positive พุ่งสูงเกินไป
3. **บทบาทของ `max_samples = 256`:**
   - การสุ่มซับเซตเพียง 256 จุดมาสร้างแต่ละต้นไม้ ช่วยลดปัญหา **Swamping** และ **Masking** อย่างมีประสิทธิภาพ ทำให้ต้นไม้แต่ละต้นไวต่อจุดแปลกแยกขนาดเล็ก และช่วยให้ความเร็วในการเทรนข้อมูล 1.4 ล้านแถวใช้เวลาเพียง **~5-10 วินาที** เท่านั้น

---

## 2. ผลการทดลอง Base Model (Without TTL / Without Geo — 38 คอลัมน์)

Base Model ทำการทดสอบบนฟีเจอร์การไหลของข้อมูล (Traffic Flow Features) ปราศจาก TTL และ Geolocation

### 2.1 ผลการทดลองบน Validation และ Test Sets:
- **Validation Set:** Macro F1 = **0.7220**, PR-AUC = **0.4435**, Recall = **0.8647**
- **Test Set:** Macro F1 = **0.7206**, PR-AUC = **0.3977**, Recall = **0.8667**

### 2.2 การวิเคราะห์พฤติกรรมของ Base Model:
- **การตรวจจับโดยไม่พึ่ง Label (High Unsupervised Recall):**
  โมเดลสามารถดักจับการโจมตีใน Test Set ได้สูงถึง **86.67%** โดยที่ไม่เคยเห็นป้ายกำกับว่าแถวไหนคือ Attack มาก่อน สะท้อนว่าพฤติกรรมการโจมตีใน UNSW-NB15 มีสถิติการไหลที่แปลกแยกจากทราฟฟิกปกติอย่างชัดเจนในเชิงโครงสร้าง (Structural Anomaly) เช่น ความยาวโฟลว์สั้นผิดปกติ (`dur`), อัตราการส่งแพ็กเก็ตสูงผิดปกติ (`spkts`, `sload`), หรือการเชื่อมต่อไปยังพอร์ตเดิมซ้ำๆ (`ct_dst_sport_ltm`)
- **ความเสถียรข้ามกาลเวลา (Temporal Robustness):**
  คะแนนบน Test Set (0.7206) แทบไม่ดรอปจาก Validation Set (0.7220) เลย แม้ข้อมูลจะถูกตัดแบ่งตามเวลา (Chronological Time Split) แสดงว่าคุณลักษณะของความผิดปกติในมิติ Flow มีความสม่ำเสมอในตัวเอง

---

## 3. ผลการทดลอง With TTL (Data Leakage Test — 41 คอลัมน์)

เมื่อเพิ่มฟีเจอร์กลุ่ม TTL (`sttl`, `dttl`, `ct_state_ttl`) จำนวน 3 คอลัมน์

### 3.1 ผลการทดลองบน Validation และ Test Sets:
- **Validation Set:** Macro F1 = **0.7616** (+0.0396), PR-AUC = **0.5374** (+0.0939), Recall = **0.9366** (+0.0719)
- **Test Set:** Macro F1 = **0.7590** (+0.0384), PR-AUC = **0.4919** (**+0.0942**), Recall = **0.9362** (+0.0695)

### 3.2 การวิเคราะห์เชิงลึก:
- **PR-AUC พุ่งขึ้นเกือบ 0.10 (+0.0942):**
  ใน Isolation Forest การเพิ่ม TTL ส่งผลต่อคะแนน **PR-AUC รุนแรงกว่าโมเดลใดๆ** โดยพุ่งจาก 0.3977 เป็น **0.4919**
- **คำอธิบายทางสถิติ:**
  ในมิติของ Isolation Tree ค่า `sttl = 254` เป็นจุดโดดเดี่ยวสุดขั้วเมื่อเทียบกับทราฟฟิกปกติที่มีค่าเฉลี่ย `sttl = 31` เมื่อต้นไม้สุ่มเลือกฟีเจอร์ `sttl` และสุ่มค่า Split ระหว่าง 31 ถึง 254 ทราฟฟิกโจมตีเกือบทั้งหมดจะถูกตัดแยก (Isolate) ออกมาได้ภายใน **1 หรือ 2 สปลิตแรกทันที** ทำให้ค่าเฉลี่ย Path Length สั้นลงอย่างมหาศาล Anomaly Score จึงสูงขึ้นจนดัน Recall ทะลุ **93.62%**
- นี่คือหลักฐานยืนยัน **Data Leakage ในสภาพแวดล้อมแล็บ** ผ่านมุมมองของโมเดล Unsupervised อย่างชัดเจน

---

## 4. ผลการทดลอง Base + Geo (Shortcut Learning Test — 42 คอลัมน์)

เมื่อเพิ่มฟีเจอร์ Geolocation (`src_country`, `dst_country`, `src_asn`, `dst_asn`) จำนวน 4 คอลัมน์ (โดยไม่มี TTL)

### 4.1 ผลการทดลองบน Validation และ Test Sets:
- **Validation Set:** Macro F1 = **0.7639** (+0.0419), PR-AUC = **0.4726** (+0.0291), Recall = **0.9383** (+0.0736)
- **Test Set:** Macro F1 = **0.7632** (**+0.0426**), PR-AUC = **0.4457** (+0.0480), Recall = **0.9385** (**+0.0718**)

### 4.2 การวิเคราะห์เชิงลึก:
- **Macro F1 สูงที่สุดในบรรดาโมเดลเดี่ยว (0.7632) และ Recall แตะ 93.85%:**
  การใส่ฟีเจอร์ Geolocation ทำให้คะแนน Macro F1 ของ Isolation Forest ขยับขึ้นสูงกว่ารอบที่มี TTL เสียอีก
- **ทำไม Unsupervised Tree ถึงไวต่อ Geolocation?**
  เมื่อทำ One-Hot Encoding บนฟีเจอร์ `src_country` ค่า `src_country_KP` จะกลายเป็น Binary Dummy (0 หรือ 1) เนื่องจากประเทศ `KP` มีจำนวนทราฟฟิกเพียง ~138,000 แถว ท่ามกลางทราฟฟิกญี่ปุ่น (`JP`) ที่มีเกือบ 1.9 ล้านแถว ในมุมมองของ Isolation Forest **ค่า `src_country_KP == 1` คือจุดส่วนน้อยที่โดดเดี่ยว (Sparse / Infrequent Category)** 
  ต้นไม้จึงเลือกสปลิตที่ฟีเจอร์นี้ได้ง่ายมาก และจัดให้แถวที่เป็น KP เป็น "Anomaly" ทันที ทำให้ดักจับการโจมตีได้ถึง 93.85% แต่ก็ทำให้ทราฟฟิกปกติจาก KP ถูกเหมารวมเป็น Anomaly ไปด้วย (False Positive ต่อทราฟฟิกเกาหลีเหนือ)

---

## 5. ผลการทดลอง With TTL + Geo (Full Features — 45 คอลัมน์)

เมื่อรวมทั้ง TTL Leakage และ Geolocation Shortcut เข้าด้วยกัน

### 5.1 ผลการทดลองบน Validation และ Test Sets:
- **Validation Set:** Macro F1 = **0.7623**, PR-AUC = **0.4653**, Recall = **0.9381**
- **Test Set:** Macro F1 = **0.7606**, PR-AUC = **0.4537**, Recall = **0.9380**

### 5.2 การวิเคราะห์ Information Redundancy:
- เมื่อใส่ทั้ง TTL และ Geo พร้อมกัน คะแนน **ไม่ได้รับผลบวกสะสม (No Additive Effect)** 
  - Macro F1 (0.7606) และ Recall (0.9380) อยู่ในระดับเดียวกับรอบ Base+Geo (0.7632)
  - ค่า PR-AUC (0.4537) กลับลดลงเมื่อเทียบกับรอบ With TTL เดี่ยวๆ (0.4919)
- **เหตุผล:** ในโครงสร้างของ Isolation Tree เมื่อมีฟีเจอร์ที่เป็น Shortcut 2 ตัวที่สะท้อนแหล่งกำเนิดเดียวกัน การสุ่มเลือกฟีเจอร์ทำให้ต้นไม้บางส่วนไปสปลิตบน `src_country_KP` แทนที่จะสปลิตบน `sttl` ซึ่งมีความต่อเนื่องทางตัวเลขที่ดีกว่า ทำให้ความแม่นยำในการจัดลำดับ Ranking (PR-AUC) ลดทอนลงเล็กน้อย

---

## 6. สรุปภาพรวม 2x2 Factorial Ablation Matrix (บน Test Set)

```
                        Without Geo (Traffic Flow)          With Geo (Traffic + Geo)
                     +-------------------------------+-------------------------------+
                     |                               |                               |
  Without TTL        |      1. BASE MODEL (คลีน)     |    3. BASE + GEO (Shortcut)   |
  (โมเดลโลกจริง)     |  Macro F1 : 0.7206            |  Macro F1 : 0.7632 (+0.0426)  |
                     |  PR-AUC   : 0.3977            |  PR-AUC   : 0.4457 (+0.0480)  |
                     |  Recall   : 0.8667            |  Recall   : 0.9385 (+0.0718)  |
                     |  Features : 38 คอลัมน์        |  Features : 42 คอลัมน์        |
                     |                               |                               |
                     +-------------------------------+-------------------------------+
                     |                               |                               |
  With TTL           |    2. WITH TTL (Leakage)      |   4. WITH TTL + GEO (Full)    |
  (แล็บจำลอง)        |  Macro F1 : 0.7590 (+0.0384)  |  Macro F1 : 0.7606 (+0.0400)  |
                     |  PR-AUC   : 0.4919 (+0.0942)  |  PR-AUC   : 0.4537 (+0.0560)  |
                     |  Recall   : 0.9362 (+0.0695)  |  Recall   : 0.9380 (+0.0713)  |
                     |  Features : 41 คอลัมน์        |  Features : 45 คอลัมน์        |
                     |                               |                               |
                     +-------------------------------+-------------------------------+
```

---

## 7. การเปรียบเทียบเชิงลึก: Logistic Regression vs Isolation Forest

เมื่อเปรียบเทียบโมเดลทั้งสองตัวที่คุณ Art ได้ทำการทดลองใน Progress 4:

| มิติการเปรียบเทียบ | Logistic Regression (Supervised) | Isolation Forest (Unsupervised) |
| :--- | :--- | :--- |
| **รูปแบบการจำแนก** | 10 คลาส (Multi-class: Normal + 9 Attacks) | 2 คลาส (Binary: Normal vs Anomaly) |
| **การใช้ป้ายกำกับ** | ต้องใช้ Label ทุกแถวในการเทรน | **ไม่ใช้ Label เลย** (Zero Supervised Label) |
| **จุดเด่น** | แยกแยะประเภทการโจมตีเฉพาะเจาะจงได้ (เช่น Fuzzers vs Recon) | **ตรวจจับ Unknown / Zero-day Attacks ได้ดี** เพราะไม่ขึ้นกับแพทเทิร์นเดิม |
| **ผลของ Base Model (Test)** | Macro F1 = 0.3757, Recall = 0.6088 | **Macro F1 = 0.7206, Recall = 0.8667** |
| **ผลของ TTL Leakage** | Macro F1 เพิ่ม +0.0432 (จาก 0.3757 $\rightarrow$ 0.4189) | PR-AUC เพิ่มมหาศาล **+0.0942** (จาก 0.3977 $\rightarrow$ 0.4919) |
| **ผลของ Geo Shortcut** | โมเดลจำทางลัดว่า `src_country_KP` = โจมตี | ต้นไม้มองว่า `src_country_KP` เป็นหมวดหมู่หายาก ดัน Recall สู่ 93.85% |

---

## 8. ข้อเสนอแนะเชิงสถาปัตยกรรม (Operational Recommendations)

> [!IMPORTANT]
> **สำหรับการนำไปใช้งานจริง (Production Deployment):**

1. **เลือกใช้ Base Model (Without TTL / Without Geo) 38 ฟีเจอร์:**
   - การได้ Recall สูงถึง **86.67%** โดยไม่ต้องพึ่งพา Label และไม่พึ่งพาทางลัดของ TTL หรือ Geo ถือเป็นผลลัพธ์ที่แข็งแกร่งมากสำหรับการทำ **First-line Anomaly Filter** ในระบบเครือข่ายองค์กร
2. **แนวทางการประยุกต์ใช้แบบไฮบริด (Ensemble Architecture):**
   - ควรนำ **Isolation Forest (Base)** มาเป็นด่านแรกเพื่อคัดกรองทราฟฟิกที่ผิดปกติออกจาก Normal (จับ Anomaly ได้ 86.67%)
   - จากนั้นส่งเฉพาะทราฟฟิกที่ถูกมาร์กว่าผิดปกติ ไปให้ **Logistic Regression (Base)** ทำการวิเคราะห์ต่อว่าการโจมตีนั้นเป็นประเภทใดใน 9 คลาส (Multi-class Categorization) เพื่อลดภาระการประมวลผลของระบบตรวจจับแบบรวมศูนย์

