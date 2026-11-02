# EDA Findings -- keen

รันจาก `python progress3/eda.py` ทั้ง 7 แนวทาง

## 1. Class Imbalance & Rare Classes

**เห็นอะไร:** Normal มี 1,959,771 แถว (95.2%) คลาสน้อยสุดคือ Worms มีแค่ 171 แถว (0.008%)

**ตัดสินใจ/เข้าใจอะไร:** ต้องใช้ Stratified Split เสมอ และวัดผลด้วย Macro F1-Score ไม่ใช่ Accuracy เพราะโมเดลที่ทายว่า Normal ทุกแถวจะได้ Accuracy 95% แต่ F1 ของคลาสโจมตี = 0

![Class Imbalance & Rare Classes](figures/keen_1_*.png)

## 2. Geo Shortcut Risk

**เห็นอะไร:** การโจมตีทั้ง 99,643 แถว มาจาก KP (ASN 131279) ทั้งหมด 100% ส่วน JP และ AU เป็น Normal ล้วน -> Geo = Perfect Predictor ในชุดนี้

**ตัดสินใจ/เข้าใจอะไร:** ต้องสร้าง Baseline Model แบบ Base-only (ไม่มี Geo) เทียบกับ Base+Geo ถ้า Geo ดีขึ้นมาก แปลว่าโมเดลอาศัย Shortcut -> ใช้จริงไม่ได้ (โลกจริงการโจมตีมาจากทุกประเทศ ไม่ใช่แค่ KP)

![Geo Shortcut Risk](figures/keen_2_*.png)

## 3. TTL Leakage

**เห็นอะไร:** sttl/dttl แยก Normal กับ Attack ได้ชัดเจนผิดปกติ เพราะ UNSW-NB15 ใช้เครื่องยิงคนละ OS (TTL ต่างกัน) ค่า ct_state_ttl ที่สูง >= 2 มักเป็น Attack

**ตัดสินใจ/เข้าใจอะไร:** ฟีเจอร์ TTL เป็น Data Leakage จาก Lab Setup -> ต้องทดลองเทรนทั้งมีและไม่มี TTL สอดคล้องกับ features.yaml ที่แยก check: [sttl, dttl, ct_state_ttl] ไว้

![TTL Leakage](figures/keen_3_*.png)

## 4. Heavy-tailed Distributions

**เห็นอะไร:** ฟีเจอร์ volume/rate (dur, sbytes, sload ฯลฯ) มี skewness สูงมาก (เช่น sbytes skew=130) ค่าส่วนใหญ่เล็กมาก แต่มี outlier สูงหลักล้าน

**ตัดสินใจ/เข้าใจอะไร:** ใช้ log10(x+1) transform ก่อนเข้าโมเดล Linear/NN เพื่อลด skewness หรือเลือกใช้ Tree-based model (XGBoost/LightGBM) ที่ไม่ไวต่อ scale

![Heavy-tailed Distributions](figures/keen_4_*.png)

## 5. Protocol & Service Profiling

**เห็นอะไร:** proto มี 135 ค่าย่อย แต่ TCP+UDP คิดเป็น 98.9% ของข้อมูลทั้งหมด -- service '-' (ไม่ระบุ) คิดเป็นกว่าครึ่ง การโจมตีแต่ละประเภทมี service profile ต่างกัน เช่น Fuzzers/DoS มักเกิดบน '-' (no service)

**ตัดสินใจ/เข้าใจอะไร:** จัดกลุ่ม proto ที่พบน้อยรวมเป็น 'other' เพื่อลดมิติ One-Hot Encoding ฟีเจอร์ service มีประโยชน์ในการแยก attack type ควรเก็บไว้

![Protocol & Service Profiling](figures/keen_5_*.png)

## 6. Multicollinearity

**เห็นอะไร:** พบ 28 คู่ฟีเจอร์ที่มี |Spearman r| >= 0.85 เช่น trans_depth<->ct_flw_http_mthd (+1.000), is_ftp_login<->ct_ftp_cmd (+0.999), swin<->dwin (+0.996), dloss<->dpkts (+0.982), spkts<->dpkts (+0.981)

**ตัดสินใจ/เข้าใจอะไร:** คู่ที่ซ้ำซ้อนมาก (เช่น ct_srv_src <-> ct_srv_dst) สามารถตัดตัวหนึ่งออกได้ เพื่อลดมิติ หรือปล่อยให้ Tree-based model จัดการเอง แต่ต้องระวังตอน interpret

![Multicollinearity](figures/keen_6_*.png)

## 7. Label Conflicts

**เห็นอะไร:** คลาส Backdoor มี conflict rate สูงสุดที่ 82.5% หมายถึง 1,636 แถวมีฟีเจอร์เหมือนกันทุกตัว แต่ถูกติดป้ายคนละ attack_cat

**ตัดสินใจ/เข้าใจอะไร:** โมเดลไม่มีทางทำนาย Backdoor/Analysis ได้ถูก 100% เพราะมี Noise/Ambiguity ในตัวข้อมูล ควรรายงานขีดจำกัดนี้ใน report และใช้เป็น upper bound สำหรับ evaluation

![Label Conflicts](figures/keen_7_*.png)

---
*สร้างอัตโนมัติจาก progress3/eda.py*
