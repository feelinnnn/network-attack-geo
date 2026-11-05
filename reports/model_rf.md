# Random Forest ของ Pimmy

## 1. สรุปสั้นส

เลือก **RF-base** เป็นโมเดลหลัก เพราะใช้ feature พฤติกรรมของ traffic โดยไม่พึ่งพา
TTL หรือข้อมูล Geo ที่อาจเป็นลักษณะเฉพาะของห้องแลป

| Model | Macro F1 | Macro PR-AUC | Macro Recall | บทบาท |
|---|---:|---:|---:|---|
| RF-base | **0.5865** | **0.6021** | **0.6687** | โมเดลหลัก |
| RF-base + TTL | 0.6021 | 0.6159 | 0.6854 | ตรวจ lab leakage |
| RF-base + Geo | 0.5869 | 0.5942 | 0.6648 | ตรวจ geo shortcut |

การเพิ่ม TTL ทำให้ Macro F1 เพิ่มขึ้น `+0.0156` แต่คะแนนที่เพิ่มอาจมาจากสัญญาณ
เฉพาะของ lab เช่น OS, เครื่อง หรือ topology จึงใช้ TTL เป็นการตรวจสอบ leakage
เท่านั้น ไม่ใช้เป็น feature ของโมเดลหลัก ส่วน Geo เพิ่ม Macro F1 เพียง `+0.0004`
แต่ทำให้ PR-AUC ลดลง จึงยังไม่เห็นประโยชน์ที่ชัดเจน

## 2. โจทย์ของงาน

โจทย์คือจำแนกประเภท network traffic จากข้อมูลการโจมตีทางไซเบอร์ที่เก็บในแลป
ออกเป็น 10 classes:

`Normal, Exploits, Generic, Fuzzers, Reconnaissance, DoS, Analysis, Backdoor, Shellcode, Worms`

ข้อมูลมี class ไม่สมดุลอย่างมาก หากใช้ Accuracy เพียงอย่างเดียว โมเดลอาจทาย
Normal ซึ่งมีจำนวนมากได้ดี แต่ยังตรวจจับ attack class ที่มีจำนวนน้อยไม่ได้
ดังนั้นจึงใช้ Macro F1 เป็น metric หลัก และรายงาน PR-AUC กับ Recall ประกอบ

## 3. ข้อมูลและการแบ่งชุด

การแบ่งข้อมูลทำโดย [split.py](../progress4/split.py) ซึ่งไม่ได้แก้ไข:

| ชุด | จำนวนแถว | การใช้งาน |
|---|---:|---|
| Train | 1,441,565 | fit preprocessing และ train model |
| Validation | 205,948 | เลือก hyperparameter |
| Test | 411,901 | วัดผลสุดท้ายเพียงครั้งเดียว |

กติกาการแบ่ง:

- เรียงข้อมูลทุกแถวตาม `stime`
- แบ่งตามเวลา 70/10/20
- ใช้ seed `42`
- ใช้ข้อมูลเต็มทั้งหมดในผล official ล่าสุด
- ไม่ได้สุ่มเหลือ 100,000 แถว

การแบ่งตามเวลาช่วยจำลองการใช้ข้อมูลในอดีตเพื่อทำนายข้อมูลในช่วงเวลาถัดไป
และลดความเสี่ยงที่ข้อมูลอนาคตจะรั่วกลับเข้า train

เนื่องจากข้อมูลไม่สมดุลมาก test มี Normal 378,486 แถว แต่ Worms มีเพียง 56 แถว
จึงต้องอ่านผลราย class ควบคู่กับคะแนนเฉลี่ย

## 4. Feature ที่ใช้

### 4.1 RF-base: โมเดลหลัก

RF-base ใช้ feature พฤติกรรม traffic จำนวน 38 ตัวจากส่วน `base` ใน
`configs/features.yaml`:

```text
proto, state, dur, sbytes, dbytes, sloss, dloss, service,
sload, dload, spkts, dpkts, swin, dwin, stcpb, dtcpb,
smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit,
sintpkt, dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports,
ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src,
ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm,
ct_dst_sport_ltm, ct_dst_src_ltm
```

จงใจไม่ใช้:

- IP และ port เพราะโมเดลอาจจำตัวเครื่องแทนการเรียนรู้พฤติกรรม
- Geo: `src_country`, `dst_country`, `src_asn`, `dst_asn`
- TTL: `sttl`, `dttl`, `ct_state_ttl`
- timestamp ในฐานะ feature

### 4.2 ชุดตรวจสอบเพิ่มเติม

ทำ ablation study โดยเพิ่ม feature ทีละชุดและใช้ hyperparameter เดียวกับ RF-base:

- **RF-base + TTL**: เพิ่ม `sttl`, `dttl`, `ct_state_ttl` รวมเป็น 41 features
- **RF-base + Geo**: เพิ่ม `src_country`, `dst_country`, `src_asn`, `dst_asn`
  รวมเป็น 42 features

วัตถุประสงค์ของรอบนี้คือดูว่าคะแนนเปลี่ยนเพราะ feature เหล่านี้หรือไม่ ไม่ใช่
จูน hyperparameter ใหม่ให้แต่ละชุด

## 5. การเตรียมข้อมูลในโมเดล

โค้ดอยู่ที่ [random_forest.py](../progress4/models/random_forest.py) และรวม
preprocessing กับ classifier ไว้ใน Pipeline เดียวกัน:

1. Numeric columns เติม missing ด้วย median ที่เรียนรู้จาก train
2. Categorical columns เติม missing ด้วยค่าที่พบบ่อย
3. Categorical columns แปลงด้วย One-Hot Encoding
4. ใช้ `handle_unknown="ignore"` เพื่อรองรับ category ที่พบใน val/test แต่ไม่พบใน train
5. ใช้ `class_weight="balanced_subsample"` ลดผลจาก Normal ที่มีจำนวนมาก
6. ใช้ `random_state=42` และ `n_jobs=-1`

การใช้ Pipeline ทำให้ encoder และค่าที่ใช้เติม missing ถูก fit จากข้อมูล train
ของรอบนั้นเท่านั้น ไม่ได้นำ val/test มาคำนวณล่วงหน้า

## 6. Metrics และกติกาการทดลอง

ใช้ [metrics.py](../progress4/metrics.py) เดิมของทีม โดยไม่แก้ไข:

- **Macro F1**: metric หลัก ใช้เลือก configuration เพราะเฉลี่ย F1 ของทุก class
  เท่า ๆ กัน เหมาะกับข้อมูลไม่สมดุล
- **Macro PR-AUC**: วัดคุณภาพการจัดอันดับความมั่นใจของแต่ละ class โดยเฉพาะ
  class ที่มีจำนวนน้อย
- **Macro Recall**: วัดความสามารถในการตรวจจับทุก class โดยไม่ให้ class ใหญ่กลบ
  class เล็ก

กติกาหลัก:

1. fit แต่ละ configuration ด้วย train เท่านั้น
2. ประเมินบน validation
3. เลือก Macro F1 สูงสุด และใช้ PR-AUC เป็นตัวตัดสินรองเมื่อคะแนนเท่ากัน
4. เมื่อเลือกค่าแล้ว จึงรวม train+val
5. fit โมเดล final ด้วย hyperparameter เดิม
6. วัด test เพียงครั้งเดียว

## 7. การจูน RF-base บน validation

ทดลอง 4 configurations โดยใช้ 30 trees ในช่วงค้นหา เพื่อให้ทดลองครบภายใต้
ทรัพยากรที่มี:

| Config | `max_depth` | `min_samples_leaf` | `max_features` | Val Macro F1 |
|---|---:|---:|---|---:|
| 1 | None | 1 | sqrt | 0.5716 |
| 2 | 30 | 1 | sqrt | 0.5689 |
| 3 | None | 2 | sqrt | **0.5947** |
| 4 | None | 1 | log2 | 0.5379 |

จึงเลือก configuration 3:

```text
n_estimators=30
max_depth=None
min_samples_leaf=2
max_features="sqrt"
class_weight="balanced_subsample"
random_state=42
```

รอบ final ใช้ค่าชุดเดียวกันทุกตัว ไม่เพิ่ม `n_estimators` เป็น 60 เพื่อให้ตรงกับ
หลักการ “จูนบน val แล้วใช้ค่าเดิมไปวัด test”

## 8. ผล RF-base ล่าสุด

### ผลบน validation

- Macro F1: **0.5947**
- Macro PR-AUC: **0.6098**
- Macro Recall: **0.6625**

### ผลบน test

- Macro F1: **0.5865**
- Macro PR-AUC: **0.6021**
- Macro Recall: **0.6687**

ผล test ต่ำกว่า validation ใน Macro F1 และ PR-AUC เล็กน้อย แต่ Recall สูงกว่า
เล็กน้อย ซึ่งเป็นเรื่องที่เป็นไปได้เมื่อ distribution ของ class และรูปแบบ traffic
เปลี่ยนไปตามเวลา

## 9. ผล leakage และ shortcut ล่าสุด

ผลทั้งหมดมาจากการ train ด้วย train+val หลังวัด validation และใช้ hyperparameter
เดียวกับ RF-base:

| Model | Macro F1 | Macro PR-AUC | Macro Recall |
|---|---:|---:|---:|
| RF-base | 0.5865 | 0.6021 | 0.6687 |
| RF-base + TTL | 0.6021 | 0.6159 | 0.6854 |
| RF-base + Geo | 0.5869 | 0.5942 | 0.6648 |

### 9.1 TTL leakage gap

- Macro F1: `0.6021 - 0.5865 = +0.0156`
- PR-AUC: `0.6159 - 0.6021 = +0.0138`
- Recall: `0.6854 - 0.6687 = +0.0167`

TTL มีสัญญาณช่วยแยก class ใน dataset นี้จริง แต่ยังสรุปไม่ได้ว่าเป็น feature
ที่ generalize ไปยังเครือข่ายอื่น เพราะอาจสะท้อน OS, เครื่อง, topology หรือเงื่อนไข
เฉพาะของห้องแลป ดังนั้นเก็บผล TTL ไว้เป็น diagnostic ของ leakage และไม่เลือกเป็น
โมเดลหลัก

### 9.2 Geo shortcut gap

- Macro F1: `0.5869 - 0.5865 = +0.0004`
- PR-AUC: `0.5942 - 0.6021 = -0.0079`
- Recall: `0.6648 - 0.6687 = -0.0039`

Geo ไม่ได้ช่วยอย่างชัดเจนภายใต้ time split นี้ แม้ Macro F1 จะเพิ่มเล็กน้อย แต่
PR-AUC และ Recall ลดลง จึงไม่เพิ่ม Geo เข้าโมเดลหลัก

## 10. Recall ราย class ของ RF-base

จาก [rf_base_class_recall.csv](./rf_base_class_recall.csv):

| Class | Recall | Test support | การอ่านผล |
|---|---:|---:|---|
| Normal | 0.9821 | 378,486 | แยก traffic ปกติได้ดี |
| Generic | 0.8705 | 9,302 | ตรวจจับได้ดี |
| Exploits | 0.8249 | 9,038 | ตรวจจับได้ค่อนข้างดี |
| Reconnaissance | 0.8145 | 4,431 | อยู่ในระดับดี |
| Shellcode | 0.7955 | 494 | อยู่ในระดับดีแต่ sample ไม่มาก |
| Fuzzers | 0.7582 | 6,899 | ยังมีการสับสนกับ class อื่น |
| Analysis | 0.5481 | 697 | ยังต้องปรับปรุง |
| Worms | 0.5179 | 56 | ไม่แน่นอนสูงเพราะ support น้อยมาก |
| Backdoor | 0.3797 | 690 | ตรวจจับได้ไม่ดี |
| DoS | 0.1958 | 1,808 | เป็น class ที่อ่อนที่สุด |

ภาพรวมจึงไม่ได้แปลว่าโมเดลตรวจจับทุก attack ได้ดีเท่ากัน แม้ Normal และบาง attack
class จะมี Recall สูง แต่ DoS, Backdoor, Analysis และ Worms ยังเป็นจุดอ่อนสำคัญ

## 11. Confusion matrix และสิ่งที่โมเดลเรียนรู้ได้

Confusion matrix อยู่ที่ [rf_base_confusion_matrix.csv](./rf_base_confusion_matrix.csv)
และช่วยดูว่าความผิดพลาดเกิดขึ้นระหว่าง class ใด ไม่ใช่ดูแค่คะแนนรวม

ประเด็นที่เห็นจากผลราย class:

- Normal ถูกจำแนกได้ดีมาก แต่ยังมีบางส่วนถูกทายเป็น Fuzzers
- DoS สับสนกับ Exploits, Analysis และ Backdoor จำนวนมาก
- Backdoor มีการสับสนกับ Analysis อย่างชัดเจน
- Worms มีตัวอย่างน้อยมาก จึงไม่ควรสรุปความสามารถจาก Recall เพียงค่าเดียว
- attack class ที่มีพฤติกรรมใกล้กันอาจแยกได้ยาก แม้ใช้ class weight แล้ว

## 12. ข้อสรุปการตัดสินใจ

ให้ใช้ **RF-base** เป็นโมเดลหลักสำหรับสื่อความสามารถของ traffic behavior features
เพราะ:

1. ไม่พึ่งพา TTL ที่อาจเป็น lab leakage
2. ไม่พึ่งพา IP-derived Geo ที่ไม่ให้ประโยชน์ชัดเจน
3. ใช้กติกา time split และ test อย่างถูกต้อง
4. ให้ผลรวมที่วัดได้และเปรียบเทียบกับโมเดลของสมาชิกคนอื่นได้

RF-base + TTL ไม่ถูกเลือกเป็นโมเดลหลัก แม้คะแนนดีกว่า เพราะคะแนนอาจสะท้อน
ลักษณะเฉพาะของ dataset มากกว่าความสามารถทั่วไป ส่วน RF-base + Geo ไม่ถูกเพิ่ม
เพราะผลดีขึ้นน้อยมากและ PR-AUC ลดลง

## 13. ข้อจำกัดของผลลัพธ์

- Test เป็นช่วงเวลาถัดไปจาก dataset เดียวกัน จึงเป็น temporal generalization
  ภายใน dataset นี้ ไม่ใช่หลักฐานว่าใช้กับเครือข่ายจริงใหม่ได้แน่นอน
- Class distribution ไม่สมดุลมาก โดยเฉพาะ Worms ที่มี test support เพียง 56 แถว
- DoS และ Backdoor ยังมี Recall ต่ำ จึงไม่ควรสรุปว่าโมเดลพร้อมใช้งานจริง
- ไม่มีการสุ่มในผล official ล่าสุด เพราะต้องการรักษาการประเมินบนข้อมูลเต็ม
- โหมด sampled training ประมาณ 100,000 แถวมีไว้เป็นทางเลือกสำรองในโค้ดเท่านั้น
  ไม่ได้ใช้สร้างตัวเลขในรายงานฉบับนี้
- ผล TTL ที่สูงขึ้นควรถูกมองเป็นสัญญาณให้ตรวจสอบ leakage ไม่ใช่หลักฐานว่า TTL
  เป็น feature ที่ดีที่สุด

## 14. สรุป

> งานของเราคือจำแนก network traffic ออกเป็น 10 classes โดยใช้ Random Forest
> เราแบ่งข้อมูลตามเวลา 70/10/20 ด้วย seed 42 เพื่อให้ train เป็นข้อมูลอดีตและ
> test เป็นข้อมูลช่วงเวลาถัดไป เนื่องจากข้อมูลไม่สมดุล เราใช้ Macro F1 เป็น metric
> หลัก และใช้ PR-AUC กับ Macro Recall เป็นหลักฐานประกอบ
>
> เราจูน hyperparameter บน validation ทั้งหมด 4 configurations และได้ชุดที่ดีที่สุด
> คือ 30 trees, max depth ไม่จำกัด, min samples leaf เท่ากับ 2 และใช้ sqrt features
> จากนั้นรวม train กับ validation แล้ววัด test เพียงครั้งเดียว ผล RF-base ได้
> Macro F1 0.5865, PR-AUC 0.6021 และ Macro Recall 0.6687
>
> เราทำ ablation เพิ่ม TTL และ Geo เพื่อดู leakage กับ shortcut การเพิ่ม TTL ทำให้
> คะแนนดีขึ้น แต่มีความเสี่ยงว่าโมเดลกำลังใช้ลักษณะเฉพาะของ lab จึงไม่เลือก TTL
> เป็นโมเดลหลัก ส่วน Geo แทบไม่ช่วยและทำให้ PR-AUC ลดลง สรุปจึงเลือก RF-base
> เป็น baseline หลัก โดยต้องระวังว่า DoS, Backdoor และ Worms ยังเป็น class ที่โมเดล
> ตรวจจับได้ไม่ดี

## 15. คำถามที่อาจถูกถาม

### ทำไมไม่ใช้ Accuracy?

เพราะ Normal มีจำนวนมากมาก Accuracy อาจสูงจากการทาย Normal เป็นส่วนใหญ่ แต่ไม่
สะท้อนคุณภาพของ attack class ที่มีจำนวนน้อย จึงใช้ Macro F1 เป็น metric หลัก

### ทำไมไม่เลือก TTL ที่คะแนนสูงกว่า?

เพราะคะแนนที่สูงขึ้นอาจเกิดจากข้อมูลเฉพาะของแลป ไม่ใช่พฤติกรรมโจมตีที่ generalize
ไปยังเครือข่ายใหม่ จึงใช้ TTL เพื่อวัดความเสี่ยง leakage เท่านั้น

### ทำไมไม่สุ่มเหลือ 100,000 แถว?

ผล official ใช้ข้อมูลเต็มทั้งหมด เพราะการสุ่มอาจทำให้ class เล็กมีจำนวนน้อยเกินไป
และจะตอบคนละคำถามกับการประเมินโมเดลบนข้อมูลเต็ม โค้ดมี sampled mode เป็นทางเลือก
สำรอง แต่ไม่ได้ใช้สร้างผลสรุปนี้


## 16. ไฟล์ผลลัพธ์ที่เกี่ยวข้อง

โค้ดและไฟล์ผลลัพธ์:

- [random_forest.py](../progress4/models/random_forest.py): โค้ดทดลองและคอมเมนต์ภาษาไทย
- [results_rf.csv](./results_rf.csv): ผล RF-base บน validation และ test
- [results_rf_ttl.csv](./results_rf_ttl.csv): ผล RF-base + TTL
- [results_rf_geo.csv](./results_rf_geo.csv): ผล RF-base + Geo
- [rf_base_class_recall.csv](./rf_base_class_recall.csv): Recall ราย class ของ RF-base
- [rf_base_confusion_matrix.csv](./rf_base_confusion_matrix.csv): confusion matrix ของ RF-base
- [rf_ttl_class_recall.csv](./rf_ttl_class_recall.csv): Recall ราย class ของ TTL
- [rf_ttl_confusion_matrix.csv](./rf_ttl_confusion_matrix.csv): confusion matrix ของ TTL
- [rf_geo_class_recall.csv](./rf_geo_class_recall.csv): Recall ราย class ของ Geo
- [rf_geo_confusion_matrix.csv](./rf_geo_confusion_matrix.csv): confusion matrix ของ Geo

## 17. กราฟ

กราฟทั้งหมดสร้างจาก CSV ผลการทดลองล่าสุดโดยไม่ train โมเดลหรือวัด test เพิ่ม:

- [rf_tuning_validation.png](./figures/rf_tuning_validation.png): เปรียบเทียบ 4 configurations บน validation
- [rf_metrics_comparison.png](./figures/rf_metrics_comparison.png): เปรียบเทียบ RF-base, TTL และ Geo บน test
- [rf_ablation_gap.png](./figures/rf_ablation_gap.png): แสดงคะแนนที่เพิ่มหรือลดจาก RF-base
- [rf_class_recall.png](./figures/rf_class_recall.png): Recall ราย class พร้อมจำนวนตัวอย่างใน test
- [rf_confusion_matrix.png](./figures/rf_confusion_matrix.png): confusion matrix แบบ normalize ต่อแถว
- [rf_confusion_matrix_counts.png](./figures/rf_confusion_matrix_counts.png):

ไฟล์ [split.py](../progress4/split.py) และ [metrics.py](../progress4/metrics.py)

