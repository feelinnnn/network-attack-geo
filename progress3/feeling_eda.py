"""Progress 3 — EDA (Exploratory Data Analysis)

Input : data/processed/features.parquet  (ผลจาก Progress 2)
        data/staging/unsw_flows.parquet   (ใช้เทียบค่าว่างก่อน clean)
        reports/ip_lookup.csv (พิกัด / ASN จาก ProdIPData)
        reports/cleaning_report.csv, reports/label_conflicts.csv
Output: reports/figures/<ชื่อ>_*.png    (กราฟ ขึ้นต้นด้วยชื่อตัวเองจะได้ไม่ซ้ำ เช่น feeling_class_count.png)
        reports/eda_<ชื่อ>.md           (สรุปข้อค้นพบของตัวเอง)

ทำอะไรก็ได้ ขั้นต่ำคนละ 5 อย่าง
แต่ละอย่างต้องตอบได้ว่า "เห็นอะไร -> เลยตัดสินใจ / เข้าใจอะไร"

ของ feeling: 7 หัวข้อ (UNSW + ProdIPData ที่ join แล้วใน features.parquet)
  เจาะลึกเฉพาะโปรเจกต์  1. เวลา   2. ประเทศ + เครื่องผู้โจมตี
  รวม 2 ชุดข้อมูล       3. แผนที่การโจมตี (พิกัด)   4. โปรไฟล์เครือข่ายต้นทาง (ASN)
  เตรียมเทรนโมเดล       5. คลาสไม่สมดุล + เพดานคะแนน   6. ทางลัด (leakage)
  ภาพรวม               7. คุณภาพข้อมูล

ติดตั้งเพิ่ม: pip install matplotlib scikit-learn
รัน (จาก root ของ repo):  python progress3/feeling_eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # บันทึกเป็นไฟล์อย่างเดียว ไม่เปิดหน้าต่าง
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import utils.py ที่ root ได้
from utils import load_config, p  # load_config = อ่าน config.yaml, p = แปลง path ให้อ้างอิงจาก root ของ repo

NAME = "feeling"            # ชื่อนำหน้าไฟล์ output (เปลี่ยนเป็นชื่อตัวเอง)
TZ = "Australia/Sydney"     # แล็บ UNSW อยู่ซิดนีย์ -> ใช้เวลาท้องถิ่นของแล็บ
GAP_MIN = 10                # เว้นว่างเกิน 10 นาที = เริ่มช่วงการโจมตี (campaign) ใหม่
SAMPLE_PER_CLASS = 20_000   # สุ่มต่อคลาสสำหรับการทดสอบที่ช้า (สถิติ, โมเดลเล็ก)

ATTACKS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
           "Analysis", "Backdoor", "Shellcode", "Worms"]  # เรียงจากมากไปน้อย
CLASSES = ["Normal"] + ATTACKS

FIG: Path = None             # โฟลเดอร์กราฟ (ตั้งค่าใน main)
REPORT: list[str] = []      # ข้อความสรุป -> eda_<ชื่อ>.md


def save(fig, name: str) -> None:  # บันทึกกราฟเป็น reports/figures/<ชื่อ>_<name>.png
    path = FIG / f"{NAME}_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def note(title: str, found: list[str], decide: list[str]) -> None:  # เก็บข้อค้นพบ "เห็นอะไร -> ตัดสินใจอะไร"
    REPORT.append(f"## {title}\n\n**เห็นอะไร**\n" + "".join(f"- {x}\n" for x in found)
                  + "\n**ตัดสินใจ / เข้าใจอะไร**\n" + "".join(f"- {x}\n" for x in decide))


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:  # สุ่มคลาสละไม่เกิน n แถว (คลาสเล็กเอาทั้งหมด)
    idx = [g.sample(min(len(g), n), random_state=seed).index for _, g in df.groupby("attack_cat")]
    return df.loc[np.concatenate(idx)]


# 1. เวลา
def eda_1_time(df: pd.DataFrame) -> None:
    print("[1] time")
    t = pd.to_datetime(df["stime"], unit="s", utc=True).dt.tz_convert(TZ)  # stime (วินาทีนับจาก 1970) -> วันที่-เวลาซิดนีย์
    gap = t.sort_values().diff() > pd.Timedelta(hours=6)  # เรียงเวลา แถวที่ห่างจากแถวก่อนเกิน 6 ชม. = True (จุดเริ่มรอบใหม่)
    session = gap.cumsum().reindex(t.index) + 1  # ช่วงเก็บข้อมูล (ห่างกันเกิน 6 ชม. = คนละรอบ)
    a = df.assign(t=t, session=session)[df["label"] == 1]  # เก็บเฉพาะการโจมตี + คอลัมน์เวลา/รอบ ใช้ทั้งหัวข้อ

    # กราฟ feeling_1_timeline: การโจมตีต่อ 10 นาที แยกประเภท แยกรอบเก็บข้อมูล
    #   แกน X = เวลา (ขีดเล็ก 1 ช่อง = 10 นาที), แกน Y = จำนวนการโจมตีใน 10 นาที, สี = ประเภท (ซ้อนกัน ขอบบนสุด = รวม)
    sessions = sorted(a["session"].unique())
    fig, axes = plt.subplots(len(sessions), 1, figsize=(13, 3.2 * len(sessions)))  # 1 แผงต่อ 1 รอบ
    for ax, sid in zip(np.atleast_1d(axes), sessions):
        g = a[a["session"] == sid]
        cnt = g.groupby([g["t"].dt.floor("10min"), "attack_cat"]).size().unstack(fill_value=0)  # ปัดเวลาลงเป็นช่อง 10 นาที แล้วนับ -> แถว = ช่องเวลา, คอลัมน์ = ประเภท
        cnt = cnt.reindex(columns=ATTACKS, fill_value=0)  # บังคับลำดับคอลัมน์ สีของแต่ละประเภทจะคงที่
        ax.stackplot(cnt.index, cnt.T.values, labels=ATTACKS, alpha=0.85)  # พื้นที่ซ้อน: ความหนาแต่ละสี = จำนวนประเภทนั้น
        ax.set_title(f"Session {sid}: {g['t'].min():%d %b %H:%M} – {g['t'].max():%d %b %H:%M} ({TZ})")
        ax.set_ylabel("attacks / 10 min")
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 10), tz=TZ))  # ขีดเล็กทุก 10 นาที = 1 ช่องของข้อมูล
        if g["t"].max() - g["t"].min() <= pd.Timedelta(hours=3):  # รอบสั้น -> ตัวเลขทุก 10 นาที
            ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10), tz=TZ))
        else:  # รอบยาว (12 ชม.) ใส่ตัวเลขทุก 10 นาทีจะซ้อนกัน -> ตัวเลขทุก 1 ชม. แต่ขีดเล็กยังทุก 10 นาที
            ax.xaxis.set_major_locator(mdates.HourLocator(tz=TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))  # วันที่อยู่ในหัวกราฟแล้ว
        ax.set_xlim(cnt.index.min(), cnt.index.max())
    np.atleast_1d(axes)[0].legend(ncol=5, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "1_timeline")

    # กราฟ feeling_1_hour_heatmap: แต่ละประเภทกระจุกช่วงไหนของวัน
    #   แกน X = ชั่วโมง 0-23, แกน Y = ประเภท, สีในช่อง = % ของประเภทนั้นที่เกิดในชั่วโมงนั้น (แต่ละแถวรวม 100% อ่านทีละแถว)
    hm = pd.crosstab(a["attack_cat"], a["t"].dt.hour, normalize="index").reindex(ATTACKS) * 100  # นับ ประเภท x ชั่วโมง แล้วทำเป็น % ต่อแถว
    hm = hm.reindex(columns=range(24), fill_value=0)  # เติมชั่วโมงที่ไม่มีการโจมตีเป็น 0 ให้ครบ 24 ช่อง
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(hm, aspect="auto", cmap="magma_r")  # วาดตารางเป็นสี: เหลืองอ่อน = 0, ม่วงดำ = มาก
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(ATTACKS)), ATTACKS)
    ax.set_xlabel(f"hour of day ({TZ})")
    fig.colorbar(im, ax=ax, label="% of that class")
    ax.set_title("When does each attack type happen?")
    save(fig, "1_hour_heatmap")

    # campaign: เหตุการณ์ประเภทเดียวกันที่ห่างกันไม่เกิน GAP_MIN นาที = ช่วงการโจมตีเดียวกัน
    camps = []
    for cat, g in a.sort_values("t").groupby("attack_cat"):
        cid = (g["t"].diff() > pd.Timedelta(minutes=GAP_MIN)).cumsum()  # เว้นว่างเกิน 10 นาที = campaign ใหม่ -> เลข campaign ของแต่ละแถว
        c = g.groupby(cid)["t"].agg(["min", "max", "size"])  # 1 แถวต่อ campaign: เวลาเริ่ม / จบ / จำนวนแถว
        camps.append(c.assign(attack_cat=cat))
    camps = pd.concat(camps, ignore_index=True)
    camps["minutes"] = (camps["max"] - camps["min"]).dt.total_seconds() / 60
    camp_sum = camps.groupby("attack_cat").agg(campaigns=("size", "size"), median_minutes=("minutes", "median"),
                                               longest_minutes=("minutes", "max"), rows=("size", "sum"))
    camp_sum = camp_sum.reindex(ATTACKS).round(1)
    camp_sum.to_csv(FIG.parent / f"eda_{NAME}_campaigns.csv")

    # กราฟ feeling_1_sequence: campaign ถัดไปที่เริ่ม (ต่างประเภท) คืออะไร -> เช่น หลัง Reconnaissance มักตามด้วยอะไร
    #   แกน Y = ประเภทปัจจุบัน, แกน X = ประเภทถัดไป, สีในช่อง = % (แต่ละแถวรวม 100%) น้ำเงินเข้ม = ตามกันบ่อย
    seq = camps.sort_values("min")["attack_cat"]  # เรียง campaign ทุกประเภทตามเวลาเริ่ม
    seq = seq[seq != seq.shift()]  # ตัดตัวที่ซ้ำติดกัน (Exploits, Exploits -> Exploits)
    trans = pd.crosstab(seq.values[:-1], seq.values[1:], normalize="index").reindex(index=ATTACKS, columns=ATTACKS,  # จับคู่ ตัวนี้ -> ตัวถัดไป แล้วนับเป็น % ต่อแถว
                                                                                    fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(trans, cmap="Blues")
    ax.set_xticks(range(9), ATTACKS, rotation=45, ha="right")
    ax.set_yticks(range(9), ATTACKS)
    ax.set_xlabel("next attack type")
    ax.set_ylabel("current attack type")
    fig.colorbar(im, ax=ax, label="%")
    ax.set_title("What attack type starts next?")
    save(fig, "1_sequence")

    share = a["session"].value_counts(normalize=True).sort_index() * 100
    rate = df.assign(session=session).groupby("session")["label"].mean() * 100  # % การโจมตีในแต่ละรอบ
    mix = pd.crosstab(a["session"], a["attack_cat"], normalize="index") * 100  # สัดส่วนประเภทในแต่ละรอบ
    mix_diff = mix.max().sub(mix.min()).max()
    peak = hm.idxmax(axis=1)
    quiet = [h for h in range(24) if hm[h].sum() == 0]  # ชั่วโมงที่ไม่มีการโจมตีเลย
    after_recon = trans.loc["Reconnaissance"].idxmax()
    note("1. การโจมตีตามเวลา",
         [f"มี {len(sessions)} รอบเก็บข้อมูล การโจมตีอยู่ในรอบ " + ", ".join(
             f"{sid}: {v:.0f}%" for sid, v in share.items()),
          (f"ไม่มีการโจมตีเลยช่วง {quiet[0]:02d}:00–{quiet[-1]:02d}:59 (มีแต่ทราฟฟิกปกติ)" if quiet else "มีการโจมตีทุกชั่วโมง"),
          "ชั่วโมงที่แต่ละประเภทเกิดมากสุด: " + ", ".join(f"{c} {h}:00" for c, h in peak.items()),
          "ช่วงการโจมตี (campaign) ยาวเฉลี่ย (median): " + ", ".join(
              f"{c} {m:.0f} นาที" for c, m in camp_sum["median_minutes"].items()),
          f"หลัง Reconnaissance ประเภทที่เริ่มต่อบ่อยสุดคือ {after_recon} "
          f"({trans.loc['Reconnaissance', after_recon]:.0f}%) (หลายประเภทรันพร้อมกัน ใช้ดูแนวโน้มเท่านั้น)",
          "สัดส่วนการโจมตีในแต่ละรอบ: " + ", ".join(f"รอบ {k} {v:.1f}%" for k, v in rate.items())
          + f" แต่สัดส่วนประเภทภายในการโจมตีต่างกันไม่เกิน {mix_diff:.1f} จุด %"],
         ["drift อยู่ที่ 'ปริมาณการโจมตี' ไม่ใช่ 'ประเภท' -> โมเดลที่เรียนจากรอบหนึ่งจะเจออัตราการโจมตีต่างไปในอีกรอบ (หลักฐาน H2)",
          "ต้องแบ่ง train / val / test ตามเวลา และเช็กว่าทุกชุดมีครบ 10 คลาส",
          "Analysis / Worms มาเป็นช่วงสั้นๆ หลายครั้ง ส่วนประเภทอื่นรันยาวต่อเนื่อง ~1 ชม. พร้อมกัน",
          "ข้อจำกัด: มีแค่ 2 รอบ และการโจมตีถูกปล่อยด้วยเครื่องมือในแล็บ "
          "ชั่วโมงที่เกิดจึงสะท้อนตารางการทดลอง ไม่ใช่พฤติกรรมแฮกเกอร์จริง"])


# 2. ประเทศ + เครื่องผู้โจมตี
def eda_2_country(df: pd.DataFrame) -> None:
    print("[2] country + attacker hosts")
    # กราฟ feeling_2_country: แต่ละประเทศต้นทางส่งอะไรมา
    #   แกน Y = ประเทศต้นทาง, แกน X = % ของแถวจากประเทศนั้น (ทุกแท่ง 100%), สี = คลาส, ตัวเลขท้ายแท่ง = จำนวนแถวจริง
    ct = pd.crosstab(df["src_country"], df["attack_cat"]).reindex(columns=CLASSES, fill_value=0)  # นับแถว ประเทศ x คลาส
    fig, ax = plt.subplots(figsize=(9, 3.5))
    (ct.div(ct.sum(axis=1), axis=0) * 100).plot.barh(stacked=True, ax=ax, colormap="tab10", width=0.8)  # หารผลรวมแถว -> % แล้ววาดแท่งซ้อน
    for i, total in enumerate(ct.sum(axis=1)):
        ax.text(101, i, f"{total:,} rows", va="center", fontsize=8)  # เขียนจำนวนจริงท้ายแท่ง (แท่ง % ไม่บอกขนาด)
    ax.set_xlabel("% of rows from this source")
    ax.set_title("What comes from each source country?")
    ax.legend(bbox_to_anchor=(1.18, 1), fontsize=7)
    save(fig, "2_country")

    # กราฟ feeling_2_attacker_hosts: ลงระดับเครื่อง KP มีหลาย IP แต่ละเครื่องแบ่งหน้าที่กันไหม
    #   แกน Y = IP เครื่องผู้โจมตี, แกน X = ประเภท, ตัวเลข/สีในช่อง = % ของการโจมตีของเครื่องนั้น (แต่ละแถวรวม 100%)
    kp = df[(df["src_country"] == "KP") & (df["label"] == 1)]  # เฉพาะการโจมตีจาก KP
    host = pd.crosstab(kp["srcip"], kp["attack_cat"], normalize="index").reindex(columns=ATTACKS) * 100  # IP x ประเภท เป็น % ต่อเครื่อง
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(host, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(9), ATTACKS, rotation=30, ha="right")
    ax.set_yticks(range(len(host)), host.index)
    for (i, j), v in np.ndenumerate(host.values):  # วนทุกช่อง (แถว i, คอลัมน์ j) แล้วเขียนตัวเลขลงในช่อง
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of host's attacks")
    ax.set_title("Attack mix per attacker host (KP)")
    save(fig, "2_attacker_hosts")
    diff = host.max().sub(host.min())  # ห่างกันมากสุดระหว่างเครื่อง (จุด %) ต่อประเภท
    spread, spread_cls = diff.max(), diff.idxmax()
    odd_host = host[spread_cls].idxmin()  # เครื่องที่ทำประเภทนั้นน้อยสุด
    odd_main = host.loc[odd_host].idxmax()  # แล้วเครื่องนั้นเน้นอะไรแทน

    # ทราฟฟิกปกติจาก KP ต่างจากการโจมตีของ KP ยังไง
    kp_all = df[df["src_country"] == "KP"]
    cmp = kp_all.groupby("label")[["sbytes", "dbytes", "dur", "spkts", "sttl"]].median().T  # median ต่อ label 0/1 แล้วกลับตาราง
    cmp.columns = ["KP normal (median)", "KP attack (median)"]
    cmp.to_csv(FIG.parent / f"eda_{NAME}_kp_normal_vs_attack.csv")

    attack_src = df.loc[df["label"] == 1, "src_country"].value_counts(normalize=True) * 100
    attack_dst = df.loc[df["label"] == 1, "dst_country"].value_counts(normalize=True) * 100
    note("2. ประเทศ + เครื่องผู้โจมตี",
         [f"การโจมตีมาจาก: " + ", ".join(f"{k} {v:.1f}%" for k, v in attack_src.items()),
          f"การโจมตีไปที่: " + ", ".join(f"{k} {v:.1f}%" for k, v in attack_dst.head(3).items()),
          f"KP ยังมีทราฟฟิกปกติ {int((kp_all['label'] == 0).sum()):,} แถว",
          f"เครื่องผู้โจมตี {len(host)} เครื่องทำครบทุกประเภท ส่วนใหญ่สัดส่วนใกล้กัน ยกเว้น {odd_host} "
          f"ทำ {spread_cls} แค่ {host.loc[odd_host, spread_cls]:.0f}% แต่เน้น {odd_main} "
          f"{host.loc[odd_host, odd_main]:.0f}% แทน -> แทบไม่ได้แบ่งหน้าที่ตามเครื่อง",
          "ทราฟฟิกปกติของ KP หน้าตาใกล้การโจมตี (median sbytes / dbytes / spkts / sttl ใกล้กัน ดู eda_*_kp_normal_vs_attack.csv)"],
         ["ประเทศบอกได้ว่า 'โจมตีหรือไม่' ทันที (ถ้าไม่ใช่ KP = ปกติ) แต่แยก 'ประเภท' ไม่ได้ "
          "เพราะทุกประเภทมาจาก KP เหมือนกัน -> H1 คาดว่าใส่ประเทศแล้วช่วยแยกโจมตี/ปกติ แต่ไม่ช่วยแยกประเภท",
          "ประเทศสะท้อนบทบาทในแล็บ ไม่ใช่ตำแหน่งผู้โจมตีจริง -> โมเดลหลักไม่ใช้ geo",
          "srcip ต้องอยู่ใน drop (ถ้าใส่ โมเดลจะจำ IP แทนการเรียนพฤติกรรม)"])


# 3. แผนที่การโจมตี (UNSW + ProdIPData: พิกัด)
def eda_3_map(df: pd.DataFrame, lk: pd.DataFrame) -> None:
    print("[3] attack map")
    # กราฟ feeling_3_map: ทราฟฟิกวิ่งจากประเทศไหนไปไหน
    #   แกน X = ลองจิจูด, แกน Y = ละติจูด, จุดดำ = ประเทศ, ลูกศร = ต้นทาง -> ปลายทาง (แดง = โจมตี, น้ำเงิน = ปกติ, หนา = แถวมาก)
    geo = (lk[lk["status"] == "FOUND"].groupby("country")
             .agg(lat=("Latitude", "first"), lon=("Longitude", "first"), city=("CityName", "first")))  # 1 จุดต่อประเทศ
    flows = (df[df["src_country"].isin(geo.index) & df["dst_country"].isin(geo.index)]
               .groupby(["src_country", "dst_country", "label"]).size().rename("rows").reset_index())
    flows = flows[flows["src_country"] != flows["dst_country"]]  # ตัดเส้นที่ต้นทาง = ปลายทาง (วาดลูกศรไม่ได้)
    mix = (df[df["label"] == 1].groupby(["src_country", "dst_country", "attack_cat"]).size()
             .rename("rows").reset_index())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for lon in range(-180, 181, 30):  # เส้นตารางทุก 30 องศา แทนแผนที่ (ไม่ต้องติดตั้งไลบรารีแผนที่)
        ax.axvline(lon, color="#eeeeee", lw=0.6, zorder=0)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color="#eeeeee", lw=0.6, zorder=0)
    top = flows["rows"].max()
    for r in flows.itertuples():
        a, b = geo.loc[r.src_country], geo.loc[r.dst_country]
        color = "#c44e52" if r.label == 1 else "#4c72b0"  # แดง = โจมตี, น้ำเงิน = ปกติ
        bend = 0.15 if r.label == 1 else -0.15  # โค้งคนละทาง ไม่ให้เส้นปกติกับเส้นโจมตีทับกัน
        ax.annotate("", xy=(b.lon, b.lat), xytext=(a.lon, a.lat),  # ลูกศรจากต้นทาง (xytext) ไปปลายทาง (xy)
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.75,
                                    lw=1 + 8 * np.sqrt(r.rows / top), connectionstyle=f"arc3,rad={bend}"))  # ความหนาตาม √แถว (ไม่ให้เส้นเล็กหายไป)
    for c, g in geo.iterrows():
        ax.scatter(g.lon, g.lat, s=120, color="k", zorder=3)
        ax.text(g.lon + 3, g.lat + 3, f"{c} ({g.city})", fontsize=10, weight="bold")
    ax.plot([], [], color="#4c72b0", lw=4, label="normal")
    ax.plot([], [], color="#c44e52", lw=4, label="attack")
    ax.legend(loc="lower left")
    ax.set_xlim(-130, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Traffic flows between source and destination (line width = rows, ProdIPData coordinates)")
    save(fig, "3_map")

    # กราฟ feeling_3_routes: เส้นทางไหนมีการโจมตีประเภทอะไร
    #   แกน Y = เส้นทาง ต้นทาง -> ปลายทาง, แกน X = % ของการโจมตีบนเส้นนั้น, สี = ประเภท, ตัวเลขท้ายแท่ง = จำนวนจริง
    route = mix.assign(route=mix["src_country"] + " → " + mix["dst_country"])
    route = route.pivot_table(index="route", columns="attack_cat", values="rows", fill_value=0)  # ตาราง เส้นทาง x ประเภท
    route = route.reindex(columns=ATTACKS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(route)))
    (route.div(route.sum(axis=1), axis=0) * 100).plot.barh(stacked=True, ax=ax, colormap="tab10", width=0.7)
    for i, total in enumerate(route.sum(axis=1)):
        ax.text(101, i, f"{int(total):,}", va="center", fontsize=8)
    ax.set_xlabel("% of attacks on this route")
    ax.set_title("Attack types per route")
    ax.legend(bbox_to_anchor=(1.12, 1), fontsize=7)
    save(fig, "3_routes")

    att = flows[flows["label"] == 1].sort_values("rows", ascending=False)
    nor = flows[flows["label"] == 0].sort_values("rows", ascending=False)
    dist = lambda a, b: np.hypot(geo.loc[a, "lat"] - geo.loc[b, "lat"], geo.loc[a, "lon"] - geo.loc[b, "lon"])
    note("3. แผนที่การโจมตี",
         ["เส้นทางการโจมตี: " + ", ".join(f"{r.src_country}→{r.dst_country} {r.rows:,} แถว" for r in att.itertuples()),
          "เส้นทางทราฟฟิกปกติ (ข้ามประเทศ): " + ", ".join(
              f"{r.src_country}→{r.dst_country} {r.rows:,} แถว" for r in nor.head(4).itertuples()),
          "ตำแหน่งจาก ProdIPData: " + ", ".join(f"{c} = {g.city}" for c, g in geo.iterrows()),
          f"มีเส้นทางการโจมตี {len(att)} เส้น ทุกประเภทใช้เส้นทางเดียวกัน"],
         ["ภาพนี้ใช้เล่าเรื่องในสไลด์และ dashboard ได้ (ใคร → ใคร มากแค่ไหน)",
          "แต่ภูมิศาสตร์ไม่มีความหลากหลาย (การโจมตีมาจาก 1 จุด) จึงใช้แยก 'ประเภท' ไม่ได้ -> สอดคล้องกับ H1",
          "ข้อจำกัด: พิกัดมาจาก ProdIPData ปี 2026 ส่วนทราฟฟิกเป็นปี 2015 และ IP ในแล็บถูกตั้งตามบทบาท "
          "แผนที่จึงแสดง 'โครงสร้างของแล็บ' ไม่ใช่ที่อยู่จริงของผู้โจมตี"])


# 4. โปรไฟล์เครือข่ายต้นทาง (UNSW + ProdIPData: ASN)
def eda_4_network(df: pd.DataFrame, lk: pd.DataFrame) -> None:
    print("[4] network profile")
    info = lk.set_index("ip")[["country", "asn", "ASNName", "ASNOrg", "ASNType"]]
    src = df[["srcip", "label", "attack_cat"]].join(info, on="srcip")  # ต่อข้อมูลเครือข่ายของ IP ต้นทางเข้าทุกแถว (เลือกแค่ 3 คอลัมน์ ประหยัด RAM)
    src["network"] = src["ASNName"].fillna(src["country"])  # IP private/loopback ไม่มีชื่อเครือข่าย -> ใช้ชื่อประเภทแทน
    src["ASNType"] = src["ASNType"].fillna("private/loopback")  # ไม่ใช่ IP บนอินเทอร์เน็ต ไม่มีประเภทเครือข่าย

    prof = src.groupby("network").agg(country=("country", "first"), asn_type=("ASNType", "first"),
                                      rows=("label", "size"), attacks=("label", "sum"))
    prof["attack_rate_%"] = (100 * prof["attacks"] / prof["rows"]).round(2)
    prof["org"] = info.drop_duplicates("ASNName").set_index("ASNName")["ASNOrg"].reindex(prof.index)
    prof = prof.sort_values("rows", ascending=False)
    prof.to_csv(FIG.parent / f"eda_{NAME}_network_profile.csv")

    typ = src.groupby("ASNType")["label"].agg(["size", "mean"])  # อัตราการโจมตีตามประเภทเครือข่าย

    # กราฟ feeling_4_network: 2 แผง แถวเดียวกัน (ชื่อเครือข่าย + ประเภทเครือข่าย)
    #   ซ้าย: แกน X = จำนวนแถวที่ส่ง (log, ขีดละ x10), น้ำเงิน = ปกติ, แดง = โจมตี | ขวา: แกน X = % ของแถวที่เป็นการโจมตี
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    y = np.arange(len(prof))
    labels = [f"{n} ({t})" for n, t in zip(prof.index, prof["asn_type"])]
    axes[0].barh(y - 0.2, prof["rows"] - prof["attacks"], height=0.4, color="#4c72b0", label="normal")  # เลื่อนขึ้น 0.2 วางคู่กับแท่งแดง
    axes[0].barh(y + 0.2, prof["attacks"].replace(0, np.nan), height=0.4, color="#c44e52", label="attack")  # 0 -> ไม่วาด (แกน log วาด 0 ไม่ได้)
    axes[0].set_xscale("log")  # 1 แถวกับ 1.9 ล้านแถวอยู่กราฟเดียวกันได้
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("rows sent (log)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Rows sent per source network (ASN type)")

    axes[1].barh(y, prof["attack_rate_%"], color="#c44e52")
    for i, v in enumerate(prof["attack_rate_%"]):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("% of rows that are attacks")
    axes[1].set_title("Attack rate per source network")
    fig.tight_layout()
    save(fig, "4_network")

    att = prof[prof["attacks"] > 0]
    pub = lk[lk["status"] == "FOUND"]
    one_to_one = (pub.groupby("country")["asn"].nunique() == 1).all() and (pub.groupby("asn")["country"].nunique() == 1).all()
    note("4. โปรไฟล์เครือข่ายต้นทาง",
         [f"มีเครือข่ายต้นทาง {len(prof)} กลุ่ม การโจมตีมาจาก {len(att)} เครือข่าย: "
          + ", ".join(f"{n} ({r.org}, {r.asn_type}) โจมตี {r.attack_rate_:.1f}% ของที่ส่ง"
                      for n, r in att.rename(columns={"attack_rate_%": "attack_rate_"}).iterrows()),
          "ทราฟฟิกปกติส่วนใหญ่มาจาก: " + ", ".join(
              f"{n} ({r.asn_type}) {r.rows:,} แถว" for n, r in prof[prof["attacks"] == 0].head(2).iterrows()),
          "อัตราการโจมตีตามประเภทเครือข่าย: " + ", ".join(
              f"{t} {m * 100:.1f}%" for t, m in typ["mean"].items())],
         [("ASN ให้ข้อมูลซ้ำกับประเทศ 100% (1 ประเทศ = 1 เครือข่ายใน UNSW) -> ไม่ต้องใส่ทั้ง country และ asn ในโมเดลทดลอง H1"
           if one_to_one else "ASN แยกย่อยกว่าประเทศ -> ลองใส่ asn แยกจาก country ในโมเดลทดลอง H1"),
          "ประเภทเครือข่าย (ISP / Business) อธิบายบทบาทในแล็บได้: ผู้ใช้ปกติและผู้โจมตีถูกจำลองเป็น ISP ส่วนเซิร์ฟเวอร์เป็นมหาวิทยาลัย",
          "ถ้าใช้ข้อมูลจริงที่มีหลายเครือข่าย ASN / ASNType จะเป็น feature ที่น่าสนใจกว่านี้ (ข้อเสนอแนะในรายงาน)"])


# 5. คลาสไม่สมดุล + เพดานคะแนน (univariate)
def eda_5_imbalance(df: pd.DataFrame, feat: dict) -> None:
    print("[5] class imbalance")
    # กราฟ feeling_5_class_count: จำนวนแถวต่อคลาส
    #   แกน X = 10 คลาส, แกน Y = จำนวนแถว (log, ขีดละ x10), น้ำเงิน = Normal, แดง = การโจมตี, ตัวเลขบนแท่ง = จำนวนจริง
    n = df["attack_cat"].value_counts().reindex(CLASSES)  # นับแถวต่อคลาส เรียง Normal ก่อน

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(n.index, n.values, color=["#4c72b0"] + ["#c44e52"] * 9)
    ax.set_yscale("log")  # Normal 1.9 ล้าน กับ Worms 171 อยู่กราฟเดียวกันได้
    for i, v in enumerate(n.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("rows (log)")
    ax.set_title("Rows per class")
    plt.xticks(rotation=30, ha="right")
    save(fig, "5_class_count")

    # เพดาน: ถ้าแถวหน้าตาเหมือนกันมีหลายคลาส ทายได้ดีที่สุดคือทายคลาสที่มากสุดในกลุ่ม -> โมเดลไหนก็เก่งเกินนี้ไม่ได้
    X = feat["base"] + feat["check"]
    key = pd.util.hash_pandas_object(df[X], index=False).values  # ลายนิ้วมือของแถว: 41 คอลัมน์ -> ตัวเลขเดียว
    cnt = pd.DataFrame({"key": key, "y": df["attack_cat"].values}).groupby(["key", "y"]).size()  # แต่ละลายนิ้วมือมีกี่แถวในแต่ละคลาส
    best = cnt.sort_values(ascending=False).reset_index().drop_duplicates("key").set_index("key")["y"]  # คลาสที่มากสุดของแต่ละลายนิ้วมือ
    pred = pd.Series(key).map(best).values  # ทุกแถวในกลุ่มเดียวกันได้คลาสที่มากสุดในกลุ่ม
    ceiling = f1_score(df["attack_cat"], pred, average="macro")
    per_cls = pd.Series(f1_score(df["attack_cat"], pred, average=None, labels=CLASSES), index=CLASSES)

    note("5. คลาสไม่สมดุล + เพดานคะแนน",
         [f"Normal {n['Normal'] / n.sum():.1%} ของข้อมูล คลาสเล็กสุด Worms {n['Worms']:,} แถว "
          f"(ห่างกัน {n['Normal'] / n['Worms']:,.0f} เท่า)",
          f"ทาย Normal ทุกแถวก็ได้ accuracy {n['Normal'] / n.sum():.1%}",
          f"เพดาน Macro F1 (แม้จำข้อมูลได้หมด) = {ceiling:.3f} คลาสที่เพดานต่ำ: "
          + ", ".join(f"{c} {v:.2f}" for c, v in per_cls.sort_values().head(3).items())],
         ["ใช้ Macro F1 เป็น metric หลัก ไม่ใช้ accuracy",
          "ใช้ class_weight='balanced' ทุกโมเดล",
          f"ตั้งความคาดหวัง: Macro F1 บน test จะต่ำกว่า {ceiling:.2f} แน่นอน"])


# 6. ทางลัด / leakage (เจาะลึก)
def eda_6_leakage(df: pd.DataFrame, feat: dict, seed: int) -> None:
    print("[6] leakage")
    s = stratified_sample(df, SAMPLE_PER_CLASS, seed)  # สุ่มคลาสละ <= 20,000 แถว (~105,000 แถว)

    # วัดพลังทำนายทีละคอลัมน์: ใช้คอลัมน์เดียวทาย attack_cat ได้ Macro F1 เท่าไร
    cols = feat["base"] + feat["check"] + feat["geo"]  # 45 คอลัมน์ (base 38 + TTL 3 + geo 4)
    X = s[cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes  # ข้อความ -> รหัสตัวเลข
    Xtr, Xte, ytr, yte = train_test_split(X, s["attack_cat"], test_size=0.3, random_state=seed,
                                          stratify=s["attack_cat"])

    def score(use: list[str], depth: int) -> float:
        m = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=seed)
        m.fit(Xtr[use], ytr)
        return f1_score(yte, m.predict(Xte[use]), average="macro")

    single = pd.Series({c: score([c], 6) for c in cols}).sort_values(ascending=False)  # ฝึก 45 ต้น ต้นละ 1 คอลัมน์
    groups = {"all base": score(feat["base"], 12),
              "base + check": score(feat["base"] + feat["check"], 12),
              "check only (TTL)": score(feat["check"], 12),
              "geo only": score(feat["geo"], 12)}
    single.rename("macro_f1").to_csv(FIG.parent / f"eda_{NAME}_single_feature.csv")

    # กราฟ feeling_6_single_feature: ใช้คอลัมน์เดียวทาย 10 คลาสได้ Macro F1 เท่าไร (15 อันดับแรก)
    #   แกน Y = คอลัมน์, แกน X = Macro F1, แดง = check / geo, น้ำเงิน = base, เส้นประ = ใช้ base ทั้งหมด (0.62)
    top = single.head(15)[::-1]  # [::-1] กลับลำดับ ให้อันดับ 1 อยู่บนสุด
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c44e52" if c in feat["check"] + feat["geo"] else "#4c72b0" for c in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(groups["all base"], ls="--", color="k", lw=1)  # เส้นอ้างอิง: แท่งไหนเกือบถึงเส้นนี้ = สงสัยเป็นทางลัด
    ax.text(groups["all base"], -0.9, " all base", fontsize=8)
    ax.set_xlabel("Macro F1 using this ONE column (red = check / geo)")
    ax.set_title("How much can a single column predict attack_cat?")
    save(fig, "6_single_feature")

    # กราฟ feeling_6_sttl: แต่ละคลาสใช้ค่า sttl อะไร
    #   แกน X = คลาส, แกน Y = % ของคลาส (ทุกแท่ง 100%), สี = ค่า sttl (254 = เครื่อง KP, 31 = เครื่อง JP)
    sttl = pd.crosstab(df["attack_cat"], df["sttl"], normalize="index").reindex(CLASSES) * 100  # คลาส x ค่า sttl เป็น % ต่อคลาส
    sttl = sttl[sttl.max().sort_values(ascending=False).index[:6]]  # เก็บแค่ 6 ค่าที่สัดส่วนสูงสุด (ค่าอื่นน้อยจนมองไม่เห็น)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sttl.plot.bar(stacked=True, ax=ax, colormap="Set2", width=0.8)
    ax.set_ylabel("% of class")
    ax.set_title("sttl values per class")
    ax.legend(title="sttl", bbox_to_anchor=(1, 1), fontsize=8)
    plt.xticks(rotation=30, ha="right")
    save(fig, "6_sttl")

    ttl_share = groups["check only (TTL)"] / groups["all base"] * 100
    ttl_by_src = df.groupby("src_country")["sttl"].agg(lambda x: x.value_counts().index[0])  # sttl ที่พบบ่อยสุดต่อประเทศต้นทาง
    ttl_purity = (df.groupby("src_country")["sttl"].agg(lambda x: x.value_counts(normalize=True).iloc[0]) * 100)
    note("6. ทางลัด (leakage)",
         ["ใช้คอลัมน์เดียวทายได้ดีสุด: " + ", ".join(f"{c} {v:.2f}" for c, v in single.head(5).items()),
          f"Macro F1 (ตัวอย่างสุ่ม คลาสละ ≤ {SAMPLE_PER_CLASS:,}): " + ", ".join(
              f"{k} {v:.2f}" for k, v in groups.items()),
          f"ใช้แค่ TTL 3 คอลัมน์ได้ {ttl_share:.0f}% ของคะแนน base ทั้งหมด",
          "sttl ที่พบบ่อยสุดตามประเทศต้นทาง: " + ", ".join(
              f"{k} {int(v)} ({ttl_purity[k]:.0f}%)" for k, v in ttl_by_src.items() if k in ("KP", "JP", "AU"))],
         ["sttl เป็น 'ลายนิ้วมือของเครื่องต้นทาง' (แต่ละประเทศ/เครื่องมีค่าประจำตัว) ทำหน้าที่เหมือน src_country "
          "-> ช่วยแยกโจมตี/ปกติได้ทันที แต่แยกประเภทได้น้อย เพราะการโจมตีทุกประเภทมาจากเครื่องชุดเดียวกัน",
          f"เพิ่ม TTL เข้า base ทำให้คะแนนขึ้นแค่ {groups['base + check'] - groups['all base']:.2f} "
          "-> เสนอทีมตัด TTL ไปไว้ใน drop (หรือทดสอบคู่กับ H1 แบบมี/ไม่มี) ใน Progress 4",
          "geo ใช้แยกประเภทได้น้อย (สอดคล้องกับหัวข้อ 2 และ 4) -> ทดสอบใน H1",
          "คะแนนนี้มาจากตัวอย่างแบ่งแบบสุ่ม ใช้จัดอันดับเท่านั้น ไม่ใช่คะแนนโมเดลจริง"])


# 7. คุณภาพข้อมูล (ภาพรวม)
def eda_7_quality(df: pd.DataFrame, cfg: dict, feat: dict) -> None:
    print("[7] data quality")
    raw = pd.read_parquet(p(cfg["paths"]["staging"]) / "unsw_flows.parquet")
    text = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    blank = raw[text].apply(lambda s: s.str.strip() == "").sum()  # ช่องว่าง " " นับเป็นค่าว่างด้วย (isna() หาไม่เจอ)
    before = raw.isna().sum().add(blank, fill_value=0)  # ค่าว่างก่อน clean = NaN + ช่องว่าง
    after = df[raw.columns].isna().sum()  # ค่าว่างหลัง clean (เฉพาะคอลัมน์ที่มีในข้อมูลดิบ)
    miss = pd.DataFrame({"before clean": before, "after clean": after})
    miss = miss[(miss > 0).any(axis=1)].sort_values("before clean")  # เก็บเฉพาะคอลัมน์ที่เคยว่างอย่างน้อยฝั่งหนึ่ง
    del raw

    # กราฟ feeling_7_missing: ค่าว่างก่อน vs หลัง clean
    #   แกน Y = คอลัมน์, แกน X = จำนวนแถวที่ว่าง (symlog: วาด 0 ได้, ขีดละ x10), แดง = ก่อน, เขียว = หลัง, ไม่มีแท่ง = 0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    miss.plot.barh(ax=ax, color=["#c44e52", "#55a868"])
    ax.set_xscale("symlog")  # log ที่วาดค่า 0 ได้
    ax.set_xlabel("missing rows (symlog)")
    ax.set_title("Missing values before vs after clean")
    save(fig, "7_missing")

    # แถวที่ feature เหมือนกันทุกตัว: แบ่งแบบสุ่มจะรั่วไปอยู่ทั้ง train และ test มากกว่าแบ่งตามเวลาแค่ไหน (หลักฐาน H3)
    X = feat["base"] + feat["check"]
    key = pd.Series(pd.util.hash_pandas_object(df[X], index=False).values, index=df.index)
    order = df["stime"].argsort(kind="stable").values
    cut = int(len(df) * 0.8)
    t_train, t_test = key.iloc[order[:cut]], key.iloc[order[cut:]]  # แบ่งตามเวลา: 80% แรก = train, 20% หลัง = test
    r_train, r_test = train_test_split(key, test_size=0.2, random_state=cfg["seed"])  # แบ่งสุ่ม 80/20
    leak_time = t_test.isin(set(t_train)).mean() * 100  # % แถว test ที่มีลายนิ้วมือเดียวกันอยู่ใน train
    leak_rand = r_test.isin(set(r_train)).mean() * 100

    conf = pd.read_csv(p(cfg["paths"]["reports"]) / "label_conflicts.csv", index_col=0)
    rep = pd.read_csv(p(cfg["paths"]["reports"]) / "cleaning_report.csv")
    n_dup = int(rep[rep["rule"].str.startswith("drop duplicates")]["rows"].sum())

    note("7. คุณภาพข้อมูล",
         [f"ค่าว่างก่อน clean อยู่แค่ {int((miss['before clean'] > 0).sum())} คอลัมน์ หลัง clean เหลือเฉพาะ port "
          f"({int(after.sum()):,} ช่อง) ซึ่งไม่เข้าโมเดล",
          f"ลบแถวซ้ำไป {n_dup:,} แถว",
          f"แถวที่ feature เหมือนกันทุกตัว: แบ่งแบบสุ่ม test {leak_rand:.1f}% มีแถวหน้าตาเดียวกันอยู่ใน train "
          f"ส่วนแบ่งตามเวลาเหลือ {leak_time:.1f}%",
          f"คลาสที่ข้อมูลทับกับคลาสอื่นมากสุด: " + ", ".join(
              f"{c} {conf.loc[c, 'conflict_pct']:.0f}%" for c in conf.index[:3])],
         ["ข้อมูลพร้อมใช้ ไม่ต้อง clean เพิ่ม",
          (f"แบ่งสุ่มรั่วมากกว่าแบ่งตามเวลา {leak_rand - leak_time:.1f} จุด % -> หลักฐาน H3 ว่าแบ่งสุ่มทำให้คะแนนสูงเกินจริง"
           if leak_rand - leak_time > 5 else
           f"แบ่งสุ่มกับแบ่งตามเวลารั่วพอๆ กัน (ต่างกัน {leak_rand - leak_time:.1f} จุด %) -> แถวซ้ำไม่ใช่เหตุผลหลักของ H3 "
           "เหตุผลที่ต้องแบ่งตามเวลามาจาก drift (หัวข้อ 1) แทน"),
          "Backdoor / Analysis จะได้ F1 ต่ำไม่ว่าใช้โมเดลไหน -> เขียนเป็นข้อจำกัด"])


def main() -> None:
    global FIG
    cfg, feat = load_config(), load_config("features.yaml")
    proc, rep = p(cfg["paths"]["processed"]), p(cfg["paths"]["reports"])
    FIG = rep / "figures" / "Feeling"
    FIG.mkdir(parents=True, exist_ok=True)
    for f in [proc / "features.parquet", rep / "ip_lookup.csv", rep / "label_conflicts.csv", rep / "cleaning_report.csv"]:
        if not f.exists():
            raise FileNotFoundError(f"not found: {f} (run progress2 first)")

    print("loading features.parquet + ip_lookup.csv")
    df = pd.read_parquet(proc / "features.parquet")
    df["attack_cat"] = df["attack_cat"].astype(str)
    lk = pd.read_csv(rep / "ip_lookup.csv", dtype={"asn": str})
    print(f"  {len(df):,} rows, {len(lk)} IPs")

    seed = cfg["seed"]
    eda_1_time(df)                     # เวลา
    eda_2_country(df)                  # ประเทศ + เครื่องผู้โจมตี
    eda_3_map(df, lk)                  # รวม 2 ชุด: พิกัด
    eda_4_network(df, lk)              # รวม 2 ชุด: ASN
    eda_5_imbalance(df, feat)          # คลาสไม่สมดุล
    eda_6_leakage(df, feat, seed)      # ทางลัด
    eda_7_quality(df, cfg, feat)       # คุณภาพข้อมูล

    out = rep / f"eda_{NAME}.md"
    out.write_text(f"# EDA — {NAME}\n\nข้อมูล: features.parquet ({len(df):,} แถว, UNSW + ProdIPData) · "
                   f"กราฟอยู่ใน reports/figures/Feeling/{NAME}_*.png\n\n" + "\n".join(REPORT), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
