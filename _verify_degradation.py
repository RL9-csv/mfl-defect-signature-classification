# 형 의심 검증용 일회성 EDA: cross-LOT 베이스노이즈 우상향이 진짜 열화냐, 캘/교란이냐
import os, glob, csv, numpy as np
from datetime import datetime

ROOT = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\data\mlft_data"
CH = ["CH-A1","CH-A2","CH-A3","CH-A4","CH-A5","CH-B1","CH-B2","CH-B3","CH-B4","CH-B5"]

def read_lot_meta(lot_dir):
    p = os.path.join(lot_dir, "LOT.CSV")
    steel=size=start=None
    if not os.path.exists(p): return None
    with open(p, errors="ignore") as f:
        for line in f:
            parts=[x.strip() for x in line.split(",")]
            if parts and parts[0]=="STEEL": steel=parts[1]
            elif parts and parts[0]=="SIZE": size=parts[1]
            elif parts and parts[0]=="LOT Start": start=parts[1]
    return dict(steel=steel,size=size,start=start)

def bar_baseline(bar_path):
    # 헤더 스킵 후 데이터, bottom 10% 평균 (10채널 flatten)
    with open(bar_path, errors="ignore") as f:
        rows=list(csv.reader(f))
    hdr_i=None
    for i,r in enumerate(rows):
        if r and r[0]=="No" and "CH-A1" in r:
            hdr_i=i; break
    if hdr_i is None: return None, None
    hdr=rows[hdr_i]; idx=[hdr.index(c) for c in CH]
    vals=[]
    for r in rows[hdr_i+1:]:
        if len(r)<=max(idx): continue
        try: vals.extend(float(r[j]) for j in idx)
        except: pass
    if not vals: return None, None
    a=np.sort(np.array(vals)); k=max(1,int(len(a)*0.1))
    return a[:k].mean(), len(rows)-hdr_i-1

# 날짜 순회, LOT 시간순 수집 (처음 12일만 — 검증엔 충분)
dates=sorted(d for d in os.listdir(ROOT) if d.isdigit())[:12]
records=[]
for d in dates:
    dpath=os.path.join(ROOT,d)
    lots=sorted(x for x in os.listdir(dpath) if x.startswith("L") and os.path.isdir(os.path.join(dpath,x)))
    for lot in lots:
        ldir=os.path.join(dpath,lot)
        meta=read_lot_meta(ldir)
        if not meta or not meta["start"]: continue
        bars=sorted(glob.glob(os.path.join(ldir,"BAR*.CSV")))
        base=[]
        for b in bars:
            v,_=bar_baseline(b)
            if v is not None: base.append(v)
        if not base: continue
        records.append(dict(lot=lot, steel=meta["steel"], size=meta["size"],
                            start=meta["start"], n=len(base),
                            lot_med=float(np.median(base)),
                            bar1=base[0], barlast=base[-1]))

records.sort(key=lambda r: r["start"])
print(f"수집 LOT 수: {len(records)} (처음 12일)\n")

# 연속 동일강종 블록 추출
blocks=[]; cur=[]
for r in records:
    if cur and r["steel"]==cur[-1]["steel"]:
        cur.append(r)
    else:
        if len(cur)>=3: blocks.append(cur)
        cur=[r]
if len(cur)>=3: blocks.append(cur)

print(f"3개 이상 연속 동일강종 블록 수: {len(blocks)}\n")
for bi,blk in enumerate(sorted(blocks,key=len,reverse=True)[:4]):
    steel=blk[0]["steel"]; size=blk[0]["size"]
    meds=[r["lot_med"] for r in blk]
    print(f"=== 블록 {bi}: 강종 {steel} size {size}, LOT {len(blk)}개 ===")
    print("  LOT별 베이스노이즈 중앙값 (시간순):")
    for r in blk:
        print(f"    {r['lot']} | n={r['n']:3d} | LOT중앙값={r['lot_med']:.4f} | bar1={r['bar1']:.4f} barlast={r['barlast']:.4f}")
    x=np.arange(len(meds)); slope=np.polyfit(x,meds,1)[0]
    corr=np.corrcoef(x,meds)[0,1] if len(meds)>1 else 0
    print(f"  >> LOT간 추세: 기울기={slope:+.5f}/LOT, 상관계수 r={corr:+.3f}")
    print(f"  >> 첫 LOT 중앙값={meds[0]:.4f} -> 끝 LOT={meds[-1]:.4f} (변화 {(meds[-1]-meds[0])/meds[0]*100:+.1f}%)\n")
