# 일회성 검증: 결함이 실제로 여러 공간패턴으로 갈리는가, staircase는 반복되는가?
# Local만 95%면 이 방향도 죽는다 -> 숫자로 확인
import os, glob, csv, numpy as np
from scipy import ndimage

ROOT = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\data\mlft_data"
CH = ["CH-A1","CH-A2","CH-A3","CH-A4","CH-A5","CH-B1","CH-B2","CH-B3","CH-B4","CH-B5"]
THR = 1.5  # 장비 reject H 기준(1.5V)과 일치 — 노이즈 스파이크 배제

def load_bar(path):
    with open(path, errors="ignore") as f:
        rows=list(csv.reader(f))
    hdr_i=None; result=None
    for i,r in enumerate(rows):
        if r and r[0]=="Result": result=r[1].strip() if len(r)>1 else None
        if r and r[0]=="No" and "CH-A1" in r: hdr_i=i; break
    if hdr_i is None: return None,None
    hdr=rows[hdr_i]; idx=[hdr.index(c) for c in CH]
    M=[]
    for r in rows[hdr_i+1:]:
        if len(r)<=max(idx): continue
        try: M.append([float(r[j]) for j in idx])
        except: pass
    if not M: return None,None
    return np.array(M), result  # shape (points, 10)

def classify_component(coords, n_points):
    # coords: list of (row, col) in component
    rows=coords[:,0]; cols=coords[:,1]
    area=len(coords)
    row_span=rows.max()-rows.min()+1
    col_span=cols.max()-cols.min()+1   # 몇 개 채널에 걸쳤나
    nchan=len(np.unique(cols))
    # 끝단 여부 (TOP/TAIL 50mm ~ 4 point)
    edge = rows.min()<=4 or rows.max()>=n_points-5
    # diagonal/staircase: row 따라 채널 중심이 단조 이동?
    diag=False
    if row_span>=3 and nchan>=3:
        # 각 row의 평균 채널위치
        order=np.argsort(rows)
        rr=rows[order]; cc=cols[order]
        # row별 채널중심
        uniq=np.unique(rr); centers=[cc[rr==u].mean() for u in uniq]
        if len(centers)>=3:
            d=np.diff(centers)
            mono = np.mean(d>0)>0.7 or np.mean(d<0)>0.7
            if mono and abs(centers[-1]-centers[0])>=2: diag=True
    # 분류
    if area<=2 and nchan<=1: return "point/noise", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)
    if diag: return "diagonal/staircase", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)
    if nchan>=3 and row_span<=2: return "circumferential/band", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)
    if row_span>=4 and nchan<=2: return "longitudinal/line", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)
    if edge: return "edge/end", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)
    return "blob/cluster", dict(area=area,row_span=row_span,nchan=nchan,edge=edge)

# 결함 bar 샘플 수집 (NO GOOD 위주, 여러 LOT)
dates=sorted(d for d in os.listdir(ROOT) if d.isdigit())[:8]
bar_paths=[]
for d in dates:
    dpath=os.path.join(ROOT,d)
    for lot in sorted(os.listdir(dpath)):
        ld=os.path.join(dpath,lot)
        if not (lot.startswith("L") and os.path.isdir(ld)): continue
        for b in sorted(glob.glob(os.path.join(ld,"BAR*.CSV"))):
            bar_paths.append(b)
        if len(bar_paths)>1200: break
    if len(bar_paths)>1200: break

from collections import Counter, defaultdict
pat_counter=Counter(); comp_per_defectbar=[]; n_defbar=0; n_goodbar=0; n_total=0
diag_examples=[]; struct=Counter()
pat_bars=defaultdict(set)  # 패턴별 distinct bar 추적 (반복성)
for bp in bar_paths:
    M,res=load_bar(bp)
    if M is None: continue
    n_total+=1
    mask=M>THR
    if not mask.any():
        if res=="GOOD" or res is None: n_goodbar+=1
        continue
    lbl,nc=ndimage.label(mask)  # default 4-conn; staircase 대각은 끊길 수 있어 8-conn 사용
    lbl,nc=ndimage.label(mask, structure=np.ones((3,3)))
    if nc==0:
        n_goodbar+=1; continue
    n_defbar+=1
    comps=ndimage.find_objects(lbl)
    cnt_this=0
    for ci in range(1,nc+1):
        coords=np.argwhere(lbl==ci)
        if len(coords)<1: continue
        cls,info=classify_component(coords, M.shape[0])
        pat_counter[cls]+=1; cnt_this+=1
        rel=os.path.relpath(bp,ROOT)
        pat_bars[cls].add(rel)
        if cls=="diagonal/staircase" and rel not in [e[0] for e in diag_examples] and len(diag_examples)<10:
            diag_examples.append((rel, info))
    comp_per_defectbar.append(cnt_this)

print(f"스캔 bar: {n_total}, 결함셀 있는 bar: {n_defbar}, 결함셀 없는 bar: {n_goodbar}\n")
total_comp=sum(pat_counter.values())
print(f"총 결함 component: {total_comp}")
print("=== 공간패턴 분포 (component 수 / 나타난 distinct bar 수) ===")
for k,v in pat_counter.most_common():
    nb=len(pat_bars[k])
    print(f"  {k:22s}: {v:5d} comp ({v/total_comp*100:4.1f}%) | {nb:4d}개 bar에 분포 ({nb/max(1,n_defbar)*100:4.1f}% of 결함bar)")
print(f"\nbar당 component 수: 평균 {np.mean(comp_per_defectbar):.1f}, 중앙 {np.median(comp_per_defectbar):.0f}, 최대 {np.max(comp_per_defectbar)}")
print(f"\n=== diagonal/staircase 예시 (반복되는 signature인지) ===")
for p,info in diag_examples:
    print(f"  {p} | {info}")
