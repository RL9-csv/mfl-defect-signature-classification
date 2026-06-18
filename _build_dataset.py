# 1단계: 결함 component 데이터셋 구축 + weak label + 센서 물리배치 역추정
import os, glob, csv, numpy as np
from scipy import ndimage
from collections import Counter, defaultdict

ROOT = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\data\mlft_data"
OUT  = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\_components.csv"
CH = ["CH-A1","CH-A2","CH-A3","CH-A4","CH-A5","CH-B1","CH-B2","CH-B3","CH-B4","CH-B5"]
THR = 1.5
N_DAYS = 20  # 처음 20일 (통계 안정 + 데이터셋)

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
    return np.array(M), result

def comp_features(coords, vals_map, n_pts):
    rows=coords[:,0]; cols=coords[:,1]
    area=len(coords); row_span=int(rows.max()-rows.min()+1)
    nchan=int(len(np.unique(cols))); col_span=int(cols.max()-cols.min()+1)
    edge = bool(rows.min()<=4 or rows.max()>=n_pts-5)
    peak=float(vals_map[rows,cols].max())
    # row별 채널중심 시퀀스 (센서배치 역추정 + 방향성)
    uniq=np.unique(rows); centers=[cols[rows==u].mean() for u in uniq]
    slope=0.0; mono=0.0
    if len(centers)>=2:
        d=np.diff(centers)
        slope=float(np.mean(d))
        mono=float(max(np.mean(d>0), np.mean(d<0)))
    diag = bool(row_span>=3 and nchan>=3 and mono>0.7 and abs(centers[-1]-centers[0])>=2)
    return dict(area=area,row_span=row_span,nchan=nchan,col_span=col_span,
                edge=edge,peak=peak,slope=slope,mono=mono,diag=diag,
                seq=[int(round(c)) for c in centers])

def weak_label(f):
    # structured(진짜 결함 후보) vs isolated(노이즈성 허위 후보) vs ambiguous
    if f["area"]<=2 and f["nchan"]<=1: return "isolated"          # 고립 단발
    if f["diag"]: return "structured"                              # staircase
    if f["nchan"]>=3 and f["row_span"]<=2: return "structured"    # band(원주)
    if f["row_span"]>=4 and f["nchan"]<=2: return "structured"    # line(길이방향)
    return "ambiguous"

dates=sorted(d for d in os.listdir(ROOT) if d.isdigit())[:N_DAYS]
rows_out=[]; lbl_cnt=Counter(); lbl_bars=defaultdict(set)
trans=np.zeros((10,10))   # 센서 인접 전이행렬 (staircase 채널 이동)
n_bar=0; n_clean=0
for d in dates:
    dpath=os.path.join(ROOT,d)
    for lot in sorted(os.listdir(dpath)):
        ld=os.path.join(dpath,lot)
        if not (lot.startswith("L") and os.path.isdir(ld)): continue
        for bp in sorted(glob.glob(os.path.join(ld,"BAR*.CSV"))):
            M,res=load_bar(bp)
            if M is None: continue
            n_bar+=1
            mask=M>THR
            if not mask.any(): n_clean+=1; continue
            lbl,nc=ndimage.label(mask, structure=np.ones((3,3)))
            bar_id=f"{d}/{lot}/{os.path.basename(bp)}"
            for ci in range(1,nc+1):
                coords=np.argwhere(lbl==ci)
                f=comp_features(coords, M, M.shape[0])
                lab=weak_label(f)
                lbl_cnt[lab]+=1; lbl_bars[lab].add(bar_id)
                rows_out.append([bar_id,ci,f["area"],f["row_span"],f["nchan"],
                                 f["col_span"],int(f["edge"]),round(f["peak"],3),
                                 round(f["slope"],3),round(f["mono"],3),int(f["diag"]),lab])
                if f["diag"]:
                    s=f["seq"]
                    for a,b in zip(s,s[1:]):
                        if 0<=a<10 and 0<=b<10: trans[a,b]+=1

with open(OUT,"w",newline="",encoding="utf-8") as fo:
    w=csv.writer(fo)
    w.writerow(["bar_id","comp_id","area","row_span","nchan","col_span","edge","peak","slope","mono","diag","label"])
    w.writerows(rows_out)

print(f"처리 bar {n_bar} (clean {n_clean}, {n_clean/n_bar*100:.1f}%) | component {len(rows_out)} 저장 -> _components.csv\n")
print("=== weak label 분포 (component / distinct bar) ===")
tot=sum(lbl_cnt.values())
for k in ["structured","isolated","ambiguous"]:
    v=lbl_cnt[k]; print(f"  {k:11s}: {v:6d} ({v/tot*100:5.1f}%) | {len(lbl_bars[k]):4d} bar")
print(f"\n=== 센서 인접 전이행렬 (staircase 채널 이동, 행=from 열=to) ===")
print("       " + " ".join(f"{c[3:]:>4s}" for c in CH))
for i in range(10):
    print(f"  {CH[i][3:]:>4s}: " + " ".join(f"{int(trans[i,j]):4d}" for j in range(10)))
# 가장 흔한 인접쌍 top
pairs=[]
for i in range(10):
    for j in range(10):
        if i!=j and trans[i,j]>0: pairs.append((trans[i,j]+trans[j,i], CH[i][3:], CH[j][3:]))
pairs=sorted(set((p,a,b) for p,a,b in pairs if a<b), reverse=True)[:8]
print("\n  최빈 인접쌍(양방향합): " + ", ".join(f"{a}-{b}({int(c)})" for c,a,b in pairs))
