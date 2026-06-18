# 2단계: component 패치 텐서 생성 (2,L,10) 원주순서, replicate padding, LOT group
import os, glob, csv, numpy as np
from scipy import ndimage

ROOT = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\data\mlft_data"
OUT  = r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0"
CH = ["CH-A1","CH-A2","CH-A3","CH-A4","CH-A5","CH-B1","CH-B2","CH-B3","CH-B4","CH-B5"]  # 원주순서
THR=1.5; L=64; N_DAYS=20
rng=np.random.RandomState(42)

def load_bar(path):
    with open(path, errors="ignore") as f: rows=list(csv.reader(f))
    hi=None
    for i,r in enumerate(rows):
        if r and r[0]=="No" and "CH-A1" in r: hi=i; break
    if hi is None: return None
    idx=[rows[hi].index(c) for c in CH]; M=[]
    for r in rows[hi+1:]:
        if len(r)<=max(idx): continue
        try: M.append([float(r[j]) for j in idx])
        except: pass
    return np.array(M) if M else None

def weak_label(coords, npts):
    rows=coords[:,0]; cols=coords[:,1]; area=len(coords)
    rs=int(rows.max()-rows.min()+1); nch=int(len(np.unique(cols)))
    if area<=2 and nch<=1: return "isolated"
    uniq=np.unique(rows); ctr=[cols[rows==u].mean() for u in uniq]
    if len(ctr)>=3:
        d=np.diff(ctr); mono=max(np.mean(d>0),np.mean(d<0))
        if rs>=3 and nch>=3 and mono>0.7 and abs(ctr[-1]-ctr[0])>=2: return "structured"
    if nch>=3 and rs<=2: return "structured"
    if rs>=4 and nch<=2: return "structured"
    return "ambiguous"

def patch(M, center):
    half=L//2; lo=center-half; hi=center+half
    pl=max(0,-lo); pr=max(0,hi-M.shape[0]); lo=max(0,lo); hi=min(M.shape[0],hi)
    p=M[lo:hi]
    if pl or pr: p=np.pad(p,((pl,pr),(0,0)),mode="edge")
    if p.shape[0]>L: p=p[:L]
    elif p.shape[0]<L: p=np.pad(p,((0,L-p.shape[0]),(0,0)),mode="edge")
    return p  # (L,10)

def to_tensor(p):
    amp=np.log1p(np.clip(p,0,None)).astype(np.float32)  # 결함 스파이크/채널간 상대크기 보존
    mask=(p>=THR).astype(np.float32)
    return np.stack([amp,mask],0).astype(np.float32)  # (2,L,10)

dates=sorted(d for d in os.listdir(ROOT) if d.isdigit())[:N_DAYS]
Xs=[]; ys=[]; grp=[]   # y: 1=structured 0=isolated
struct=[]; iso=[]
for d in dates:
    dp=os.path.join(ROOT,d)
    for lot in sorted(os.listdir(dp)):
        ld=os.path.join(dp,lot)
        if not (lot.startswith("L") and os.path.isdir(ld)): continue
        for bp in sorted(glob.glob(os.path.join(ld,"BAR*.CSV"))):
            M=load_bar(bp)
            if M is None: continue
            mask=M>THR
            if not mask.any(): continue
            lbl,nc=ndimage.label(mask,structure=np.ones((3,3)))
            for ci in range(1,nc+1):
                coords=np.argwhere(lbl==ci)
                lab=weak_label(coords,M.shape[0])
                if lab=="ambiguous": continue
                center=int(round(coords[:,0].mean()))
                rec=(to_tensor(patch(M,center)), 1 if lab=="structured" else 0, lot)
                (struct if lab=="structured" else iso).append(rec)

# isolated 언더샘플 -> structured와 동수 (불균형 완화, 원본 비율은 평가때 복원 가능)
rng.shuffle(iso); iso=iso[:len(struct)]
data=struct+iso; rng.shuffle(data)
X=np.stack([r[0] for r in data]); y=np.array([r[1] for r in data]); g=np.array([r[2] for r in data])
np.save(os.path.join(OUT,"X_patch.npy"),X); np.save(os.path.join(OUT,"y_patch.npy"),y); np.save(os.path.join(OUT,"g_patch.npy"),g)
print(f"패치 저장: X{X.shape} y{y.shape} | structured {len(struct)} / isolated(언더샘플) {len(iso)} | LOT {len(set(g))}개")
print(f"  positive(structured) 비율 {y.mean()*100:.1f}% | dtype {X.dtype} | 용량 ~{X.nbytes/1e6:.0f}MB")
