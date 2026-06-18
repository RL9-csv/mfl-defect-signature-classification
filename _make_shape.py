# 2차: structured 내부 형태 3클래스 (staircase/band/line) 패치
import os, glob, csv, numpy as np
from scipy import ndimage
ROOT=r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0\data\mlft_data"
OUT=r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0"
CH=["CH-A1","CH-A2","CH-A3","CH-A4","CH-A5","CH-B1","CH-B2","CH-B3","CH-B4","CH-B5"]
THR=1.5; L=64; N_DAYS=20; rng=np.random.RandomState(42)
MAP={"staircase":0,"band":1,"line":2}

def load_bar(p):
    try:
        with open(p,errors="ignore") as f: txt=f.read().replace("\x00","")
        rows=list(csv.reader(txt.splitlines()))
    except: return None
    hi=None
    for i,r in enumerate(rows):
        if r and r[0]=="No" and "CH-A1" in r: hi=i;break
    if hi is None: return None
    idx=[rows[hi].index(c) for c in CH]; M=[]
    for r in rows[hi+1:]:
        if len(r)<=max(idx): continue
        try: M.append([float(r[j]) for j in idx])
        except: pass
    return np.array(M) if M else None

def shape_label(coords):
    rows=coords[:,0];cols=coords[:,1];area=len(coords)
    rs=int(rows.max()-rows.min()+1); nch=int(len(np.unique(cols)))
    if area<=2 and nch<=1: return None
    uniq=np.unique(rows);ctr=[cols[rows==u].mean() for u in uniq]
    if len(ctr)>=3:
        d=np.diff(ctr);mono=max(np.mean(d>0),np.mean(d<0))
        if rs>=3 and nch>=3 and mono>0.7 and abs(ctr[-1]-ctr[0])>=2: return "staircase"
    if nch>=3 and rs<=2: return "band"
    if rs>=4 and nch<=2: return "line"
    return None

def patch(M,c):
    h=L//2; lo=c-h; hi=c+h; pl=max(0,-lo); pr=max(0,hi-M.shape[0]); lo=max(0,lo); hi=min(M.shape[0],hi)
    p=M[lo:hi]
    if pl or pr: p=np.pad(p,((pl,pr),(0,0)),mode="edge")
    if p.shape[0]>L: p=p[:L]
    elif p.shape[0]<L: p=np.pad(p,((0,L-p.shape[0]),(0,0)),mode="edge")
    return p
def tens(p):
    amp=np.log1p(np.clip(p,0,None)).astype(np.float32)
    mask=(p>=THR).astype(np.float32)
    dl=np.gradient(amp,axis=0).astype(np.float32)   # 길이방향 변화율
    dc=np.gradient(amp,axis=1).astype(np.float32)   # 채널방향 변화율
    return np.stack([amp,mask,dl,dc],0).astype(np.float32)  # 4채널

buckets={0:[],1:[],2:[]}
for d in sorted(x for x in os.listdir(ROOT) if x.isdigit()):  # 전체 86일
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
                co=np.argwhere(lbl==ci); s=shape_label(co)
                if s is None: continue
                c=int(round(co[:,0].mean()))
                buckets[MAP[s]].append((tens(patch(M,c)),MAP[s],lot))
print(f"클래스 원수량: staircase {len(buckets[0])} / band {len(buckets[1])} / line {len(buckets[2])}")
rng.shuffle(buckets[0]); buckets[0]=buckets[0][:8000]  # staircase만 캡(나머지 전량) -> class weight로 불균형 처리
data=buckets[0]+buckets[1]+buckets[2]
rng.shuffle(data)
X=np.stack([r[0] for r in data]); y=np.array([r[1] for r in data]); g=np.array([r[2] for r in data])
np.save(OUT+r"\Xs.npy",X); np.save(OUT+r"\ys.npy",y); np.save(OUT+r"\gs.npy",g)
print(f"저장 X{X.shape} | LOT {len(set(g))}개 | 클래스분포 {np.bincount(y)}")
