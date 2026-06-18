# late-fusion: CNN(raw 패치) + 방향벡터(amp가중 공분산 eigen) concat
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report
P=r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0"
X=np.load(P+r"\Xs.npy"); y=np.load(P+r"\ys.npy"); g=np.load(P+r"\gs.npy")

def dirfeat(X):
    """각 패치의 방향 기술 벡터 (6차원): amp가중 결함분포의 방향성"""
    out=np.zeros((len(X),6),dtype=np.float32)
    for k in range(len(X)):
        m=X[k,1]; rr,cc=np.where(m>0.5)
        if len(rr)<2: continue
        w=np.clip(X[k,0][rr,cc],1e-3,None); w=w/w.sum()
        r=rr-(rr*w).sum(); c=cc-(cc*w).sum()
        cov=np.array([[ (w*r*r).sum(),(w*r*c).sum()],[(w*r*c).sum(),(w*c*c).sum()]])
        ev,evec=np.linalg.eigh(cov+1e-6*np.eye(2))
        ratio=ev[0]/(ev[1]+1e-6)                 # 정렬도(작을수록 한 방향)
        ang=np.arctan2(evec[1,1],evec[0,1])      # 주축 각도
        # row별 col중심 기울기(staircase 진행)
        uniq=np.unique(rr); ctr=np.array([cc[rr==u].mean() for u in uniq])
        slope=np.polyfit(uniq,ctr,1)[0] if len(uniq)>1 else 0.0
        out[k]=[ratio,np.cos(2*ang),np.sin(2*ang),r.std() if len(rr)>1 else 0,
                len(np.unique(cc)),slope]
    return out

F6=dirfeat(X)
F6=(F6-F6.mean(0))/(F6.std(0)+1e-6)
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=42).split(X,y,g))
NC=int(y.max()+1); dev="cuda" if torch.cuda.is_available() else "cpu"

class Fusion(nn.Module):
    def __init__(s,ic,nf,nc):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,padding_mode="replicate"),nn.BatchNorm2d(o),nn.GELU())
        s.b1=blk(ic,32);s.b2=blk(32,64);s.b3=blk(64,128);s.pool=nn.MaxPool2d(2);s.drop=nn.Dropout(0.3)
        s.ffc=nn.Sequential(nn.Linear(nf,32),nn.GELU())
        s.fc=nn.Linear(128+32,nc)
    def forward(s,x,f):
        c=s.pool(s.b1(x));c=s.pool(s.b2(c));c=s.b3(c);c=F.adaptive_avg_pool2d(c,1).flatten(1)
        return s.fc(torch.cat([s.drop(c),s.ffc(f)],1))

net=Fusion(X.shape[1],6,NC).to(dev);opt=torch.optim.Adam(net.parameters(),1e-3)
cnt=np.bincount(y[tr],minlength=NC); W=torch.tensor(cnt.sum()/(cnt+1)/NC,dtype=torch.float32).to(dev)
Xt=torch.tensor(X[tr]).to(dev);Ft=torch.tensor(F6[tr]).to(dev);yt=torch.tensor(y[tr]).long().to(dev)
Xe=torch.tensor(X[te]).to(dev);Fe=torch.tensor(F6[te]).to(dev)
for ep in range(30):
    net.train();perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),256):
        b=perm[i:i+256];opt.zero_grad();F.cross_entropy(net(Xt[b],Ft[b]),yt[b],weight=W).backward();opt.step()
net.eval()
with torch.no_grad(): pr=net(Xe,Fe).argmax(1).cpu().numpy()
print(f"[FUSION CNN+dir] macroF1 {f1_score(y[te],pr,average='macro'):.3f}")
print(classification_report(y[te],pr,target_names=["staircase","band","line"],digits=3))
