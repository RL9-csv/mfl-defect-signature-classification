# 3단계: CNN + GB baseline, LOT group split, PR-AUC/F1
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, f1_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

P=r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0"
X=np.load(P+r"\X_patch.npy"); y=np.load(P+r"\y_patch.npy"); g=np.load(P+r"\g_patch.npy")
import sys
if len(sys.argv)>1 and sys.argv[1]=="shuffle":
    perm=np.random.RandomState(7).permutation(10); X=X[...,perm]
    print("== CHANNEL SHUFFLED ==", perm)
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=42).split(X,y,g))
print(f"train {len(tr)} / test {len(te)} | train LOT {len(set(g[tr]))} test LOT {len(set(g[te]))}")

# --- GB baseline: 패치에서 단순 통계 feature ---
def feats(X):
    amp=X[:,0]; mask=X[:,1]
    return np.stack([mask.sum((1,2)), mask.any(2).sum(1), mask.any(1).sum(1),
                     amp.max((1,2)), amp.mean((1,2)), mask.sum(1).max(1)],1)
gb=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(feats(X[tr]),y[tr])
pgb=gb.predict_proba(feats(X[te]))[:,1]
print(f"[GB shape-feat]  PR-AUC {average_precision_score(y[te],pgb):.3f}  F1 {f1_score(y[te],pgb>0.5):.3f}")

# --- CNN ---
dev="cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,padding_mode="replicate"), nn.BatchNorm2d(o), nn.GELU())
        s.b1=blk(2,32); s.b2=blk(32,64); s.b3=blk(64,128)
        s.pool=nn.MaxPool2d(2); s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,2)
    def forward(s,x):
        x=s.pool(s.b1(x)); x=s.pool(s.b2(x)); x=s.b3(x)
        return s.fc(s.drop(F.adaptive_avg_pool2d(x,1).flatten(1)))
net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
Xtr=torch.tensor(X[tr]).to(dev); ytr=torch.tensor(y[tr]).long().to(dev)
Xte=torch.tensor(X[te]).to(dev)
for ep in range(20):
    net.train(); perm=torch.randperm(len(Xtr))
    for i in range(0,len(Xtr),512):
        b=perm[i:i+512]; opt.zero_grad()
        loss=F.cross_entropy(net(Xtr[b]),ytr[b]); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    pc=torch.softmax(net(Xte),1)[:,1].cpu().numpy()
print(f"[2D CNN]        PR-AUC {average_precision_score(y[te],pc):.3f}  F1 {f1_score(y[te],pc>0.5):.3f}  (dev={dev})")
print(classification_report(y[te],pc>0.5,target_names=["isolated","structured"],digits=3))
