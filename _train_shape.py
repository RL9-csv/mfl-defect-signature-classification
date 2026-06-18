# 2차 형태분류: CNN vs GB, 정상 vs 채널셔플 (셔플 붕괴가 본게임)
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
P=r"C:\Users\지엘이테크\Desktop\claude 참고\MFL_2.0"
X=np.load(P+r"\Xs.npy"); y=np.load(P+r"\ys.npy"); g=np.load(P+r"\gs.npy")
SHUF = len(sys.argv)>1 and sys.argv[1]=="shuffle"
if SHUF:
    perm=np.random.RandomState(7).permutation(10); X=X[...,perm]; print("== CHANNEL SHUFFLED ==")
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=42).split(X,y,g))
NC=int(y.max()+1)
def feats(X):
    a=X[:,0];m=X[:,1]
    return np.stack([m.sum((1,2)),m.any(2).sum(1),m.any(1).sum(1),a.max((1,2)),a.mean((1,2))],1)
gb=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(feats(X[tr]),y[tr])
print(f"[GB]  macroF1 {f1_score(y[te],gb.predict(feats(X[te])),average='macro'):.3f}")
dev="cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
    def __init__(s,nc):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,padding_mode="replicate"),nn.BatchNorm2d(o),nn.GELU())
        s.b1=blk(X.shape[1],48);s.b2=blk(48,96);s.b3=blk(96,192);s.pool=nn.MaxPool2d(2);s.drop=nn.Dropout(0.25);s.fc=nn.Linear(192,nc)
    def forward(s,x):
        x=s.pool(s.b1(x));x=s.pool(s.b2(x));x=s.b3(x)
        return s.fc(s.drop(F.adaptive_avg_pool2d(x,1).flatten(1)))
net=Net(NC).to(dev);opt=torch.optim.Adam(net.parameters(),1e-3)
cnt=np.bincount(y[tr],minlength=NC); W=torch.tensor(cnt.sum()/(cnt+1)/NC,dtype=torch.float32).to(dev)  # class weight
Xt=torch.tensor(X[tr]).to(dev);yt=torch.tensor(y[tr]).long().to(dev);Xe=torch.tensor(X[te]).to(dev)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,50)
for ep in range(50):
    net.train();perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),256):
        b=perm[i:i+256];opt.zero_grad();F.cross_entropy(net(Xt[b]),yt[b],weight=W,label_smoothing=0.05).backward();opt.step()
    sch.step()
net.eval()
with torch.no_grad(): pr=net(Xe).argmax(1).cpu().numpy()
print(f"[CNN] macroF1 {f1_score(y[te],pr,average='macro'):.3f}  (dev={dev}, shuffle={SHUF})")
print(classification_report(y[te],pr,target_names=["staircase","band","line"],digits=3))
