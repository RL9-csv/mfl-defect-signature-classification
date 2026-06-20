"""의사결정나무로 '결함 형태 규칙'을 검증·시각화 (해석가능 EDA).

수작업 형태 규칙(weak label)을 데이터로 학습한 트리가 재현하는지 확인.
feature 이름은 전부 직관적인 평이한 단어로 표기.
사용: COMPONENTS_CSV=<경로> python -m scripts.decision_tree
산출: assets/06_decision_tree.png, 07_tree_importance.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

CSV = os.environ.get("COMPONENTS_CSV", "_components.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)


def shape(r):
    if r["nchan"] >= 3 and r["row_span"] <= 2:
        return "band"          # 같은 위치, 여러 센서 (원주방향)
    if r["row_span"] >= 4 and r["nchan"] <= 2:
        return "line"          # 한 센서, 길이방향
    if r["diag"] == 1:
        return "staircase"     # 대각 진행
    return None


df["shape"] = df.apply(shape, axis=1)
d = df.dropna(subset=["shape"]).copy()
d["slope"] = d["slope"].abs()

# 트리 입력 feature — 전부 쉬운 이름으로
feats = ["row_span", "nchan", "col_span", "slope", "area", "edge"]
nice = {
    "row_span": "length",        # 길이방향 범위
    "nchan": "n_sensors",        # 걸친 센서 수
    "col_span": "width",         # 센서 폭
    "slope": "diagonal",         # 대각 기울기
    "area": "size",              # 결함 크기
    "edge": "at_edge",           # 끝단 여부
}
X, y = d[feats], d["shape"]

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=80, random_state=0).fit(X, y)
print("형태 재현 정확도:", round(tree.score(X, y), 3))
print("클래스 수:", y.value_counts().to_dict())

# --- 트리 시각화 (깔끔·직관) ---
plt.figure(figsize=(22, 9))
plot_tree(
    tree,
    feature_names=[nice[f] for f in feats],
    class_names=list(tree.classes_),
    filled=True, rounded=True, impurity=False, proportion=True,
    fontsize=11, precision=1,
)
plt.title("Decision Tree — which feature decides each defect shape?  (max depth 3)",
          fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUT}/06_decision_tree.png", dpi=130)
plt.close()

# --- feature 중요도 ---
imp = sorted(zip([nice[f] for f in feats], tree.feature_importances_),
             key=lambda x: x[1])
plt.figure(figsize=(9, 5))
plt.barh([k for k, _ in imp], [v for _, v in imp], color="#4da3ff")
plt.title("Feature importance — what drives defect shape")
plt.xlabel("importance")
plt.tight_layout()
plt.savefig(f"{OUT}/07_tree_importance.png", dpi=130)
plt.close()

print("중요도:")
for k, v in reversed(imp):
    print(f"  {k:10s} {v:.3f}")
