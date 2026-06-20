"""통계 EDA 시각화 생성 — PE 직무 연결(SPC·상관·분포)과 딥러닝 근거.

사용: MFL_DATA=<원본경로> python -m scripts.visualize
산출: assets/ 에 PNG 5종.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mfl.config import CHANNELS, DATA_ROOT, DEFECT_THRESHOLD
from mfl.defects import extract_components, extract_patch, shape_label, to_tensor
from mfl.io import load_bar
from mfl.topology import sensor_transition_matrix

sns.set_theme(style="whitegrid", context="talk")
CH = [c.replace("CH-", "") for c in CHANNELS]
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def trimmed_base(bar, frac=0.1):
    """bar 10채널 flatten 중 bottom 10% 평균 = 베이스 노이즈(표본평균)."""
    v = np.sort(bar.flatten())
    k = max(1, int(len(v) * frac))
    return v[:k].mean()


# ---- 데이터 수집 (처음 며칠 LOT 샘플) ----
dates = sorted(d for d in os.listdir(DATA_ROOT) if d.isdigit())[:4]
lot_series, all_vals, comps, sample_bar = {}, [], [], None
shape_ex = {"staircase": None, "band": None, "line": None}
for d in dates:
    for lot in sorted(os.listdir(os.path.join(DATA_ROOT, d))):
        ld = os.path.join(DATA_ROOT, d, lot)
        if not (lot.startswith("L") and os.path.isdir(ld)):
            continue
        for bp in sorted(glob.glob(os.path.join(ld, "BAR*.CSV"))):
            bar = load_bar(bp)
            if bar is None:
                continue
            if sample_bar is None and bar.shape[0] > 450:
                sample_bar = bar
            lot_series.setdefault(lot, []).append(trimmed_base(bar))
            if len(all_vals) < 300:
                all_vals.append(bar.flatten())
            for co in extract_components(bar):
                comps.append(co)
                lab = shape_label(co)
                if lab in shape_ex and shape_ex[lab] is None:
                    shape_ex[lab] = to_tensor(extract_patch(bar, int(round(co[:, 0].mean()))))[0]

# ---- 1. SPC 관리도 ----
lot = max(lot_series, key=lambda k: len(lot_series[k]))
s = np.array(lot_series[lot])
mu, sd = s.mean(), s.std()
plt.figure(figsize=(11, 5))
plt.plot(s, "o-", color="#4da3ff", label="base noise (trimmed-10% mean)")
for k, c, ls in [(3, "#b04a4a", "--"), (0, "#888", "-")]:
    plt.axhline(mu + k * sd, color=c, ls=ls, lw=1.3,
                label=f"mean{'+3σ (UCL)' if k else ' (CL)'}")
plt.axhline(mu - 3 * sd, color="#b04a4a", ls="--", lw=1.3, label="mean−3σ (LCL)")
out = np.where(np.abs(s - mu) > 3 * sd)[0]
if len(out):
    plt.scatter(out, s[out], color="red", zorder=5, s=90, label="out-of-control")
plt.title(f"SPC Control Chart — base-noise per bar (LOT {lot})")
plt.xlabel("bar sequence"); plt.ylabel("base noise (V)"); plt.legend(fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/01_spc_control_chart.png", dpi=130); plt.close()

# ---- 2. 채널 상관 히트맵 (슈머: 파라미터 상관) ----
corr = np.corrcoef(sample_bar.T)
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, xticklabels=CH, yticklabels=CH, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7}, square=True, cbar_kws={"shrink": .8})
plt.title("Inter-channel Signal Correlation (single bar)")
plt.tight_layout(); plt.savefig(f"{OUT}/02_channel_correlation.png", dpi=130); plt.close()

# ---- 3. 신호 분포: raw vs log1p ----
vals = np.concatenate(all_vals)
vals = vals[(vals > 0) & (vals < 5)]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(vals, bins=80, color="#888", ax=ax[0])
ax[0].set_title("Raw amplitude (right-skewed)"); ax[0].set_xlabel("V")
sns.histplot(np.log1p(vals), bins=80, color="#4da3ff", ax=ax[1])
ax[1].set_title("log1p amplitude (→ used for CNN input)"); ax[1].set_xlabel("log1p(V)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_distribution_log1p.png", dpi=130); plt.close()

# ---- 4. 결함 형태 예시 heatmap ----
fig, ax = plt.subplots(1, 3, figsize=(13, 5))
for a, (name, patch) in zip(ax, shape_ex.items()):
    if patch is None:
        continue
    sns.heatmap(patch.T, cmap="magma", cbar=False, ax=a, xticklabels=False, yticklabels=CH)
    a.set_title(f"{name}"); a.set_xlabel("scan direction")
fig.suptitle("Defect spatial signatures (CNN input, log1p amplitude)")
plt.tight_layout(); plt.savefig(f"{OUT}/04_defect_shapes.png", dpi=130); plt.close()

# ---- 5. 센서 전이행렬 히트맵 (원주순서 근거) ----
T = sensor_transition_matrix(comps)
plt.figure(figsize=(8, 6.5))
sns.heatmap(T, xticklabels=CH, yticklabels=CH, cmap="viridis",
            annot=True, fmt=".0f", annot_kws={"size": 7}, square=True, cbar_kws={"shrink": .8})
plt.title("Sensor Transition Matrix (topology recovery: A5–B1 adjacency)")
plt.xlabel("to channel"); plt.ylabel("from channel")
plt.tight_layout(); plt.savefig(f"{OUT}/05_sensor_transition.png", dpi=130); plt.close()

print("saved 5 figures to assets/:")
for f in sorted(os.listdir(OUT)):
    print("  -", f)
