import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
P = sys.argv[1]; out = sys.argv[2]
r = pd.read_csv(f"{P}/outputs/b2_results_by_seed.csv")
names = {"logreg": "Logistic regression", "epss_pit": "Point-in-time EPSS", "gbm": "Gradient boosting (primary)",
         "cvss_v3": "CVSS v3 (current practice)", "random": "Random"}
learned = {"logreg", "gbm"}
g = r[r.scorer.isin(names)].groupby("scorer").pr_auc.agg(["mean", "std"]).sort_values("mean")
BLUE, ORANGE, SURF, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
y = range(len(g))
colors = [BLUE if s in learned else ORANGE for s in g.index]
ax.barh(y, g["mean"], height=0.55, color=colors, edgecolor=SURF, linewidth=2)
ax.errorbar(g["mean"], y, xerr=g["std"].fillna(0), fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
for i, (m, s) in enumerate(zip(g["mean"], g["std"].fillna(0))):
    ax.text(m + s + 0.0004, i, f"{m:.4f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(list(y), [names[s] for s in g.index], fontsize=8.5, color=INK)
ax.set_xlabel("PR-AUC (mean ± std over 10 seeds; deterministic scorers have no bar)", fontsize=8, color=INK2)
ax.set_xlim(0, g["mean"].max() * 1.22)
ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=INK2, labelsize=8); ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="learned model"), Patch(color=ORANGE, label="baseline")],
          loc="lower right", fontsize=8, frameon=False)
ax.set_title("B2: KEV prediction on the evaluation set (309 KEV in 226,463 CVEs)", fontsize=10, color=INK, loc="left", pad=8)
fig.tight_layout(); fig.savefig(out, facecolor=SURF)
