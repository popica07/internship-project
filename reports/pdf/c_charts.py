import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
P, S = sys.argv[1], sys.argv[2]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

def style(ax):
    ax.set_facecolor(SURF); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=INK2, labelsize=7.5); ax.tick_params(axis="y", length=0)

# ---- overview of all 17 methods
names = {"C1R:logreg": "Logistic regression", "C1R:xgboost": "XGBoost", "C1R:lightgbm": "LightGBM",
         "C1R:catboost": "CatBoost", "C1R:random_forest": "Random forest", "C1R:decision_tree": "Decision tree",
         "C2:lgbm_lambdamart": "LambdaMART (LightGBM)", "C2:xgb_lambdamart": "LambdaMART (XGBoost)",
         "C2:xgb_pairwise": "Pairwise / RankNet loss (XGBoost)", "C2:lgbm_binary": "Pointwise (LightGBM)",
         "C2:xgb_binary": "Pointwise (XGBoost)", "C2:lgbm_xendcg": "XE-NDCG (LightGBM)",
         "C3:ft_transformer": "FT-Transformer", "C3:mlp": "MLP", "C3:tabnet": "TabNet",
         "baseline:epss_pit": "Point-in-time EPSS", "baseline:cvss_v3": "CVSS v3 (current practice)"}
group_color = {"C1R": BLUE, "baseline": ORANGE, "C2": AQUA, "C3": YELLOW}
r = pd.read_csv(f"{P}/outputs/c6_global_by_seed.csv")
g = r.groupby("method").pr_auc.agg(["mean", "std"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200, facecolor=SURF)
y = range(len(g))
ax.barh(y, g["mean"], height=0.62, color=[group_color[m.split(":")[0]] for m in g.index], edgecolor=SURF, linewidth=2)
ax.errorbar(g["mean"], y, xerr=g["std"].fillna(0), fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5)
for i, (m, s) in enumerate(zip(g["mean"], g["std"].fillna(0))):
    ax.text(m + s + 0.0003, i, f"{m:.4f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(list(y), [names[m] for m in g.index], fontsize=7.8, color=INK)
ax.set_xlim(0, g["mean"].max() * 1.17); style(ax)
ax.set_xlabel("Test PR-AUC (mean ± std over 10 seeds)", fontsize=7.8, color=INK2)
ax.legend(handles=[Patch(color=BLUE, label="C1 classical"), Patch(color=AQUA, label="C2 learning-to-rank"),
                   Patch(color=YELLOW, label="C3 deep"), Patch(color=ORANGE, label="baseline")],
          loc="lower right", fontsize=7.5, frameon=False)
ax.set_title("Study C: all models on the same test set (309 KEV in 226,463 CVEs)", fontsize=9.5, color=INK, loc="left", pad=8)
fig.tight_layout(); fig.savefig(f"{S}/c_overview.png", facecolor=SURF); plt.close(fig)

# ---- ablation (XGBoost and LR)
a = pd.read_csv(f"{P}/outputs/c4_l1_results_by_seed.csv")
a = a[a.kind == "ablation"]
label = {"all": "All features", "cvss+epss": "CVSS + EPSS", "text+epss": "Text + EPSS", "epss": "EPSS only",
         "cvss+text": "CVSS + text (no EPSS)", "cvss": "CVSS only", "text": "Text only"}
order = ["text", "cvss", "cvss+text", "epss", "text+epss", "cvss+epss", "all"]
m = a.groupby(["variant", "model"]).pr_auc.mean().unstack()
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200, facecolor=SURF)
yy = range(len(order)); h = 0.36
ax.barh([i + h / 2 for i in yy], m.loc[order, "xgboost"], height=h, color=BLUE, edgecolor=SURF, linewidth=1.5, label="XGBoost")
ax.barh([i - h / 2 for i in yy], m.loc[order, "logreg"], height=h, color=AQUA, edgecolor=SURF, linewidth=1.5, label="Logistic regression")
for i, v in enumerate(order):
    for off, col in ((h / 2, "xgboost"), (-h / 2, "logreg")):
        ax.text(m.loc[v, col] + 0.0003, i + off, f"{m.loc[v, col]:.4f}", va="center", fontsize=6.5, color=INK2, bbox=dict(facecolor=SURF, edgecolor="none", pad=0.3))
ax.axvline(0.0068, color=ORANGE, linewidth=1.2, linestyle="--", zorder=0)
ax.set_ylim(-0.6, len(order) + 0.15)
ax.text(0.0068, len(order) - 0.45, " CVSS v3 ordering (0.0068)", color=INK2, fontsize=7, va="bottom")
ax.set_yticks(list(yy), [label[v] for v in order], fontsize=7.8, color=INK)
ax.set_xlim(0, m.max().max() * 1.2); style(ax)
ax.set_xlabel("Test PR-AUC (XGBoost: mean of 10 seeds)", fontsize=7.8, color=INK2)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
ax.set_title("C4: which feature groups carry the signal", fontsize=9.5, color=INK, loc="left", pad=8)
fig.tight_layout(); fig.savefig(f"{S}/c4_chart.png", facecolor=SURF); plt.close(fig)
print("ok")
