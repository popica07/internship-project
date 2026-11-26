import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# C1-R — Classical suite under selection protocol v2 (post-hoc for C1)

C1's validation fold (2022, **39 KEV**) did not predict test rank: LightGBM was selected and came 5th on test. Protocol v2 (pre-registered in `reports/analysis_plan_preregistration.md`, commit e21fe21, *before* this run) uses a wider fold: train on CVEs published < 2021-05-01, validate on 2021-05-01 … 2022-12-31 with KEV as of 2023-01-01 — **652 train / 145 validation positives**. The same 20 configs + default per family as C1 are re-scored; the selected configs are refitted with 10 seeds and tested once.

**Status:** C1's primary verdict stands (H2, H1b not supported). C1-R is reported next to it, never instead of it. v2 is binding for C2–C6.

This notebook reads saved outputs. Full run (~5 h on 5 threads): `python -m src.models.c1_revised --stage tune|final|diagnostic|analyse`."""),
code("""import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
pd.set_option("display.width", 220)"""),
md("## Validation search on the v2 fold"),
code("""t = pd.read_csv("outputs/c1r_tuning.csv")
t.groupby("family").val_pr_auc.describe().round(4)"""),
code('pd.read_csv("outputs/c1r_selected_configs.csv", index_col=0)'),
md("## Test results (10 seeds, mean ± std) — C1-R vs C1"),
code("""r1 = pd.read_csv("outputs/c1_results_by_seed.csv"); r1 = r1[r1.variant == "tuned"]
r2 = pd.read_csv("outputs/c1r_results_by_seed.csv")
cmp = pd.concat({"C1 (v1 fold)": r1.groupby("family")[["pr_auc", "precision_at_100"]].mean(),
                 "C1-R (v2 fold)": r2.groupby("family")[["pr_auc", "precision_at_100"]].mean()}, axis=1)
cmp.round(4).sort_values(("C1-R (v2 fold)", "pr_auc"), ascending=False)"""),
code('pd.read_csv("outputs/c1r_summary.csv", index_col=[0, 1])'),
md("## H2 and H1b re-tested under v2 (paired bootstrap, seed-averaged scores)"),
code('pd.read_csv("outputs/c1r_hypothesis_tests.csv", index_col=0).round(4)'),
md("""**Reading.** Best-validation GBM is now XGBoost (5th → 2nd-best on test vs C1's pick). H2 is still not supported: XGBoost does not beat LR (−0.0013 [−0.0070, 0.0031]); it does beat the decision tree. H1b: LR beats point-in-time EPSS on PR-AUC (+0.0060 [0.0013, 0.0121]) but not on P@100 (+0.01 [−0.03, 0.05]); H1b requires both, so it is **not supported** as registered — a partial result."""),
md("## Selection reliability: does validation rank predict test rank?\n\nAll 126 configs fitted once on the full training set (seed 0) and scored on test (diagnostic only — never used for selection). Spearman between validation and test PR-AUC, per fold."),
code('pd.read_csv("outputs/c1r_selection_reliability.csv").round(4)'),
md("""The v2 fold ranks configurations more like the test set does: pooled ρ 0.68 → 0.78, and within the GBM families that failed in C1 the gain is large (LightGBM 0.16 → 0.68, XGBoost 0.40 → 0.77). Within LR the configs barely differ on test (0.0181–0.0243), so its ρ is uninformative. The test PR-AUC of the v2 pick is closer to the best achievable for LightGBM and LR; for XGBoost and CatBoost the v1 pick happened to be better. This supports the v2 fold for C2–C6, with the caveat that ρ is estimated on 21 configs per family."""),
md("## Friedman across families (46 monthly batches)"),
code('pd.read_csv("outputs/c1r_friedman.csv", index_col=0).round(4)'),
code('pd.read_csv("outputs/c1r_nemenyi_map.csv", index_col=0).round(3)'),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nbf.write(nb, "notebooks/C1R_selection_protocol_v2.ipynb")
