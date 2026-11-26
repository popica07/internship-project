from common import *
write("notebooks/C2_learning_to_rank.ipynb", [
md("""# C2 — Learning-to-rank: pointwise vs pairwise vs listwise (H3)

Six gradient-boosted contenders that differ **only in the objective** (pointwise: XGBoost/LightGBM binary; pairwise: XGBoost `rank:pairwise` = RankNet loss; listwise: XGBoost and LightGBM LambdaMART, LightGBM XE-NDCG). Training queries = publication months; evaluation queries = B4's monthly batches. Each contender re-uses the exact C1 search draws of its library; all are selected on the v2 fold by **mean monthly validation NDCG@100**, refitted with 10 seeds and tested once. Design: Study C addendum 2 (commit 59bd4b1).

**H3:** best-validation LambdaMART beats the best-validation pointwise contender on per-month test NDCG@100 (month bootstrap CI of the mean difference > 0).

Reads saved outputs; full run (~25 min): `python -m src.models.c2_ltr --stage tune` then `--stage final`."""),
SETUP,
md("## Validation search (v2 fold, 20 months)"),
code('t = pd.read_csv("outputs/c2_tuning.csv"); t.groupby("contender").val_ndcg_at_100.describe().round(4)'),
code('pd.read_csv("outputs/c2_selected_configs.csv", index_col=0)[["config_id", "val_ndcg_at_100", "config"]]'),
md("## Test results (10 seeds, mean ± std; global over all 226,463 test CVEs)"),
code('pd.read_csv("outputs/c2_summary.csv", index_col=0)'),
md("## H3 — LambdaMART vs pointwise, per-month NDCG@100 (46 months)"),
code('pd.read_csv("outputs/c2_h3.csv", index_col=0)'),
md("""**H3 not supported.** The selected LambdaMART (LightGBM) is ahead of the selected pointwise model (LightGBM binary) by +0.0049 monthly NDCG@100, but the 95% month-bootstrap CI is [−0.0061, 0.0158] and the Wilcoxon median difference is 0 (p = 0.68). Ranking objectives did not measurably matter at this label density (~7 KEV per monthly queue)."""),
md("## Secondary: global paired bootstrap (seed-averaged scores)"),
code('pd.read_csv("outputs/c2_secondary_tests.csv", index_col=0).round(4)'),
md("## Friedman across the six objectives (46 months)"),
code('pd.read_csv("outputs/c2_friedman.csv", index_col=0).round(4)'),
code('pd.read_csv("outputs/c2_nemenyi_ndcg_at_100.csv", index_col=0).round(3)'),
md("""**Reading.** Friedman is significant (NDCG@100 p = 0.005, MAP p = 1e-4), driven mainly by XE-NDCG being clearly worst (it also had the most unstable validation search). Among the five working objectives the mean ranks are within ~0.6 of each other. Pairwise XGBoost has the highest global NDCG@100 and P@100, LambdaMART the highest global PR-AUC, but no pairwise difference to the pointwise model is significant. Nothing here beats point-in-time EPSS significantly (best LambdaMART − EPSS-PIT PR-AUC +0.0037 [−0.0020, 0.0103])."""),
])
