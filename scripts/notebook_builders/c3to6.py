from common import *
write("notebooks/C3_deep_tabular.ipynb", [
md("""# C3 — Deep tabular models: MLP, TabNet, FT-Transformer (H4)

Inputs: C1-R features with median imputation + missing indicators + standard scaling; unweighted BCE; no early stopping (epochs searched). Search on the v2 fold (MLP 20 configs, TabNet 10, FT-Transformer 8 — budgets fixed in advance from a speed-only benchmark), selection by validation PR-AUC, 10 seeds, test once. Design: Study C addendum 2 (commit 59bd4b1).

**H4 (registered: deep models do NOT beat GBMs):** best-validation deep model vs C1-R XGBoost, paired bootstrap of PR-AUC; supported if the CI of deep − GBM is not entirely above 0.

Reads saved outputs; full run ≈ 2.5 h on 5 CPU threads (FT-Transformer ≈ 8 min per fit): `python -m src.models.c3_deep --stage tune` then `--stage final`. Note: torch must be imported before LightGBM/XGBoost on macOS (two OpenMP runtimes otherwise segfault)."""),
SETUP,
md("## Validation search (v2 fold)"),
code('t = pd.read_csv("outputs/c3_tuning.csv"); t.groupby("model").val_pr_auc.describe().round(4)'),
code('t.sort_values("val_pr_auc", ascending=False).head(8)[["model", "config_id", "val_pr_auc", "seconds", "config"]]'),
md("FT-Transformer is the best deep model on validation only because of **one** configuration (0.091); its other seven were 0.004–0.011. Selection is by the registered rule regardless."),
md("## Test results (10 seeds, mean ± std)"),
code('pd.read_csv("outputs/c3_summary.csv", index_col=0)'),
md("## H4 and secondary comparisons (paired bootstrap, seed-averaged scores)"),
code('pd.read_csv("outputs/c3_hypothesis_tests.csv", index_col=0).round(4)'),
md("""**H4 supported, in its weak (non-superiority) form.** FT-Transformer − XGBoost PR-AUC = +0.0003 [−0.0043, 0.0087]: no evidence that the deep model is better — but the GBM is not significantly better either, so this is "no difference detected", not "GBMs win". FT-Transformer is also the noisiest model across seeds (PR-AUC ± 0.0017 vs ± 0.0002 for XGBoost) and needs ~100× the compute. MLP and TabNet are numerically below XGBoost and LR; MLP is significantly below LR (−0.0056 [−0.0118, −0.0014]). FT-Transformer is marginally above point-in-time EPSS on PR-AUC (+0.0050 [0.0001, 0.0118], secondary), not on P@100."""),
])

write("notebooks/C4_ablation_imbalance.ipynb", [
md("""# C4 — Feature ablation and class imbalance (exploratory)

L1: feature groups CVSS / description text / point-in-time EPSS — each alone, each pair, all — with the C1-R-selected XGBoost (10 seeds) and LR, **not re-tuned per subset**. Imbalance on XGBoost: none, `scale_pos_weight` √(neg/pos) and neg/pos, undersampling to 1:20; LR balanced. L3 (**simulated, reported separately**): finding-only vs context-only vs all. Design: Study C addendum 2 (commit 59bd4b1). No hypothesis registered.

Reads saved outputs; run: `python -m src.models.c4_ablation --stage l1` and `--stage l3` (~45 min)."""),
SETUP,
md("## L1 — ablation and imbalance (10 seeds, mean ± std)"),
code('pd.read_csv("outputs/c4_l1_summary.csv", index_col=[0, 1, 2])'),
md("## Each subset / imbalance variant vs the full model (paired bootstrap)"),
code('pd.read_csv("outputs/c4_l1_tests.csv", index_col=0).round(4)'),
md("""**Reading (L1).**
- **EPSS carries the signal.** Every set without point-in-time EPSS collapses to CVSS-baseline level (CVSS+text PR-AUC ≈ 0.007, vs 0.0068 for CVSS ordering). CVSS+EPSS is statistically indistinguishable from all features (XGBoost −0.0023 [−0.0068, 0.0012]; LR −0.0008 [−0.0071, 0.0058]).
- **Text adds little once CVSS and EPSS are present**, and the 64-d description embedding is weak on its own (PR-AUC ≈ 0.005).
- **Imbalance handling does not help ranking:** no weighting/undersampling variant changes PR-AUC or P@100 significantly — but it **destroys calibration** (Brier 0.0014 → 0.064 with full `scale_pos_weight`, ECE 0.20). C5 shows Platt scaling repairs this. Conclusion: train unweighted.
- Consistency check: "all / xgboost" reproduces C1-R XGBoost exactly (0.0220 ± 0.0002)."""),
md("## L3 — SIMULATED estate (target: High/Critical, 10 grouped-by-asset splits)"),
code('pd.read_csv("outputs/c4_l3_summary.csv", index_col=[0, 1])'),
md("""**Reading (L3, simulated — not evidence about a real estate).** Finding-only and context-only features each reach PR-AUC ≈ 0.39; together ≈ 0.998, i.e. XGBoost recovers the documented label formula almost exactly once it sees both halves of it. This confirms the formula really needs both the finding and the asset context (neither alone suffices), and it is why L3 performance must never be quoted as a model result. The "clean" and "noisy" targets are identical because the simulator's `noise_level` is 0.0, so the registered noise sensitivity is vacuous here (logged in the Deviations Log)."""),
])

write("notebooks/C5_calibration.ipynb", [
md("""# C5 — Calibration of every Study C model

Raw Brier and ECE (locked: 10 equal-width bins) plus an **adaptive ECE (15 equal-mass bins, added** — at a 0.14% base rate almost all mass sits in the first equal-width bin). Platt and isotonic recalibration fitted on v2-fold validation predictions of the same configuration (seed 0), applied to every test seed. Ranking objectives (C2) have no raw probabilities, so only recalibrated rows exist for them. Design: Study C addendum 2 (commit 59bd4b1).

Reads saved outputs; run: `python -m src.models.c5_calibration` (~5 min)."""),
SETUP,
code("""r = pd.read_csv("outputs/c5_results_by_seed.csv")
m = r.groupby(["model", "method"])[["brier", "ece", "adaptive_ece", "pr_auc", "mean_prob"]].mean()
m.round(6)"""),
md("Base rate of the test set: 309 / 226,463 = 0.001364. A model that always predicts the base rate has Brier ≈ 0.001362."),
md("## Platt vs raw Brier (paired bootstrap; positive = Platt better)"),
code('pd.read_csv("outputs/c5_platt_vs_raw_brier.csv", index_col=0)'),
md("## Reliability diagrams (seed-mean scores, 15 equal-mass bins, log axes)"),
*[code(f'Image("outputs/c5_reliability_{s}.png")') for s in ("c1r", "c2", "c3", "c4")],
md("""**Reading.**
- Unweighted models are already close to calibrated: raw Brier 0.00135–0.00141 vs 0.00136 for a constant base-rate prediction — at this base rate Brier barely separates models; adaptive ECE and the diagrams are more informative.
- **Platt helps where raw scores are visibly off:** decision tree, random forest, CatBoost, TabNet and XGBoost-binary (C2 config) improve significantly; LR, C1-R XGBoost, LightGBM, FT-Transformer and MLP do not (CIs include 0).
- **Class-weighted / undersampled models are badly miscalibrated raw** (mean predicted 2–20% vs 0.14% observed) and Platt fully repairs them (Brier 0.0637 → 0.00135).
- Platt turns ranking-objective scores (LambdaMART, pairwise) into probabilities as good as the pointwise models'.
- **Isotonic lowers PR-AUC** (it creates ties: LR 0.0235 → 0.0201) and is worse than Platt on Brier for the deep models; Platt is the recommended calibrator.
- Caveat: calibrators are learned from a fold model trained on less data than the test model."""),
])

write("notebooks/C6_statistical_rigour.ipynb", [
md("""# C6 — Statistical comparison across Study C

All 15 final Study C models (C1-R 6, C2 6, C3 3) plus the B2 baselines CVSS v3 and point-in-time EPSS: 10 seeds each. Global test metrics; monthly MAP (primary), P@10, NDCG@10, NDCG@100, MRR over 46 monthly batches; Friedman + Nemenyi with critical-difference diagrams; Wilcoxon vs the best mean-rank model and vs EPSS-PIT, Holm-adjusted. Design: Study C addendum 2 (commit 59bd4b1).

Reads saved outputs; run: `python -m src.models.c6_statistics` (~3 min). Note: `scikit-posthocs` 0.16 `critical_difference_diagram` overwrites the p-value matrix it is given with 0/1 flags — the code passes a copy (found and fixed during this run)."""),
SETUP,
md("## Global test metrics (10 seeds, mean ± std)"),
code('pd.read_csv("outputs/c6_global_summary.csv", index_col=0)'),
md("## Friedman across 17 methods (46 months)"),
code('pd.read_csv("outputs/c6_friedman.csv", index_col=0)[["statistic", "p_value", "n_months", "best_mean_rank"]]'),
md("## Critical-difference diagrams"),
*[code(f'Image("outputs/c6_cd_{m}.png")') for m in ("map", "ndcg_at_100", "precision_at_10")],
md("## Wilcoxon vs best mean-rank model and vs EPSS-PIT (Holm-adjusted per metric)"),
code("""w = pd.read_csv("outputs/c6_wilcoxon_holm.csv")
w[w.metric == "map"].round(4)"""),
code('w[(w.reference == "baseline:epss_pit") & (w.p_holm < 0.05)].round(4)'),
md("""**Reading.**
- Methods differ (Friedman p < 1e-4 on MAP, P@10, NDCG@10, NDCG@100; MRR p = 0.08).
- **Top tier is flat.** On monthly MAP, logistic regression has the best mean rank (6.5), but after Holm correction the nine gradient-boosted classifiers and LTR objectives are **not distinguishable from it**. Significantly worse than LR: decision tree, random forest, XE-NDCG, FT-Transformer, TabNet and CVSS v3.
- **Nothing beats point-in-time EPSS month by month** after Holm correction, on any of the five metrics. The global PR-AUC advantage of LR over EPSS-PIT (C1-R, +0.0060 [0.0013, 0.0121]) does not survive as a per-month ranking advantage once 16 comparisons are corrected.
- CVSS v3 is in the bottom group on every metric — the H1 conclusion of Study B holds across all 15 models.
- Nemenyi (17 methods, 46 blocks) is conservative; the diagrams show wide cliques. Wilcoxon-Holm is reported alongside, never instead."""),
])
