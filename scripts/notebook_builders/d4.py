from common import *

write("notebooks/D4_interpretable_models.ipynb", [
md("""# D4 — Interpretable models: the price of interpretability (descriptive)

Glass-box models trained on the same L1 data, features, temporal split and v2 protocol as C1-R:
- **EBM:** `interpret` 0.7.8, handles missing values natively.
- **RuleFit:** `imodels` 3.0.4.
- **Sparse optimal tree:** GOSDT 1.0.4, `balance=True`, threshold-guess binarisation with 200 depth-2 stumps. Each leaf is scored by its training KEV rate, so the tree can rank.

RuleFit and GOSDT get median imputation plus missing indicators, fitted on training rows.

Hyperparameters were searched on the v2 fold (seed 0, one fit per config) and selected by validation PR-AUC. The selected config was then refitted on the full training set with 10 seeds and tested once. Each model is compared with C1-R LR, C1-R XGBoost and EPSS-PIT by a paired bootstrap on the seed-mean scores. Design: Study D addendum (commit c0a3ee1). **No hypothesis**: D4 is descriptive.

Reads saved outputs. Run: `python -m src.models.d4_interpretable --stage all` (search about 1 h, final about 5 h on this machine)."""),
SETUP,
md("## (1) Search on the v2 fold"),
code('pd.read_csv("outputs/d4_search.csv").drop(columns=["fit_seconds"]).round(4)'),
md("""**The depth-4 GOSDT configs failed.** GOSDT's memory use grew until swap filled a nearly full disk (`OSError: No space left on device`), and the process died without a traceback. They are recorded as failed and, by the rule fixed then, not retried. Selection is therefore among depth-3 trees only, and the GOSDT result below is bounded at depth 3. Since then, every GOSDT search fit runs in its own subprocess.

The class-weighted CART fallback was not triggered, because both depth-3 configs returned trees with more than one leaf."""),
code('pd.read_csv("outputs/d4_selected_configs.csv", index_col=0)[["config", "val_pr_auc"]]'),
md("## (2) Test performance, 10 seeds (mean ± std)"),
code('pd.read_csv("outputs/d4_summary.csv", index_col=0)'),
md("## (3) Paired bootstrap vs C1-R and EPSS-PIT (seed-mean scores; diff = D4 model − reference)"),
code('pd.read_csv("outputs/d4_tests.csv", index_col=0).round(4)'),
md("## (4) Model size"),
code("""s = pd.read_csv("outputs/d4_model_size.csv")
s.groupby("kind").agg(fit_minutes=("fit_seconds", lambda x: round(x.mean() / 60, 1)), n_terms=("n_terms", "mean"),
                      n_rules=("n_rules", "mean"), n_linear=("n_linear", "mean"), n_leaves=("n_leaves", "mean"))"""),
md("## (5) EBM global term importance, and its agreement with D1's rankings (Kendall τ-b, 69 features)"),
code('pd.read_csv("outputs/d4_ebm_term_importance.csv", index_col=0).sort_values("importance", ascending=False).head(12).round(3)'),
code('pd.read_csv("outputs/d4_global_ranking_agreement.csv", index_col=0).round(3)'),
md("""**Reading.**
- **EBM (PR-AUC 0.0187) and RuleFit (0.0196) perform like the black boxes.** Every paired-bootstrap CI against C1-R LR, C1-R XGBoost and EPSS-PIT includes zero. Point estimates are slightly below C1-R (−0.001 to −0.005) and slightly above EPSS-PIT (+0.001 / +0.004). P@100 is 0.10 for both.
- **There is no price of interpretability on this task, but not for the usual reason.** The best model in Study C, logistic regression, is already a glass box. EBM and RuleFit add shape functions and rules without adding ranking performance. What they offer is a different *form* of explanation, such as per-feature shape plots and if-then rules. D6 is meant to test whether that form helps engineers.
- **A depth-3 optimal tree cannot rank 1-in-240 events.** It has 3 leaves, PR-AUC 0.0027 and ROC 0.63, and is significantly worse than every reference. With only three distinct scores it cannot order findings within a month. This conclusion is bounded at depth 3 (see the incident in (1)).
- **EBM's own importance repeats the SHAP background artefact.** `cvss_version` is again the top term (4.37, against about 0.58 for each EPSS feature). EBM's importance is the mean |term score| around the *training* intercept, so it inherits the same era shift that dominated training-background SHAP in D1 and D2. It agrees best with SHAP M-XGB (τ 0.42) and worst with permutation importance (τ 0.09–0.18). Two independent attribution methods share the artefact; the permutation-based ranking does not.
- RuleFit used 33 rules plus 25 linear terms on average, and seeds vary in which rules survive. A rule list that changes from seed to seed is a stability concern for presenting rules to analysts."""),
])
