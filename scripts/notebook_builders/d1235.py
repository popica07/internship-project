from common import *

DESIGN = "Design: Study D addendum (commit c0a3ee1), registered before any Study D explanation was computed."

write("notebooks/D1_global_xai.ipynb", [
md(f"""# D1 — Global explainability: SHAP, permutation importance, ALE

Models explained: **M-LR** (C1-R logistic regression) and **M-XGB** (C1-R XGBoost, seed 0), refitted exactly as C1-R. The refit reproduces C1-R's cached test predictions, and the run aborts if it doesn't. SHAP is in log-odds: interventional TreeSHAP for XGBoost, exact linear SHAP for LR, with a shared 1,000-row training background, on 10,000 random test CVEs. Permutation importance is the PR-AUC drop on the full test set, 10 repeats. ALE is first-order, 20 quantile bins, on 20,000 test rows. Part (5) is an explanation-correctness check on **L3 (SIMULATED)**, where the true label formula is known. {DESIGN}

Reads saved outputs. Run: `python -m src.explainability.shap_global` (~12 min)."""),
SETUP,
md("## Text components: what they mean (top TF-IDF terms of each SVD component)"),
code('pd.read_csv("outputs/d_text_component_terms.csv", index_col=0).head(25)'),
md("Component 0 and several others encode **rejected / withdrawn CVE boilerplate** (\"candidate … rejected reason\"). They are data-quality signals, not vulnerability semantics."),
md("## (1) SHAP: share of total mean |SHAP| by feature group, and top features"),
code('pd.read_csv("outputs/d1_shap_group_share.csv", index_col=0).round(3)'),
code('pd.read_csv("outputs/d1_shap_mean_abs.csv", index_col=0).sort_values("M-XGB", ascending=False).head(15).round(4)'),
code('Image("outputs/d1_beeswarm_xgboost.png")'),
code('Image("outputs/d1_beeswarm_logreg.png")'),
md("## (2) Permutation importance (PR-AUC drop, 10 repeats, full test set)"),
code("""p = pd.read_csv("outputs/d1_permutation_importance.csv")
p[p.unit.str.startswith("GROUP:")].round(4)"""),
code("""p[~p.unit.str.startswith("GROUP:")].sort_values(["model", "pr_auc_drop_mean"], ascending=[True, False]).groupby("model").head(8).round(4)"""),
md("## (3) Accumulated local effects"),
code('Image("outputs/d1_ale.png")'),
md("## (4) Do the global rankings agree? (Kendall τ-b over 69 features)"),
code('pd.read_csv("outputs/d1_global_ranking_agreement.csv", index_col=0).round(3)'),
md("""**Reading (L1).**
- **The two global methods tell different stories.** Permutation importance, i.e. what the model's *ranking* actually depends on, says **EPSS-PIT**: permuting the two EPSS features removes 0.017–0.019 of 0.022–0.024 PR-AUC, about 80%. CVSS removes about 0.0066, and text 0.0046 (M-XGB) to 0.0087 (M-LR); group drops are not additive. This agrees with the C4 ablation. Mean |SHAP| gives EPSS only 14–15% of attribution mass, while CVSS gets 59% (M-XGB) and text 45% (M-LR).
- **Why: `cvss_version` dominates SHAP** (M-XGB 1.80, M-LR 3.34 mean |φ|, rank 1 for both) but matters little to the ranking (permutation drop 0.002 / 0.0005). Test CVEs are 90% CVSS v3.1, while the training background is 36% v2.0. SHAP therefore attributes a large, nearly constant amount to "this is a v3 CVE", measured against a background from another era. That shifts every test CVE about equally and barely changes the order. The exploratory background check (`DX` section in D5) quantifies it.
- Rank agreement is correspondingly low: SHAP vs permutation τ ≈ 0.17 for both models. Even permutation importance agrees only τ = 0.29 across the two models, because the many weak text features rank noisily.
- **ALE:** M-XGB is flat in EPSS below about 0.008 and jumps above it, a threshold rule. M-LR is linear in the EPSS percentile, which is where its signal sits. Both rise with CVSS score; XGBoost does so in one step at about 7.5."""),
md("## (5) L3 explanation-correctness check — SIMULATED (true formula known)"),
code("""t = pd.read_csv("outputs/d1_l3_shap_vs_formula.csv", index_col=0)
share = t.loc[t.in_formula, "mean_abs_shap"].sum() / t.mean_abs_shap.sum()
print(f"share of mean |SHAP| on features the formula uses: {share:.3f}")
t.round(4)"""),
md("""**Reading (L3, simulated).** The model fits the formula almost perfectly (held-out PR-AUC 0.9995). SHAP puts 86% of its mass on features the formula really uses, and its importance ranking agrees with the formula's own term importance at τ = 0.61. Two errors are instructive:
- **`epss_score` (not in the formula) gets about twice the attribution of `epss_percentile` (in the formula).** The two are almost perfectly rank-correlated, and SHAP credits whichever one the trees happened to split on. **SHAP explains the model, not the data-generating process.**
- The scale of SHAP's attributions does not track the formula's weights: `sla_breached` and `internet_facing` rank above `epss_percentile`. This is because SHAP measures effect on the log-odds of crossing a threshold, not the linear score.

An analyst reading SHAP here would get the right *set* of drivers, attribute one of them to the wrong correlated proxy, and weight them differently from the true rule."""),
])

write("notebooks/D2_local_xai.ipynb", [
md(f"""# D2 — Local explainability: TreeSHAP and LIME per finding

SHAP (log-odds) and LIME (`LimeTabularExplainer`: continuous features, `sample_around_instance=True`, 5,000 samples, all 69 features, explaining P(KEV)) for both models, on E-rand (1,000 random complete-case test CVEs) ∪ E-top (top-10 per month, 460) ∪ E-kev (308 KEV positives). The nine case studies follow a fixed rule: from E-top under M-XGB, the three highest-ranked KEV hits and the three highest-ranked non-KEV CVEs; from E-kev, the three KEV CVEs ranked lowest in their month. {DESIGN}

Reads saved outputs. Run: `python -m src.explainability.shap_local` (~3 min). Attributions are cached for D5."""),
SETUP,
md("## How well does LIME's local surrogate fit? (local R²)"),
code('pd.read_csv("outputs/d2_lime_r2.csv").round(3)'),
md("LIME's linear surrogate explains only **about 37% of the variance** of M-XGB around a typical CVE (median R² 0.37), and 0.43 in the top-10. It fits M-LR better: 0.69 in the top-10, where the logistic curve is steepest."),
md("## Case studies"),
code('pd.read_csv("outputs/d2_case_studies.csv")'),
code('Image("outputs/d2_case_studies.png")'),
md("""**Reading.**
- **Top-ranked CVEs** (KEV hits and non-hits look alike): SHAP says high EPSS-PIT (+2.7 to +3.1 log-odds), then `cvss_version` (+2.6 to +2.7), then the EPSS percentile. The model cannot tell the three KEV hits from the three non-hits, and neither can the explanation. One "top-ranked, not in KEV" CVE (CVE-2025-29927) is widely reported as exploited, which is a reminder that KEV is a partial label.
- **KEV CVEs the model ranks low:** SHAP says low EPSS at scoring time (−0.4) is the reason, and `cvss_version` (+1.85) is again the biggest attribution. At scoring time these CVEs looked like everything else; the information was not there yet.
- **SHAP and LIME read differently at the top of the list.** LIME ranks `cvss_version` first in most cases, and it can flip a sign: for the low-ranked KEV CVEs, SHAP gives EPSS −0.40 (below the background), while LIME gives EPSS a positive weight (raising EPSS locally would raise the score). The two methods answer different questions, deviation from a baseline versus local slope, and a reader is not told that. D5 measures this systematically.
- **`cvss_version` dominates both explainers.** Under SHAP it is a background artefact (see D1). Under LIME it is a perturbation artefact: sampling with training-scale noise pushes the instance toward v2.0, where the model's score falls steeply."""),
])

write("notebooks/D3_counterfactuals.ipynb", [
md(f"""# D3 — Counterfactual explanations

**L3 (SIMULATED, the primary part of D3, because only L3 has actionable context):** DiCE ("random", 4 counterfactuals) for 100 held-out findings that the L3 model flags High/Critical. Only remediation levers may vary: `internet_facing`, `has_admin_path`, `admin_principal_count`, `principal_count`. Each counterfactual is checked for validity under the model **and under the true label formula**, using a re-scorer that reproduces `priority_score` to 1.7e-16.

**L1:** a CVE's CVSS score and description never change, so the only meaningful what-if is EPSS. For each E-rand CVE outside its month's top-100, the minimum point-in-time EPSS that lifts its M-XGB score to τ₁₀₀ is found by a deterministic search over 200 log-spaced values. The EPSS percentile follows from that scoring date's empirical EPSS→percentile map. {DESIGN}

Reads saved outputs. Run: `python -m src.explainability.counterfactuals` (~2 min)."""),
SETUP,
md("## L3 — SIMULATED: DiCE with remediation levers"),
code('pd.read_csv("outputs/d3_l3_summary.csv").T'),
code("""cf = pd.read_csv("outputs/d3_l3_counterfactuals.csv")
cf.changed.value_counts()"""),
md("""**Reading (L3, simulated).**
- **36 of 100 flagged findings can be de-escalated by a lever an engineer controls**, and every such counterfactual removes internet exposure. Only 46 of the 100 were internet-facing at all, so about 78% of those can be fixed this way. For the rest no lever in the set changes the label, which is correct: the formula uses no IAM feature.
- **Validity under the true formula: 96.5%.** Model and formula agree on almost every counterfactual. The one IAM-only counterfactual the model accepted is **invalid under the formula**. This is exactly the model artefact predicted in the pre-registration: the model learned a tiny spurious dependence on IAM features.
- **Sparsity failure (exploratory check, added after a 5-finding pilot):** 99.3% of valid counterfactuals change `internet_facing` *plus* an irrelevant IAM count, even though flipping `internet_facing` alone would have been enough. DiCE's random method does not minimise the number of changes, so its suggestions over-prescribe work. A real recommendation needs a minimality post-processing step."""),
md("## L1 — what EPSS would put this CVE in the monthly top-100?"),
code('pd.read_csv("outputs/d3_l1_summary.csv").T'),
code("""t = pd.read_csv("outputs/d3_l1_epss_triggers.csv")
t[["epss_now", "epss_needed", "pct_needed", "uplift_ratio"]].describe().round(4)"""),
md("""**Reading (L1).** τ₁₀₀ = 0.0065. For 99% of the CVEs outside the top-100, *some* EPSS value would lift them in. The median trigger is **EPSS ≈ 0.021**, about the 84th percentile on its scoring date, against a current median of 0.0005, a 28× rise. This gives a concrete, monitorable statement per CVE: "escalate if EPSS crosses 0.02". It is a natural use of counterfactuals when the finding itself cannot be changed, only watched."""),
])

write("notebooks/D5_stability.ipynb", [
md(f"""# D5 — Stability and inter-explainer agreement (H6)

**H6 (registered: SHAP and LIME disagree materially on per-finding attributions)** was operationalised in advance. For each E-rand CVE, compute Kendall's τ-b between |TreeSHAP| and |LIME| over all 69 features (M-XGB, seed 0). Take the median over 1,000 CVEs, with a 95% bootstrap CI (2,000 resamples). **Supported if the CI's upper bound is < 0.5**, not supported if its lower bound is ≥ 0.5, inconclusive otherwise.

Noise ceilings:
- LIME against itself across 10 seeds (first 200 E-rand CVEs);
- TreeSHAP across 10 XGBoost training seeds;
- a Wilcoxon test of SHAP–LIME τ against LIME–LIME τ.

{DESIGN}

Reads saved outputs. Run: `python -m src.explainability.stability` (after D2, ~5 min)."""),
SETUP,
md("## H6"),
code('pd.read_csv("outputs/d5_h6.csv").T'),
md("## SHAP vs LIME on every set and model"),
code('pd.read_csv("outputs/d5_shap_vs_lime.csv").round(3)'),
md("## Noise ceilings"),
code('pd.read_csv("outputs/d5_noise_ceilings.csv").round(4)'),
md("""**H6 SUPPORTED as registered:** median τ = **0.350 [0.344, 0.354]**, entirely below 0.5. The same holds on E-top (0.347), E-kev (0.354) and for M-LR (0.31–0.41).

**What the noise ceilings add, and why it matters for the write-up:**
- **Over all 69 features, LIME barely agrees with itself:** τ = 0.366 across LIME seeds on the same model and instance. SHAP–LIME (0.341 on the same 200 CVEs) is significantly lower (Wilcoxon p < 1e-10) but only slightly. Most of the full-ranking disagreement is therefore **LIME's sampling noise in the long tail of weak features**, not a systematic conflict. TreeSHAP is more stable across *model* retraining (τ = 0.66).
- **At the top of the explanation, the part an analyst reads, the disagreement is systematic.** LIME's own top-5 is stable across seeds (88% overlap), and TreeSHAP's top-5 is stable across model seeds (93%). Yet SHAP and LIME share only **54%** of their top-5 features, and only **25%** of features are shared *with the same sign*. Two stable methods give different answers.
- Reason: the two methods answer different questions. SHAP attributes the deviation from a background expectation; LIME fits the local slope of P(KEV). A low-EPSS CVE gets a *negative* SHAP value for EPSS (below the background) and a *positive* LIME weight (more EPSS would raise its score). Both are "correct", and neither says which question it answers.
- Group-level agreement (CVSS vs text vs EPSS) is high for M-XGB on E-rand (median τ = 1.0) but not in the top-10 (0.33)."""),
md("## EXPLORATORY (not pre-registered): SHAP background sensitivity"),
md("""Added after the D2 case studies showed `cvss_version` dominating every explanation. SHAP on E-rand was recomputed with a **test-era background** (1,000 random test rows, no labels used). The pre-registered results above stand."""),
code('pd.read_csv("outputs/dx_background_sensitivity_summary.csv", index_col=0).round(4)'),
md("""- `cvss_version` takes **51% of all |SHAP| mass for M-XGB** (38% for M-LR) under the training background, against **15%** (4%) under a test-era background.
- The global ranking otherwise holds: τ ≈ 0.91 between backgrounds. Per-instance rankings move more (median τ 0.79 / 0.71).
- SHAP–LIME agreement stays low under the test-era background (median τ 0.32 / 0.29), so H6 does not hinge on the background choice.
- **Practical rule for deployment:** a SHAP background must come from the population being scored (recent CVEs), not from the training archive. Otherwise about half of every explanation describes the CVE's era."""),
])
