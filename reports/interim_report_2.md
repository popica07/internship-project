# Interim Report 2 — Studies A, B and C

**Project:** Cloud Security Finding Prioritisation — ML & XAI
**Student:** Popa Stefan Andrei (6415250) · Master AI, Maastricht University DACS
**Supervisors:** Mikhail Glazunov (UM DACS) · Romulus-Cristinel Viesanu (Deutsche Bank CSO)
**Covers:** Study A (data), Study B (exploitation prediction), Study C (model comparison)
**Date:** 2026-11-25 · PDF version: `../Progress_Report_Studies_A_B_C.pdf` (built by `reports/pdf/`)

Studies A and B are reported in detail in `interim_report_1.md`; this report
summarises them and adds Study C.

---

## 1. Summary

Machine learning ranks exploited vulnerabilities clearly better than sorting by
CVSS (current practice). A simple, transparent **logistic regression is as good
as anything more complex**: gradient boosting, learning-to-rank objectives and
deep networks add no measurable value. The signal comes mostly from
**point-in-time EPSS**; the best model beats EPSS on overall PR-AUC, but not
month by month once multiple comparisons are corrected.

All hypotheses and analysis plans were pre-registered before the corresponding
models ran; every result is the mean of 10 seeds on one untouched test set
(309 exploited CVEs among 226,463).

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Learned model beats CVSS ordering | **Supported** | B2: PR-AUC +0.0095 [0.0033, 0.0227], P@100 +0.08 [0.02, 0.15]; holds for all 15 Study C models |
| H1b | Best model beats point-in-time EPSS (PR-AUC and P@100) | **Partly** | C1-R (post-hoc): PR-AUC +0.0060 [0.0013, 0.0121], P@100 +0.01 [−0.03, 0.05]; C6: no model better month by month after Holm |
| H2 | Gradient boosting beats LR and single trees | **Not supported** | XGBoost − LR −0.0013 [−0.0070, 0.0031] |
| H3 | LambdaMART beats pointwise on NDCG@100 | **Not supported** | monthly NDCG@100 +0.0049 [−0.0061, 0.0158] |
| H4 | Deep tabular models do not beat GBMs | **Supported** (no difference) | FT-Transformer − XGBoost +0.0003 [−0.0043, 0.0087] |
| H5 | Performance decays over time | **Partly** | consistent direction, ρ = −0.53, p = 0.14 |
| H6 | SHAP and LIME disagree | Pending | Study D |

## 2. Studies A and B (recap)

- **Data:** L1 real (NVD + EPSS + KEV, 430,678 CVEs, 1,790 KEV), L2 real
  (3,870 misconfiguration checks), L3 simulated estate (39,989 findings),
  reported separately.
- **Tool severity is not ground truth** (Prowler vs ScoutSuite κ = 0.179), so
  labels come only from CISA KEV. **Present-day EPSS leaks** the outcome
  (49.5× vs 5.7× point-in-time) and is banned as a feature.
- **B2:** H1 supported vs CVSS; not better than point-in-time EPSS; LR beat the
  GBM. **B4:** monthly ranking differences significant (Friedman). **B5:** decay
  consistent but underpowered. **B6:** under a fixed monthly budget the model
  catches ~2× the exploited CVEs CVSS catches.

## 3. Study C — model comparison

Same data, features, labels and test set as B2. Hyperparameters selected on a
time-ordered validation fold only; 10 seeds; test touched once per model.

### C1 / C1-R — classical families and model selection

C1 tuned six families on a 2022 validation year with **39 positives**;
validation rank did not predict test rank (LightGBM best on validation, worst
GBM on test). C1's verdict stands (H2 not supported). A revised protocol
(v2: wider window, **145 positives**) was pre-registered and re-run (C1-R,
post-hoc for C1, binding for C2–C6). Validation-vs-test Spearman over all 126
configurations rose from 0.68 to 0.78 (LightGBM 0.16 → 0.68, XGBoost
0.40 → 0.77).

| Test PR-AUC (10 seeds) | |
|---|---|
| Logistic regression | **0.0235** |
| XGBoost | 0.0220 ± 0.0002 |
| FT-Transformer (C3) | 0.0210 ± 0.0017 |
| LambdaMART, LightGBM (C2) | 0.0207 ± 0.0009 |
| Point-in-time EPSS | 0.0175 |
| CVSS v3 | 0.0068 ± 0.0014 |

### C2 — learning-to-rank (H3 not supported)

Six boosted models differing only in objective (pointwise, pairwise/RankNet,
LambdaMART ×2, XE-NDCG); each month's CVEs form one query. LambdaMART vs
pointwise: +0.0049 [−0.0061, 0.0158] monthly NDCG@100, Wilcoxon p = 0.68.
About seven exploited CVEs per monthly queue leave little list structure.

### C3 — deep tabular models (H4 supported)

MLP, TabNet, FT-Transformer. FT-Transformer ≈ XGBoost (+0.0003 [−0.0043,
0.0087]) at ~100× the compute and with the largest seed variance; MLP is
significantly below LR.

### C4 — features and imbalance

- Without point-in-time EPSS every model falls to CVSS level; CVSS + EPSS ≈
  all features (−0.0023 [−0.0068, 0.0012]); text adds little.
- Class weighting / undersampling never improve ranking significantly and
  wreck calibration (Brier 0.0014 → 0.064): train unweighted.
- L3 (simulated): finding-only 0.39, context-only 0.39, both 0.998 PR-AUC —
  the model recovers the documented label formula; not evidence about a real
  estate.

### C5 — calibration

Unweighted models are near-calibrated raw. Platt scaling significantly
improves decision tree, random forest, CatBoost and TabNet, repairs weighted
models and converts ranking scores to probabilities; LR needs none. Isotonic
lowers PR-AUC. An adaptive (equal-mass) ECE was added next to the locked ECE.

### C6 — statistical comparison (17 methods, 46 months)

Friedman p < 1e-4 on MAP, P@10, NDCG@10, NDCG@100. LR has the best mean rank
on MAP; after Holm it is indistinguishable from the other boosted/LTR models,
the MLP and point-in-time EPSS, and significantly better than the decision
tree, random forest, XE-NDCG, FT-Transformer, TabNet and CVSS v3. No model
beats point-in-time EPSS month by month on any metric after Holm.

## 4. Rigour and deviations

All deviations are in the CLAUDE.md Deviations Log. Study C incidents, none
changing a registered decision: an MLflow naming error (C4 re-run);
torch/OpenMP import-order crash (fixed); `scikit-posthocs` overwriting its
p-value matrix when drawing CD diagrams (caught, fixed, C6 re-run); L3 label
noise is zero, so the registered noise check was vacuous; smaller deep-model
search budgets fixed in advance from a speed-only benchmark.

## 5. Next — Study D

Pre-register D1–D6 first. Proposed primary model: logistic regression (top of
the tied group, interpretable), with XGBoost for TreeSHAP comparison.

**For discussion with supervisors:** framing "simple models are enough; ML
beats CVSS and matches but does not clearly beat EPSS" as the headline; the L3
label amendment; LR as Study D's primary model; scope and timing of a pilot
user study with CSO engineers.
