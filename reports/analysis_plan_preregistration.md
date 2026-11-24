# Pre-Registered Analysis Plan

**Experiment:** A6 · **Registered:** 2026-09-09 · **Before:** any Study B model is trained

Written after A6's EDA and before any modelling, per CLAUDE.md rule 13. Its
purpose is to state what will be tested and what would count as a negative
result *before* seeing any model output, so that a disappointing result cannot
be quietly reframed afterwards.

---

## Research questions

**SQ1** — Which ML methods are most suitable for classifying/prioritising
security findings?
**SQ2** — How can XAI make those predictions understandable to security
engineers?

---

## Hypotheses

| # | Hypothesis | Decides |
|---|---|---|
| H1 | A learned model beats CVSS-score ordering on PR-AUC and Precision@100 for KEV prediction (B2) | Whether ML adds anything over current practice |
| H2 | Gradient-boosted trees (XGBoost/LightGBM/CatBoost) beat logistic regression and single trees on PR-AUC (C1) | Model family choice for SQ1 |
| H3 | Learning-to-rank (LambdaMART) beats pointwise classification on NDCG@100 (C2) | Whether ranking objectives matter |
| H4 | Deep tabular models (TabNet, FT-Transformer) do **not** beat GBMs at this data scale (C3) | Whether complexity is justified |
| H5 | Performance decays as the test year moves further from the training year (B5) | Model maintenance burden |
| H6 | SHAP and LIME disagree materially on per-finding attributions (D5, Kendall's τ) | Reliability of explanations for SQ2 |

**Every one of these is allowed to fail.** H1 failing — no model beating CVSS —
is a publishable negative result and will be reported as the headline finding
if it occurs, not buried.

---

## Primary analyses

1. **B1 baselines** — CVSS v3 ordering, severity label, publication recency,
   EPSS-current ranking, random. Establishes the bar.
2. **B2 KEV prediction** — binary classification, temporal split at 2023-01-01,
   point-in-time EPSS only (see protocol §3).
3. **C1 model comparison** — 6 classical families × 10 seeds, Friedman +
   Nemenyi.
4. **C5 calibration** — Brier/ECE/reliability for every probabilistic model.
5. **D1/D2 explainability** — global SHAP + per-finding TreeSHAP/LIME.

## Features (fixed now)

**Permitted for B2:** CVSS base score, CVSS severity, CVSS version, CVE age at
split date, `public_exploit` (NVD `Exploit` reference tag), `has_patch`,
description-derived text features, **point-in-time EPSS only**.

**Forbidden for B2:** `epss_score` from the current snapshot; `kev_date_added`
and any field derived from it; anything computed after the split date.

## Label sources

- **L1/L2 experiments:** CISA KEV membership only. Never invented, never
  tool-severity-derived — A3 measured Prowler↔ScoutSuite agreement at
  κ = 0.179, so tool severity is not a trustworthy label.
- **L3 experiments:** the documented 60/40 formula, reported separately and
  never described as ground truth.

### KEV label timing under a temporal split (registered 2026-11-24, before B1)

`in_kev` is KEV membership *today*. 174 of the 1,038 training-period KEV CVEs
(published before 2023-01-01) were added to KEV only after that date.

- **Primary:** training labels are KEV **as of the split date**
  (`kev_date_added < T`; 864 positives); test labels are KEV as of the data
  snapshot (752 positives). This is exactly what a model built at `T` could
  have known.
- **Sensitivity:** training labels are KEV as of today (1,038 positives).
  Reported alongside the primary result, never in place of it.
- Implemented once in `src/evaluation/labels.py` (`temporal_labels`); every
  Study B experiment must use it.
- **Consequence to state in the write-up:** under the primary rule the 174
  late-added CVEs are training *negatives*. KEV's lag is therefore label noise
  in training, which bounds what B2 can achieve (see threat 5).

## Stopping and reporting rules

- Metrics are those in the locked protocol; no metric will be introduced after
  seeing results in order to make a model look better.
- Model selection is on PR-AUC alone.
- Minimum 10 seeds, mean ± std, everywhere.
- Any deviation from this plan is recorded in the CLAUDE.md Deviations Log with
  its reason and the date.

---

## Known threats to validity (stated up front)

1. **EPSS point-in-time coverage** — daily archives verified back to 2021; CVEs
   published before then have no contemporaneous EPSS and will be missing that
   feature rather than back-filled.
2. **The L3 label is partly our own function.** A model recovering it is not
   evidence the approach works on real bank data.
3. **L1/L2 scale mismatch in the L3 label** (A5) — resolved 2026-09-09 by the
   label amendment (evaluation protocol §7), but the residual `source_layer`
   signal (AMI ≈ 0.09) is not zero; L3 results must report it alongside any
   feature-ablation finding.
4. **No real enterprise data exists in this project.** No L3 result may be
   presented as a measurement of Deutsche Bank's estate.
5. **KEV is a lagging, partial ground truth** — it records exploitation that was
   *observed and published*, so absence of KEV membership is not evidence of
   non-exploitation. This bounds the ceiling on any B2 result.
