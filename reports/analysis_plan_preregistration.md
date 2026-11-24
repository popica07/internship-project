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

---

## Study B design addendum (registered 2026-11-24, after B1, before any B2–B6 model)

Fixed before any learned model was trained. B1 (fixed orderings, no training)
had run; nothing below was tuned on a model result.

**Scoring date (point-in-time features).** A model built at split date `T`
scores each CVE as it would have been scored in practice:
- training CVEs (published < `T`): features as of `T`;
- test CVEs: features as of the first monthly EPSS snapshot on or after
  publication (1st of each month, 2021-05 onward; the 2026-11-23 download for
  the last weeks). Mean lag ≈ 15 days.

**Evaluation set for B2–B4/B6.** Test CVEs already in KEV on their scoring date
are excluded: an analyst would already know, so there is nothing to predict.
The number excluded is reported. Every scorer, B1 baselines included, is
re-evaluated on this same set, so all comparisons are like for like.

**Features — primary (strict point-in-time):** CVSS base score, CVSS severity
(ordinal), CVSS version, description text (TF-IDF → 64-d SVD, fitted on
training text only), point-in-time EPSS score and percentile.
- `public_exploit` / `has_patch` are moved to a **sensitivity** run. They come
  from NVD reference tags *as of today*, and NVD adds `Exploit` tags
  retroactively, often after exploitation is public — the same leakage class as
  EPSS-current. Stricter than the original plan, decided before any result.
- "CVE age at split date" is **dropped**: training ages span decades while test
  ages span days, so the feature can only encode KEV lag, not risk.
- Residual threat, stated not fixed: CVSS scores and descriptions are also
  today's versions; NVD rarely rescores, so this is minor.

**Models.** B2 primary: `HistGradientBoostingClassifier` (scikit-learn GBM; the
LightGBM build needs `libomp`, absent on this machine — LightGBM/XGBoost/
CatBoost belong to C1 anyway). Reference: L2-regularised logistic regression.
No class re-weighting, so probabilities stay calibratable for B6. 10 seeds.

**Labels.** As registered above: train = KEV as of `T`; test = KEV as of the
data snapshot. Sensitivity: train = KEV as of today.

**B3 (EPSS regression).** Target: logit of EPSS. Training target = EPSS as of
`T` (known at `T`); EPSS features excluded (it is the target). Evaluated as a
regression (Spearman ρ against today's EPSS on test) and as a KEV ranker.

**B4 (ranking).** Every scorer ranks each *monthly* batch of test CVEs — the
queue an analyst actually works. Metrics per month (P@10, NDCG@10, MAP, MRR);
Friedman across scorers with months as blocks, Nemenyi post-hoc, Wilcoxon for
B2 vs CVSS v3. Months with no KEV positive are excluded (metrics undefined).

**B5 (temporal drift).** Train on publication year `N`, test on `N+1..N+3`,
`N ∈ {2018,…,2021}`. To separate drift from KEV lag (B1 finding), labels use a
**fixed 365-day horizon**: positive iff added to KEV within 365 days of
publication, for train and test alike; only test years whose every CVE has
had 365 days (≤ 2024) are used. EPSS is excluded from B5 features (no
archive before 2021, so it would be absent for most `N`). H5 is tested here.

**B6 (cost-sensitive).** No bank cost data exists, so the cost ratio
`C_FN / C_FP` is swept over {10, 50, 100, 500}. Thresholds are chosen on a
validation year (train < 2022-01-01, validate on 2022, KEV as of 2023-01-01),
never on test. Compared policies: B2 model at its chosen threshold, CVSS ≥ 7,
CVSS ≥ 9, remediate-all, remediate-none; plus a fixed monthly remediation
budget (top-k per month). Calibration (Brier, ECE) of B2 is reported first.

**Deviation (2026-11-24, before B5 ran — a data property, not a result):** KEV
was launched on 2021-11-03, so a 365-day-horizon label is 0 for *every* CVE
published in 2016–2019 (14 positives in 2020). Training years 2018–2021 would
train on no positives. B5 therefore uses `N ∈ {2021, 2022, 2023, 2024}`, test
years `N+1..N+3` up to 2025, where 2025 is restricted to CVEs published by
2025-11-23 (the last date with a complete 365-day horizon). 2021 itself is
partially affected (KEV did not yet exist for its first ten months) — stated
as a caveat on the `N = 2021` row.
