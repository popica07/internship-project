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

---

## Study C addendum — C1 design + Study B exploratory follow-ups (registered 2026-11-24, after Study B, before any Study C model)

Written after seeing Study B (so B results motivate it), but before any C1
model was fitted or tuned. Nothing below was chosen by looking at a C result.

### C1 — classical model suite

**Data, features, labels, evaluation set:** exactly B2's (split 2023-01-01,
`src/features.py` primary features incl. point-in-time EPSS, train labels KEV
as of T via `temporal_labels`, test CVEs already in KEV at their scoring date
excluded: 309 KEV of 226,463). C1 results are therefore directly comparable
with every B2 number. No class re-weighting (calibration matters for C5; class
weighting belongs to C4).

**Families (6):** logistic regression (L2), decision tree, random forest,
XGBoost, LightGBM, CatBoost. LR/DT/RF get median imputation + missing
indicators; the three GBM libraries use native missing handling.
(`libomp` was installed 2026-11-24, so the B2 restriction to scikit-learn GBM
no longer applies.)

**Tuning — new compared with B2, where nothing was tuned.** One time-ordered
validation fold, never touching test: fit on CVEs published < 2022-01-01
(labels KEV as of 2022-01-01: 309 positives), validate on CVEs published in
2022 (scoring dates and exclusions as in B2; labels KEV as of 2023-01-01:
39 positives, 26,372 CVEs). Random search, **20 configurations per family**,
sampled with seed 0 from the fixed spaces below, each fitted once (seed 0);
the configuration with the highest validation PR-AUC is selected. The selected
configuration is then refitted on all training data (< 2023-01-01) with 10
seeds and evaluated once on test. Default-hyperparameter results are reported
alongside, so the effect of tuning is visible.

*Stated threat:* 39 validation positives make selection noisy (B6 showed the
same for thresholds). The small budget (20) limits over-fitting to the fold;
no larger budget will be run after seeing results.

Search spaces (fixed now):

| family | space |
|---|---|
| LR | C log-uniform [1e-3, 10] |
| DT | max_depth 3–20; min_samples_leaf 10–500 (log); criterion {gini, entropy} |
| RF | 300 trees; max_depth {None, 10, 20, 30}; min_samples_leaf 1–50 (log); max_features {sqrt, 0.3, 0.5} |
| XGBoost | n_estimators 200–800; learning_rate log [0.01, 0.2]; max_depth 3–10; subsample [0.5, 1]; colsample_bytree [0.5, 1]; min_child_weight 1–20 (log); reg_lambda log [0.1, 10] |
| LightGBM | n_estimators 200–800; learning_rate log [0.01, 0.2]; num_leaves 15–127 (log); min_child_samples 10–200 (log); subsample [0.5, 1] (freq 1); colsample_bytree [0.5, 1]; reg_lambda log [0.1, 10] |
| CatBoost | iterations 200–800; learning_rate log [0.01, 0.2]; depth 4–8; l2_leaf_reg log [1, 10] |

No early stopping anywhere (a random validation carve-out of time-ordered data
is forbidden; tree count is a searched hyperparameter instead).

**H2 (as registered):** GBMs beat LR and single trees on PR-AUC. Operationalised
now: the GBM family with the best *validation* PR-AUC (chosen without seeing
test) is compared with LR and with DT by paired bootstrap over test CVEs
(seed-averaged scores, 500 resamples). H2 is supported if both PR-AUC
differences have 95% CIs above 0. All three GBM families are reported against
LR and DT as well, as secondary.

**H1b (new, confirmatory — the second bar after B2):** the best C1 model
(highest validation PR-AUC across all families) beats point-in-time EPSS on
PR-AUC *and* P@100, by the same paired bootstrap. The models receive EPSS as a
feature, so H1b asks whether anything adds value *on top of* EPSS. A failure is
reported as a headline negative result, exactly as H1 would have been.

**Family comparison (feeds C6):** per-month metrics on B4's monthly batches
(seed-averaged), Friedman across the 6 families with months as blocks, Nemenyi
post-hoc; critical-difference diagram in C6.

### Study B exploratory follow-ups (labelled exploratory, never replacing the primary B results)

**B6-R — threshold robustness.** The validated threshold was unstable (39
validation KEV). Exploratory alternative: the threshold minimising the *mean*
validation cost over 200 bootstrap resamples of the validation set (same
validation fold, same models, same ratios). Reported next to the primary
validated threshold and the Bayes reference; the primary B6 table is
unchanged.

**B5-M — monthly drift.** B5 had only 9 (N, test-year) cells. Exploratory
re-analysis of the same B5 models: per test *month* (CVEs grouped by
publication month), GBM PR-AUC minus CVSS v3 PR-AUC (tie-broken, 10 seeds);
months with < 3 positives dropped (count reported). Decay is summarised as
the slope of that margin on months since the end of the training year, per N,
with a bootstrap CI over months. The registered B5 result (ρ = −0.53,
p = 0.14) remains the H5 test.

**Deviation (2026-11-24, before B6-R ran — a property of the definition, not a
result):** as written, B6-R is identical to the primary B6 rule. Cost is
linear in the sample weights and a bootstrap resample has expected weight 1
per CVE, so the *mean* bootstrap cost of any threshold equals its cost on
the original validation set, and the argmin is the same threshold. B6-R is
therefore defined as the **median of the 200 per-resample cost-minimising
thresholds** (a bagged threshold). That keeps the stated intent, which is to
reduce dependence on the 39 validation positives. Everything else is
unchanged.

---

## Selection protocol v2 + C1-R (registered 2026-11-24, after C1, before any run below)

**Why.** C1 showed that the registered selection fold (validate on 2022, 39
KEV) cannot select models: LightGBM was best on validation and weakest of the
GBMs on test, and tuning made LR worse. **C1's registered verdict stands
(H2 and H1b not supported) and is reported as the primary C1 result.**
Everything below is a revision made *after* seeing C1's test results. C1-R
is therefore labelled post-hoc with respect to C1, while protocol v2 is
prospective and binding for C2–C6.

**What v2 changes, and why it is not tuned to C1's outcome.** The change
follows one mechanism, too few validation positives, and was chosen by
counting positives (a data property), not by looking at any model's score:

- *Validation window:* CVEs published 2021-05-01 → 2022-12-31 (2021-05-01 is
  the first EPSS archive), scored at their first snapshot like test, with
  already-known CVEs excluded like test.
- *Fold training set:* CVEs published < 2021-05-01, features as of
  2021-05-01.
- *Fold labels:* KEV as of 2023-01-01 (the split date T) for both fold
  training and fold validation. This is exactly what was knowable when the B2
  model was built at T, so nothing after T is used. The C1 fold instead used
  KEV as of each fold's own start, which left almost nothing to learn from.
- Effect: 652 fold-training positives (was 309) and **145 validation
  positives (was 39)**.
- Same 20 configurations per family as C1 (same sampling seeds), one fit
  each (seed 0), selection by validation PR-AUC. Only the fold changes.

Stated caveats: (1) as in the C1 fold, EPSS v1 files (2021 to early 2022)
have no percentile, so `epss_pit_pct` is missing for fold-training rows;
(2) validation labels for 2021 CVEs had a longer observation window than for
2022 CVEs.

**Selection-reliability diagnostic (new, never used for selection).** All
126 searched configurations (6 × (20 + default)) are fitted once on the full
training set (seed 0) and scored on test. For each fold (C1 fold, v2 fold)
we report the Spearman correlation between validation PR-AUC and test
PR-AUC, within each family and pooled. This measures whether a fold can
select models at all. It is the evidence for or against v2, and it is
reported whatever it shows.

**C1-R.** The v2-selected configuration per family is refitted with 10 seeds
on the full training set and tested once. H2 and H1b are re-tested exactly as
registered in the Study C addendum (best-validation GBM vs LR and DT;
best-validation model overall vs point-in-time EPSS; paired bootstrap).
Reported as "under protocol v2 (post-hoc for C1)", next to the C1 primary
result, never instead of it.

**Binding for C2–C6:** model and hyperparameter selection use the v2 fold.
