# Evaluation Protocol — LOCKED

**Experiment:** A6 · **Locked:** 2026-09-09 · **Status:** binding for Studies B, C, D
**Amended:** 2026-11-23 — L1 refreshed to the November catalog before any Study B
model existed. Rules unchanged; only the dataset counts below moved. See the
CLAUDE.md Deviations Log ("A1 — November refresh").

This protocol is fixed before any model is trained (CLAUDE.md rule 11). Any
deviation must be recorded in the Deviations Log in `CLAUDE.md` with its
justification — not applied silently.

---

## 1. Metrics

Implemented in `src/evaluation/metrics.py`; call `evaluate()` so every model is
scored identically.

| Role | Metric | Notes |
|---|---|---|
| **Primary** | PR-AUC | Model selection is decided on this |
| **Primary** | Precision@k, k ∈ {10, 50, 100} | What an engineer actually reads |
| Ranking | NDCG@k, MAP, MRR, Recall@k | k ∈ {10, 50, 100} |
| Calibration | Brier score, ECE (10 bins), reliability diagram | Every probabilistic model |
| Secondary | ROC-AUC | **Reported, never used to select** |

**Why ROC-AUC is demoted.** At the L1 base rate (1,790 KEV in 430,678 CVEs,
≈1:240) it is actively misleading. Measured on a synthetic imbalanced problem,
a *random* ranker scores ROC-AUC 0.556 while its PR-AUC is 0.006 and its
Precision@10 is 0.00. A model may not be called good on ROC-AUC.

---

## 2. Splits

Implemented in `src/evaluation/splits.py`, which raises rather than return a
silently broken split.

**L1 / L2 — temporal, always.** Train on CVEs published strictly before date
`T`, test on `T` onwards. Never shuffle. Default `T = 2023-01-01`
(train 203,652 / test 227,026; KEV 1,038 / 752). `assert_no_temporal_leakage()` must be called
after every split.

**L3 — grouped by asset.** No `asset_id` may appear in both train and test;
findings on one asset share its context features and are not independent.
Default test fraction 0.25.

**Study B5 (temporal drift)** trains on year *N* and tests on *N+1, N+2, N+3*
using the same machinery.

---

## 3. The EPSS leakage rule (binding)

**`epss_scores-current.csv.gz` must NOT be used as a feature in any temporally
split experiment.**

Evidence (A6). Among CVEs published before 2023-01-01 — i.e. rows that would be
*training* data:

| Group | n | median present-day EPSS |
|---|---|---|
| Never in KEV | 202,614 | 0.0142 |
| KEV added **before** 2023-01-01 | 864 | 0.6178 |
| KEV added **after** 2023-01-01 | 174 | **0.7028** |

CVEs whose exploitation became known only *after* the split point already carry
a median EPSS 49.5× the never-exploited baseline, and indistinguishable from
those exploited before it. Today's EPSS is close to an oracle for future KEV
membership.

**Required fix:** use point-in-time EPSS via
`src.ingestion.epss.fetch_epss_asof(split_date)`. Daily archives are available
back to at least 2021. Re-running the same test with EPSS as of 2023-01-01
drops the ratio from 49.5× to **5.7×**
(Sep 9 data: 45.7× → 5.0×; same conclusion) — the residual is legitimate predictive
signal, which is what EPSS is designed to provide.

EPSS-current remains permitted for: descriptive statistics, and as a **baseline
ranker to beat** in B1 (where no training occurs).

---

## 4. Statistical rigour

- **Minimum 10 seeds** per model; report mean ± standard deviation. A single
  run is never a result.
- **Across model families:** Friedman test, then Nemenyi post-hoc, with a
  critical-difference diagram.
- **Pairwise:** Wilcoxon signed-rank.
- Implemented in `src/evaluation/statistical.py` (already holds Cohen's κ with
  bootstrap CI from A3).

---

## 5. Baselines first

No learned model is reported before the B1 baseline battery has run: CVSS v3
score ordering, severity label, publication recency, EPSS-current ranking, and
random. **A model that cannot beat sorting by CVSS is reported honestly as a
negative result** (CLAUDE.md rule: that is publishable, not a failure).

---

## 6. Reporting rules

- **L1/L2 and L3 results are reported in separate tables. Always.** They do not
  share a label source and are not comparable.
- L3's label is a documented formula (60% real + 40% simulated), never
  presented as ground truth. High L3 accuracy is evidence a model can recover a
  deterministic function of its inputs — the informative L3 experiments are
  about robustness (label noise, weight sweeps, ablation), not raw performance.
- Every experiment logs to MLflow before the next one starts.

---

## 7. L3 label amendment (resolved 2026-09-09)

The original formula mixed a *probability* (CVE: EPSS) with an *ordinal
severity* (misconfiguration), a units mismatch that left `source_layer` alone
predicting the label at AMI 0.289 — a shortcut that would have dominated every
Study C ablation. Three coupled changes were adopted together:

1. `epss_score` → `epss_percentile` in the L3 label (rank, not probability);
2. explicit `misconfig_scale = 0.55` for the cross-layer balance;
3. thresholds recalibrated to 0.545 / 0.420 / 0.285 to preserve the original
   operational class sizes.

Result: shortcut AMI 0.289 → 0.095; vulnerabilities 99.4% `Low` → 80.6/15.6/
3.3/0.4. (Rebuilt on the November L1: AMI 0.098, vulnerabilities 81.5% `Low`,
overall 54.1/40.3/5.2/0.4 — thresholds still reproduce the target class sizes.) Raw `epss_score` is unchanged in L1 experiments (B2/B3), where
calibration rather than discrimination is what matters.

**Binding for Study C:** `misconfig_scale` is swept in C7, and any sweep must
either recalibrate the thresholds or report the resulting class-size shift —
they are coupled. Residual AMI ≈0.1 is not zero and must not be reported as
if layer carried no signal.
