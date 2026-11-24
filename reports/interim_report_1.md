# Interim Report 1 — Studies A and B

**Project:** Cloud Security Finding Prioritisation — ML & XAI
**Student:** Popa Stefan Andrei (6415250) · Master AI, Maastricht University DACS
**Supervisors:** Mikhail Glazunov (UM DACS) · Romulus-Cristinel Viesanu (Deutsche Bank CSO)
**Covers:** Study A (data foundation) and Study B (exploitation prediction & baselines)
**Date:** 2026-11-24

---

## 1. Summary

The data foundation is built and audited, the evaluation protocol and analysis
plan were locked before any model was trained, and all six Study B experiments
have run (10 seeds each, logged to MLflow).

**Headline result.** A model trained on public NVD fields ranks exploited
vulnerabilities clearly better than sorting by CVSS — the current default
practice — so **H1 is supported**. It does **not**, however, beat
point-in-time EPSS, and a plain logistic regression beats the gradient-boosted
model. ML adds value over CVSS; the open question for Study C is whether any
model adds value over EPSS.

| Question | Answer |
|---|---|
| Does ML beat CVSS ordering for KEV prediction? (H1) | **Yes** — PR-AUC +0.0095 [0.0033, 0.0227], P@100 +0.08 [0.02, 0.15], 10/10 seeds |
| Does it beat point-in-time EPSS? | **No** — PR-AUC −0.0017 [−0.0116, 0.0043] |
| Does performance decay over time? (H5) | Consistently in direction, **not significant** (ρ = −0.53, p = 0.14) |
| Is the model usable for threshold decisions? | Well calibrated (ECE 0.00035); under a fixed monthly budget it finds ~2× the KEV CVSS finds |

## 2. Constraint and approach

No Deutsche Bank internal data is available. The project uses a three-layer
digital twin, and real and simulated results are always reported separately:

- **L1 (real):** NVD + EPSS + CISA KEV — 430,678 CVEs, 1,790 KEV (1:240).
- **L2 (real):** Prowler, Checkov and ScoutSuite rule catalogs — 3,870 checks.
- **L3 (simulated):** a parameterised financial-institution estate —
  39,989 findings on simulated assets, with a documented priority-label formula.

Study B labels come only from CISA KEV — never invented, never derived from
tool severity.

## 3. Study A — data foundation

| Step | Outcome |
|---|---|
| A1 ingestion | NVD API 2.0 (legacy feeds are dead), EPSS, KEV joined on CVE ID; refreshed to the November catalog before Study B |
| A2 misconfig catalogs | 3,870 checks (Checkov 1,684 · Prowler 1,608 · ScoutSuite 578) in one schema |
| A3 severity disagreement | Prowler vs ScoutSuite **κ = 0.179** [0.015, 0.330] — only slight agreement |
| A4 remediation families | 138 ± 3 families; membership only moderately stable (ARI 0.53) |
| A5 estate simulator | Built; label formula amended (below) |
| A6 EDA + protocol lock | EPSS leakage quantified and fixed; protocol and hypotheses pre-registered |

**Findings that shape the rest of the project:**

1. **Tool severity is not a trustworthy label.** Two mainstream scanners agree
   only slightly, and systematically — every matched CloudWatch alarm check is
   `medium` in Prowler and `danger` in ScoutSuite. Checkov's open-source repo
   ships no severities at all. This justifies KEV-only labels.
2. **Today's EPSS leaks the outcome.** Under a temporal split, CVEs that
   entered KEV after the split already carry present-day EPSS 49.5× that of
   never-exploited CVEs. Using archived daily EPSS as it stood at the time cuts
   this to 5.7× (real signal). Present-day EPSS is banned as a feature.
3. **L3 label amendment (2026-09-09, for discussion).** As originally specified,
   the label put CVEs and misconfigurations on incompatible scales: 99.4% of
   vulnerabilities were labelled `Low` and the finding type alone predicted
   the label (AMI 0.289). The amended formula uses EPSS percentile and an
   explicit cross-layer weight with recalibrated thresholds (AMI 0.095,
   vulnerabilities spread 80.6/15.6/3.3/0.4%). It is a documented change to a
   formula declared fixed, and it is in the Deviations Log.
4. **Silent data bugs were found and fixed** — e.g. pandas date inference
   silently destroyed 91% of publication dates on re-read; a "refresh" that
   did not refresh. Both are now guarded in code.

## 4. Study B — exploitation prediction

**Setup (pre-registered).** Temporal split at 2023-01-01. Features are read *as
of the date an analyst would first see the CVE*: CVSS score/severity/version,
description text (TF-IDF → 64-d SVD fitted on training text only) and
point-in-time EPSS. Training labels are KEV *as of the split date*. Test CVEs
already in KEV when first triaged are excluded — 443 of 752. KEV is fast for
the CVEs it catches, so the task left is the slow ones: 309 KEV in 226,463
test CVEs.

### B1 — baselines

On the full test set CVSS v3 ordering reaches PR-AUC 0.0148 and P@100 0.037:
96 of its top 100 are not exploited, yet its ROC-AUC is 0.786. This is why
ROC-AUC is reported only as a secondary metric. On the B2 evaluation set
(already-known CVEs removed) CVSS v3 falls to PR-AUC 0.0068.

### B2 — KEV prediction (H1)

| Scorer | PR-AUC | P@100 |
|---|---|---|
| Logistic regression | **0.0235** | 0.13 |
| Point-in-time EPSS | 0.0175 | 0.11 |
| Gradient boosting (primary) | 0.0143 ± 0.0016 | 0.089 ± 0.022 |
| CVSS v3 | 0.0068 ± 0.0014 | 0.017 ± 0.012 |
| Random | 0.0016 ± 0.0005 | 0.002 ± 0.004 |

Leaky reference (present-day EPSS, not a fair competitor): PR-AUC 0.351.

- H1 supported against CVSS on both pre-registered metrics (paired bootstrap).
- Not better than point-in-time EPSS: EPSS is itself a model trained on
  exploit-activity data we do not have.
- Logistic regression outperforms the GBM. With 864 training positives and
  noisy labels, a low-variance model wins. Hyperparameters were fixed in
  advance and were not tuned after seeing this.
- Labelling the training set with today's KEV does not help (0.0131), so the
  stricter, leak-free labelling rule costs nothing.

### B3 — EPSS regression

Public NVD fields reproduce EPSS only weakly (Spearman ρ = 0.36). As a KEV
ranker this model is no better than CVSS (PR-AUC 0.0066). EPSS's signal is
mostly not in CVSS and the description text.

### B4 — ranking the monthly queue

Each scorer ranks each month's new CVEs, the queue an analyst actually works
(46 months). Scorers differ significantly on every metric (Friedman,
p ≤ 3×10⁻⁴). GBM beats CVSS by paired Wilcoxon (p 0.003–0.013), but not by
Nemenyi (p ≥ 0.31), which is conservative with six scorers. Both are reported.

### B5 — temporal drift (H5)

Train on year N ∈ {2021…2024}, test on N+1…N+3, with a fixed 365-day KEV label
so that drift is not confused with KEV lag. The model's margin over CVSS falls
with the gap (0.034 → 0.022 → 0.016) and falls within every training year.
With 9 cells it is not significant (p = 0.14): **consistent decay,
underpowered**. KEV only started in November 2021, which limits the training
years (pre-registered deviation).

### B6 — cost-sensitive evaluation

No bank cost data exists, so the cost of a missed exploited CVE relative to an
unnecessary fix is swept over 10/50/100/500.

- The model is well calibrated (Brier 0.00137, ECE 0.00035).
- With a fixed monthly remediation budget it catches about twice the KEV that
  CVSS ordering catches for the same effort (top-10/month: 24 vs 11). It is
  the cheapest policy at ratios 50 and 100.
- The pre-registered threshold chosen on a validation year is unstable,
  because 2022 has only 39 KEV. The calibrated Bayes threshold is stable and
  cheapest at ratio 500 (CVSS ≥ 9 is close behind).
- At ratio 10, doing nothing is cheapest: exploitation is rare enough that
  fixing costs more than it saves.

## 5. Threats to validity

- KEV is a lagging, partial ground truth: absence from KEV is not evidence of
  no exploitation.
- CVSS scores and descriptions are today's versions (NVD rarely rescores).
- B5 and B6 are limited by few positives per year.
- No result is a measurement of Deutsche Bank's estate; L3 results will be
  reported separately and never as ground truth.

## 6. Deviations from plan

All are logged with reasons in the CLAUDE.md Deviations Log, recorded when
found. Key ones: NVD bulk feeds retired (API used instead); Checkov has no
open-source severities; the L3 label amendment; present-day EPSS replaced by
archived point-in-time EPSS; NVD reference-tag features demoted to a
sensitivity check (retroactive tagging leaks); B5 training years moved to
2021–2024 because of KEV's launch date; LightGBM deferred to Study C
(`libomp` missing on this machine).

**Schedule.** This report was planned for end of September; Studies A and B
took longer, mostly because of the data-quality problems in §3. Study C
starts now, against the 1 February 2027 deadline.

## 7. Next steps — Study C

1. C1 classical model suite on the B2 evaluation set. H2 already has a
   counter-example (logistic regression > GBM).
2. Make "beat point-in-time EPSS" an explicit second bar alongside CVSS.
3. C5 calibration and C6 statistical comparison (Friedman/Nemenyi,
   critical-difference diagrams).

**For discussion with supervisors:** the L3 label amendment (§3.3), and
whether "does not beat EPSS" should be framed as the headline negative result.

---

*Reproducibility:* code in `src/`, one executed notebook per experiment in
`notebooks/`, all runs in MLflow (`experiments/mlflow.db`). Pre-registration
and locked protocol: `reports/analysis_plan_preregistration.md`,
`reports/evaluation_protocol.md`.
