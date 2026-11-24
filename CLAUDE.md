# Cloud Security Finding Prioritisation — ML & XAI
## Master AI Internship · Deutsche Bank CSO · Maastricht University DACS

---

## Project Identity

**Student:** Popa Stefan Andrei (6415250)
**Academic supervisor:** Mikhail Glazunov (Maastricht University DACS)
**Daily supervisor:** Romulus-Cristinel Viesanu (Deutsche Bank CSO)
**Period:** August 2026 – February 2027
**Deadline:** 1 February 2027

**Central Research Question:**
To what extent can machine learning and explainable AI support the
prioritisation of cloud security and vulnerability findings in an
enterprise financial environment?

**SQ1:** Which machine learning methods are most suitable for
classifying or prioritising security findings?

**SQ2:** How can explainable AI techniques make model predictions
understandable for security engineers and stakeholders?

---

## Critical Constraint — Zero Internal Data

We have NO access to Deutsche Bank internal data. The entire project
uses a three-layer digital twin approach:

- **L1 (REAL):** NVD + EPSS + CISA KEV — real vulnerability
  intelligence, real external labels, never invented
- **L2 (REAL):** Prowler + Checkov + ScoutSuite + CIS Benchmarks —
  real cloud misconfiguration rule catalogs extracted from GitHub repos
- **L3 (SIMULATED):** Parameterised financial institution estate —
  assets, IAM graph, criticality tiers, SLA, remediation dynamics

**The scientific defensibility rule:** labels for Study B experiments
come ONLY from CISA KEV (external, real, not invented). L3 labels are
a documented formula (60% real KEV/EPSS component + 40% simulated
context). Always report L1/L2 experiments and L3 experiments separately.
Never conflate them.

---

## Four Studies — 24 Experiments

### Study A — Data Foundation (Month 1: August)
- A1: NVD + EPSS + KEV ingestion and joining on CVE ID
- A2: Prowler + Checkov + ScoutSuite rule catalog extraction from GitHub
- A3: Cross-tool severity disagreement analysis (Cohen's κ)
- A4: Remediation family clustering (HDBSCAN on check embeddings)
- A5: Estate simulator build (assets, IAM graph, criticality, SLA, temporal dynamics)
- A6: Exploratory data analysis + lock evaluation protocol + pre-register analysis plan

### Study B — Exploitation Prediction & Baselines (Month 2: September)
- B1: Baseline battery (CVSS v3/v4, severity label, publication recency, random)
- B2: KEV membership prediction (binary classification, strict temporal split, no leakage)
- B3: EPSS regression (continuous target, enables ranking without binary labels)
- B4: Pure ranking evaluation (NDCG@k, MAP, MRR, Precision@k)
- B5: Temporal drift (train year N, test N+1, N+2, N+3 — measure decay)
- B6: Cost-sensitive evaluation (bank cost matrix, threshold optimisation)

### Study C — Model Comparison (Month 3: October)
- C1: Classical ML suite (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)
- C2: Learning-to-rank (LambdaMART, RankNet — pointwise vs pairwise vs listwise)
- C3: Deep tabular models (TabNet, FT-Transformer, standard MLP)
- C4: Imbalance and feature ablation (CVE-only vs context-only vs text-only vs all combined)
- C5: Calibration study (Brier score, ECE, reliability diagrams, Platt scaling)
- C6: Statistical rigour (≥10 seeds, mean ± std, Friedman + Nemenyi post-hoc, critical-difference diagrams)

### Study D — Explainability & Human Evaluation (Month 4: November)
- D1: Global explainability (SHAP global summaries, permutation importance, ALE plots)
- D2: Local explainability (TreeSHAP, LIME per individual finding)
- D3: Counterfactual explanations (DiCE — actionable what-if statements per finding)
- D4: Interpretable models (EBM, RuleFit, optimal sparse decision trees)
- D5: Stability and agreement analysis (Kendall's τ, SHAP vs LIME inter-explainer agreement)
- D6: Human evaluation study with CSO engineers (6 explanation formats, comprehension, trust, decision time)

---

## Evaluation Protocol (LOCKED — do not change without documenting the deviation)

**Primary metrics:** PR-AUC and Precision@k
- Do NOT use ROC-AUC as primary — class imbalance makes it misleading
- ROC-AUC can be reported as secondary only

**Ranking metrics:** NDCG@k, MAP, MRR, Recall@k
- k values: 10, 50, 100 unless otherwise noted

**Calibration:** Brier score, Expected Calibration Error (ECE), reliability diagrams
- Every probabilistic model must be calibration-checked before reporting

**Statistical testing:** Friedman test across model families + Nemenyi post-hoc
- Wilcoxon signed-rank for pairwise comparisons
- Never report a single run as a result

**Data splits:**
- L1 data: ALWAYS temporal split — train on CVEs before date T, test after
- NEVER random shuffle time-ordered vulnerability data — this leaks future
  exploitation information into training and invalidates all results
- L3 data: grouped-by-asset split to prevent same asset in train and test

**Seeds:** minimum 10 random seeds per model, report mean ± standard deviation

**Baselines must be run first:** if a model cannot beat sorting by CVSS score,
report it honestly — it is a publishable negative result, not a failure

---

## Project Structure

```
cloud-security-prioritisation/
├── CLAUDE.md                    ← you are here — read this first every session
├── README.md                    ← project overview
├── requirements.txt             ← all dependencies pinned
├── .gitignore                   ← data/, experiments/, __pycache__/ excluded
│
├── data/
│   ├── raw/                     ← NVD JSON, EPSS CSV, KEV JSON, cloned repos
│   ├── processed/               ← l1_vulnerabilities.csv, l2_misconfig_catalog.csv
│   └── simulator/               ← l3_simulated_findings.csv
│
├── notebooks/
│   ├── A1_ingestion.ipynb       ← NVD + EPSS + KEV download and join
│   ├── A2_misconfig.ipynb       ← Prowler + Checkov rule extraction
│   ├── A3_disagreement.ipynb    ← cross-tool severity analysis
│   ├── A4_clustering.ipynb      ← remediation family clustering
│   ├── A5_simulator.ipynb       ← estate simulator build
│   ├── A6_eda.ipynb             ← EDA and evaluation protocol lock
│   ├── B1_baselines.ipynb
│   ├── B2_kev_prediction.ipynb
│   ├── B3_epss_regression.ipynb
│   ├── B4_ranking.ipynb
│   ├── B5_temporal_drift.ipynb
│   ├── B6_cost_sensitive.ipynb
│   ├── C1_classical_ml.ipynb
│   ├── C2_learning_to_rank.ipynb
│   ├── C3_deep_tabular.ipynb
│   ├── C4_ablation.ipynb
│   ├── C5_calibration.ipynb
│   ├── C6_statistical_rigour.ipynb
│   ├── D1_global_xai.ipynb
│   ├── D2_local_xai.ipynb
│   ├── D3_counterfactuals.ipynb
│   ├── D4_interpretable_models.ipynb
│   ├── D5_stability.ipynb
│   └── D6_human_study.ipynb
│
├── src/
│   ├── ingestion/
│   │   ├── nvd.py               ← NVD API download and parsing
│   │   ├── epss.py              ← EPSS CSV download
│   │   ├── kev.py               ← CISA KEV JSON download
│   │   └── join.py              ← join all three on CVE ID
│   ├── misconfig/
│   │   ├── repos.py             ← shallow-clone the three tool repos
│   │   ├── prowler_extractor.py ← extract checks from Prowler repo
│   │   ├── checkov_extractor.py ← extract checks from Checkov repo (AST)
│   │   ├── scoutsuite_extractor.py ← extract findings + ruleset severity
│   │   ├── normaliser.py        ← unified schema normalisation (A2 entry point)
│   │   ├── matching.py          ← cross-tool check pairing (embeddings)
│   │   ├── disagreement.py      ← A3 Cohen's κ analysis
│   │   └── clustering.py        ← A4 remediation family clustering
│   ├── simulator/
│   │   ├── bank.py              ← bank structure and business units
│   │   ├── assets.py            ← asset generation
│   │   ├── iam.py               ← IAM graph simulation
│   │   ├── injector.py          ← inject L1/L2 findings onto assets
│   │   ├── temporal.py          ← lifecycle simulation (SLA, recurrence)
│   │   └── labeller.py          ← priority label computation
│   ├── models/
│   │   ├── baselines.py         ← CVSS sorting, severity label, random
│   │   ├── classical.py         ← sklearn models
│   │   ├── ranking.py           ← LambdaMART, RankNet
│   │   ├── deep_tabular.py      ← TabNet, FT-Transformer, MLP
│   │   └── calibration.py       ← Platt scaling, isotonic regression
│   ├── explainability/
│   │   ├── shap_global.py       ← global SHAP analysis
│   │   ├── shap_local.py        ← per-finding TreeSHAP
│   │   ├── lime_local.py        ← LIME explanations
│   │   ├── counterfactuals.py   ← DiCE counterfactuals
│   │   └── stability.py         ← Kendall's τ, agreement analysis
│   └── evaluation/
│       ├── metrics.py           ← PR-AUC, NDCG@k, MAP, MRR, Brier
│       ├── statistical.py       ← Friedman, Nemenyi, Wilcoxon
│       └── cost_matrix.py       ← bank cost matrix evaluation
│
├── experiments/                 ← MLflow tracking (auto-generated, gitignored)
├── outputs/                     ← plots, tables, CSV results
└── reports/                     ← thesis chapters, interim reports
```

---

## Tech Stack

```
# Data ingestion and processing
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.65.0
python-dateutil>=2.8.2

# NLP and embeddings (for clustering check descriptions)
sentence-transformers>=2.2.2

# Clustering
hdbscan>=0.8.33
umap-learn>=0.5.3

# Classical ML
scikit-learn>=1.3.0
xgboost>=1.7.6
lightgbm>=4.0.0
catboost>=1.2.0

# Deep tabular
pytorch-tabnet>=4.1.0
torch>=2.0.0

# Explainability
shap>=0.42.0
lime>=0.2.0.1
interpret>=0.4.0
dice-ml>=0.9

# Statistical testing
scipy>=1.11.0
scikit-posthocs>=0.7.0

# Experiment tracking
mlflow>=2.7.0
dvc>=3.0.0

# Notebooks and visualisation
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

---

## Non-Negotiable Rules for Every Session

1. **Read CLAUDE.md at the start of every session** — never assume context from memory
2. **Never random-split time-ordered data** — always temporal split on L1
3. **Never invent labels** — L1 labels come from CISA KEV only, never generated
4. **Always run baselines before building models** — establish the bar first
5. **Always report variance** — minimum 10 seeds, mean ± std, never a single run
6. **Log every experiment to MLflow** before moving to the next experiment
7. **Document every simulator parameter** — nothing undocumented, everything sweepable
8. **Start each study with EDA** before writing any model code
9. **One notebook per experiment** — keep notebooks small and focused
10. **Never commit raw data to git** — data/ is in .gitignore
11. **Lock evaluation protocol in A6** before touching any model in Study B
12. **Report L1/L2 and L3 experiments separately** — never conflate real and simulated labels
13. **Pre-register the analysis plan** after A6 EDA, before running any models
14. **Update this file the moment reality diverges from the plan** — a dead URL,
    a missing field, a method that cannot work as specified, a dependency that
    had to change. Record it in the Deviations Log below *when it is found*,
    not at the end of the study. A plan that silently rots is worse than no plan,
    and these deviations are thesis content: they are the real constraints of
    working with public security data.

---

## Data Sources — URLs and Access

```python
# L1 — Vulnerability Intelligence
# DEAD (HTTP 403) — NIST retired the legacy JSON 1.0/1.1 feeds in Dec 2023.
# NVD_BULK_URL = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
NVD_API_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"   # use this
EPSS_URL     = "https://epss.cyentia.com/epss_scores-current.csv.gz"  # 301 -> epss.empiricalsecurity.com
KEV_URL      = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# L2 — Cloud Misconfiguration Catalogs (clone from GitHub)
PROWLER_REPO   = "https://github.com/prowler-cloud/prowler"
CHECKOV_REPO   = "https://github.com/bridgecrewio/checkov"
SCOUTSUITE_REPO = "https://github.com/nccgroup/ScoutSuite"
```

**NVD API 2.0 notes:** paginate `resultsPerPage=2000` via `startIndex`; without
an API key the limit is ~5 req/30s and a full pull (~431k CVEs, ~216 pages)
takes 1.5–2h. `src/ingestion/nvd.py` caches every page to `data/raw/nvd/` so an
interrupted run resumes for free; a refresh re-fetches only short (formerly
last) pages and the new tail, then overlays every CVE NVD modified since the
cache was built (`lastModStartDate`, cached in `data/raw/nvd_updates/`). KEV and
EPSS-current are saved as dated raw copies in `data/raw/kev/` and
`data/raw/epss/` — they are the label provenance. Run long pulls under
`caffeinate -s -d -i -m` (see Deviations Log).

---

## L3 Simulator — Label Formula (documented and fixed)

```python
# Priority score = 60% real component + 40% context component
# Real component (from L1 — external, not invented):
#   For vulnerabilities: 0.40 * epss_PERCENTILE + 0.35 * in_kev + 0.25 * public_exploit
#   For misconfigurations: severity_map * misconfig_scale
#       severity_map = {critical:1.0, high:0.75, medium:0.5, low:0.25}
#       misconfig_scale = 0.55   ← explicit cross-layer balance, swept in C7
#
# AMENDED 2026-09-09 (A5/A6 evidence). Two changes from the original:
#   1. epss_score -> epss_percentile. Same real EPSS data as a rank. Raw EPSS
#      is ~0.003 for a typical CVE, so the original squashed 74.9% of
#      vulnerabilities below 0.1 and labelled 99.4% of them Low.
#   2. misconfig_scale added. The original mixed a probability (CVE) with an
#      ordinal severity (misconfig) — a units mismatch that no weighting can
#      fix, only hide. This states the cross-layer preference explicitly.
# Effect: source_layer alone predicted the label at AMI 0.289; now 0.095.
# Vulnerabilities went from 99.4% Low to 80.6/15.6/3.3/0.4.
# Full reasoning in the Deviations Log (A5).

# Context component (from L3 — simulated, documented):
#   0.30 * tier_weight[criticality_tier]   # Tier 0=1.0, Tier 4=0.0
#   0.20 * internet_facing                  # boolean
#   0.20 * (pci_scope OR dora_scope)        # regulatory boolean
#   0.15 * sla_breached                     # boolean
#   0.15 * min(recurrence_count / 5, 1.0)  # capped at 1.0

# Label thresholds (recalibrated with the amendment above — they are COUPLED
# to the component scales; changing one alone silently changes class sizes):
#   score >= 0.545 → Critical
#   score >= 0.420 → High
#   score >= 0.285 → Medium
#   score <  0.285 → Low
# Chosen to reproduce the ORIGINAL operational class sizes (0.4% Critical,
# ~5.5% High, ~40% Medium). Any C7 sweep of misconfig_scale must recalibrate
# these or report the resulting class-size shift.

# All weights are tunable parameters swept in experiment C7
# Label noise injection controlled by noise_level parameter (0.0 to 0.40)
```

---

## Deviations Log

Where reality diverged from this plan. Maintained under rule 14 — append the
moment something is discovered, not retrospectively. Several of these are
thesis-worthy findings about the state of public security data, not just
housekeeping.

### A1 — ingestion

- **NVD bulk JSON feed is dead.** `NVD_BULK_URL` returns HTTP 403; NIST retired
  the JSON 1.1 feeds in December 2023. Switched to the paginated API 2.0.
- **EPSS is a *current snapshot*, not a point-in-time score.** The
  `epss_scores-current.csv.gz` endpoint gives today's EPSS for every CVE, not
  the score as it stood at each CVE's publication. Joining it onto 1988–2026
  CVEs means old CVEs carry a 2026 score. **This is a leakage risk for Study B**
  — an EPSS feature in a temporal split encodes information from after the
  split point. Either drop EPSS as a feature in B2, or source the historical
  daily EPSS archives before using it. Not yet resolved; decide in A6.
- **Class imbalance is severe:** 1,695 KEV of 387,494 CVEs = 1:227.6. PR-AUC
  and Precision@k (already the locked primary metrics) are the right call.
- **macOS sleep silently freezes long runs.** Multi-hour pulls stalled with the
  process alive but consuming zero CPU; `caffeinate -i` was not enough because
  it only blocks *idle* sleep. Use `caffeinate -s -d -i -m` on AC power.
- **`requests`' `timeout=` is a per-read inactivity timeout, not a deadline.** A
  server trickling bytes never trips it. `nvd.py` adds a SIGALRM hard deadline.

### A2 — misconfiguration catalog

- **Checkov's OSS repo ships no severities at all** (0 of 1,684 checks; only 5
  files even mention the word). Severities live in the commercial Bridgecrew
  platform. This removes Checkov from any severity-based analysis.
- **ScoutSuite has only two levels** (`danger`/`warning`) against Prowler's
  five. Mapping between them is lossy by construction; the documented mapping
  lives in `src/misconfig/normaliser.py`.
- **`check_id` is not a unique key.** Checkov reuses ids across IaC frameworks,
  Prowler reuses them across providers, and Checkov has one genuine upstream
  collision (`CKV_NCP_22` on two unrelated checks). Use `catalog_id`.
- **Files added beyond the planned structure:** `scoutsuite_extractor.py`
  (ScoutSuite is in the L2 definition but was missing from the file list),
  `repos.py` (clone helper), `matching.py`, `disagreement.py`. Added `pyyaml`
  to requirements (Checkov graph checks are YAML).

### A3 — severity disagreement

- **A three-way Cohen's κ is impossible.** With Checkov severity-less, the
  statistic is Prowler vs ScoutSuite only.
- **CIS recommendation numbers are NOT a safe join key.** ScoutSuite ships CIS
  1.2.0, Prowler ships 1.4/1.5/2.0/3.0, and sections were renumbered between
  them — CIS 1.2.0 §2.1 (CloudTrail in all regions) is §3.1 from 1.4 onward.
  Joining on the raw numbers produces silently wrong pairs. Checks are paired
  by semantic similarity instead, with a threshold sweep.
- **Result: κ = 0.179, 95% CI [0.015, 0.330]** — only *slight* agreement, stable
  across every threshold 0.50–0.80. Disagreement is systematic, not noise (all
  13 matched CloudWatch alarm checks: Prowler `medium`, ScoutSuite `danger`).
  **Implication: tool-assigned severity is not a trustworthy label source**,
  independently justifying rule 3 (KEV-only labels).
- **Caveat for the write-up:** 161 pairs is a modest sample and the CI is wide;
  report the direction (poor agreement), not κ as a precise point estimate.

### A1 — silent date-parsing corruption (found during A5)

- **`pd.to_datetime` without `format=` silently destroyed 91% of the dates.**
  Pandas 2.x infers one datetime format from the first value and coerces every
  row that differs to `NaT`. `l1_vulnerabilities.csv` held mixed
  representations (`...09.143000+00:00` where fractional seconds existed,
  `...04:00:00+00:00` where they did not), so re-reading it left 353,744 of
  387,494 `published_date` values as `NaT` — with `errors="coerce"` this
  happened with no warning at all.
- **What it broke:** A5's vulnerability pool silently shrank from ~100k
  candidate CVEs to **715**, so the first simulator run drew from 709 distinct
  CVEs instead of tens of thousands. The A1 CSV itself was fine; the bug was on
  every *re-read*.
- **Fix:** `join.py` now writes one uniform second-precision ISO format, and
  every consumer passes `format="ISO8601"` explicitly. **Any new code reading a
  timestamp column must pass `format=` — never rely on inference.**
- **Lesson worth carrying into Study B:** `errors="coerce"` converts a loud
  failure into a silent one. Assert on null counts after parsing, rather than
  trusting that a parse succeeded.

### A4 — remediation family clustering

- **UMAP cannot be dropped, even though it is the only stochastic step.**
  Clustering the 384-d embeddings directly is perfectly reproducible (ARI 1.0)
  but degenerate: 2–6 clusters at silhouette 0.02–0.05. With UMAP first: 138
  families at silhouette 0.66. Tested explicitly, not assumed.
- **Family membership is only moderately reproducible: pairwise ARI 0.53 ±
  0.04 over 10 seeds.** The *aggregate* structure is stable (138 ± 3 families,
  16.3% ± 0.7% noise), but which checks land together shifts between runs.
  **Do not treat a family id as a stable object**, and be careful using
  `family` as a categorical feature in Study C — it is seed-dependent. If a
  stable partition is needed later, use consensus clustering over seeds.
- **These are topic families more than remediation families.** Only 46% of
  checks ship remediation text, so for the rest the embedding sees "what the
  check is about", not "how it is fixed". Inspection confirms it: the largest
  family mixes RDS auto-upgrade, audit logging, Multi-AZ and monitoring —
  one service, four different fixes. Families are coherent but noisy at the
  edges (a secrets family contains a CircleCI "orbs" check).
- **The useful part holds regardless:** 106 of 138 families span more than one
  tool and 99 span more than one provider, which is the evidence that the same
  remediation themes recur across vendors — the thing A4 was for.

### A5 — estate simulator

- **`public_exploit` did not exist.** The label formula requires it but A1 never
  extracted it. Rather than invent the field, it is now taken from NVD's own
  reference tagging (a reference tagged `Exploit`), re-parsed from the cached
  pages at no network cost. It behaves sensibly: 22.5% of non-KEV CVEs carry
  one against 42.3% of KEV CVEs. `has_patch` was added the same way.

- **⚠ THE LABEL FORMULA PUTS L1 AND L2 ON INCOMMENSURATE SCALES — needs a
  decision before Study C.** Implemented exactly as documented, the real
  component gives:

  | layer | median | mean | p99 |
  |---|---|---|---|
  | L1 (CVEs)   | 0.001 | 0.037 | 0.263 |
  | L2 (checks) | 0.500 | 0.548 | 1.000 |

  A *median misconfiguration* outranks a *99th-percentile vulnerability*. The
  cause is structural: misconfigurations get an absolute severity map
  (0.25–1.00) while vulnerabilities get `0.40*epss + 0.35*in_kev +
  0.25*public_exploit`, and EPSS is ~0.003 for a typical CVE while KEV
  membership is ~0.2% of sampled findings. **Consequence: 99.4% of
  vulnerability findings are labelled Low, and the L3 priority label is
  effectively "is this a misconfiguration?"** Any model trained on it will
  learn that shortcut. Options: (a) rank-normalise EPSS within the pool,
  (b) calibrate the misconfig severity map down to the CVE scale, (c) label
  the two layers on separate scales and never mix them.

- **RESOLVED 2026-09-09 — decision and evidence.** The diagnosis was refined
  first: the damaging part is not that misconfigurations outrank CVEs on
  average (arguably they should — an exposed bucket is *definitionally*
  exposed, a median CVE has ~0.3% exploitation probability). It is that
  `source_layer` alone predicted the label at **AMI 0.289 / 67.9% accuracy**,
  so "is this a misconfiguration?" was a shortcut that would have dominated
  every Study C feature ablation. Four configurations were measured:

  | option | overall L/M/H/C | vulns `Low` | AMI shortcut |
  |---|---|---|---|
  | 1. original | 56.8/37.1/5.7/0.4 | 99.4% | 0.289 |
  | 2. percentile only | 55.5/38.4/5.7/0.4 | 95.1% | 0.228 |
  | 3. + misconfig ×0.55 | 91.4/8.5/0.1/0.0 | 95.1% | 0.010 |
  | **4. + recalibrated thresholds (ADOPTED)** | **53.8/40.3/5.4/0.4** | **80.6%** | **0.095** |

  Option 2 alone barely dented the shortcut. Option 3 removed it but collapsed
  91.4% of findings into `Low` — because thresholds and component scales are
  coupled and only one was changed. Option 4 changes all three coupled pieces
  together and is what ships.

  **Why a rank rather than the calibrated probability:** the L3 label is an
  *ordinal priority class*, so it needs discrimination, not calibration.
  Calibration matters where real outcomes are predicted — B2/B3 on L1 — and
  there the raw `epss_score` is used unchanged. Rank for ordinal labels,
  probability for probabilistic targets.

  **Why an explicit cross-layer parameter rather than a "correct" value:**
  there is no correct value in the data. "Is a medium misconfiguration more
  urgent than a 60th-percentile CVE?" is the bank's value judgement. Leaving it
  implicit meant it was being answered accidentally by a units mismatch;
  `misconfig_scale` states it, and C7 sweeps it.

  **Why not label the layers separately and never mix (option c):** the central
  research question is explicitly about prioritising cloud security *and*
  vulnerability findings — one queue. Refusing to mix dodges the question.

  Residual AMI of 0.095 is not zero and should not be reported as if it were:
  layer still carries some signal, which is legitimate (the layers genuinely
  differ) rather than degenerate.

- **KEV is too rare inside the simulated estate to study.** Only 29 of 12,451
  sampled vulnerability findings are KEV-listed (0.23%), because KEV is 0.44%
  of NVD and sampling is recency-weighted. Any L3 experiment about KEV would
  be underpowered; KEV work belongs in L1 (Study B), not L3.

- **Structure added beyond the planned file list:** `config.py` (every
  parameter in one dataclass, per rule 7) and `build.py` (the orchestrator).
  Added `networkx` to requirements for the IAM graph.

- **Documented simplifications in the injector:** misconfigurations are matched
  to assets by cloud provider only (the 474-value L2 service vocabulary does
  not align with simulated asset types), and CVEs are drawn by recency weight
  rather than CPE-matched to a software inventory, which does not exist here.
  CVE presence is therefore a weighted draw, not a claim of applicability.

### A6 — EDA and protocol lock

- **EPSS-current leaks the outcome under a temporal split — quantified.** Among
  CVEs published before 2023-01-01, those added to KEV only *after* that date
  already carry a median present-day EPSS of 0.643 — **45.7×** the 0.014 of
  never-KEV CVEs, and indistinguishable from the 0.618 of CVEs already in KEV
  before the split. Today's EPSS is close to an oracle for future exploitation.
- **Fixed, not dropped.** Historical daily EPSS archives turned out to be
  available (verified back to 2021), so `fetch_epss_asof(date)` was added to
  `src/ingestion/epss.py`. Re-running the test with EPSS as of the split date
  cuts the ratio from 45.7× to **5.0×** — the residual is genuine predictive
  signal, which is what EPSS is for. **This resolves open decision #1.**
- **Binding rule now in the protocol:** `epss_scores-current.csv.gz` may not be
  a feature in any temporally split experiment. Still allowed for descriptive
  statistics and as a B1 baseline ranker (no training involved).
- **KEV grew mid-project** (1,695 → 1,697 entries). Figure captions must be
  computed from the data, never hardcoded — one already went stale.
- **Files added beyond the planned list:** `src/evaluation/splits.py` (temporal
  and grouped splits, which raise rather than return a leaking partition) and
  `src/evaluation/eda.py`. `metrics.py` and `statistical.py` were already
  planned.

### A1 — November refresh (2026-11-23, before any Study B model)

- **Decision: Study B runs on the November catalog**, not the Sep 9 lock.
  430,678 CVEs (was 387,494), 1,790 KEV (was 1,697), imbalance 1:239.6. The
  pre-2023 training partition is unchanged at 203,652 rows; only the test side
  grew (183,842 → 227,026; KEV 1,038 train / 752 test). The Sep 9 L1 and L3 are
  kept in `data/snapshots/` (gitignored) as the record of what was locked.
- **The refresh had silently not refreshed.** `nvd.py` took `totalResults` from
  cached page 0, so a November run returned the September total. Fixed with a
  live count — and that exposed a second bug: the cached *last* page from
  September held 1,494 CVEs, and reusing it would have dropped the 506 CVEs that
  have since filled it. Short non-final pages are now re-fetched.
- **Cached pages go stale, not just incomplete.** 98,016 CVEs (23% of the
  catalog) were modified by NVD between Sep 7 and Nov 23 — late CVSS scoring,
  new reference tags, rejections. Appending new pages alone would have mixed
  vintages; the pull now overlays all records modified since the cache was
  built. NVD paging is append-only (verified: 0 duplicate ids across vintages).
  Caveat: paging a `lastMod` query while NVD edits records can shift results
  (98,094 reported vs 98,016 unique), so a handful of records may keep their
  older version.
- **KEV and EPSS-current were never cached**, so a rebuild silently relabelled
  CVEs (a same-day partial rebuild had already swapped Nov KEV/EPSS onto Sep
  CVEs). The Sep 9 label set cannot be regenerated, only read from the snapshot.
  Both fetchers now keep dated raw copies.
- **Effect on A-results:** A2–A4 reproduce exactly (κ 0.179, 138 families, ARI
  0.53). A5 on the new L1: AMI 0.098 (was 0.095), vulnerabilities 81.5% `Low`
  (was 80.6%), overall 54.1/40.3/5.2/0.4 — thresholds still hold; 28 KEV
  findings in the estate (was 29). A6 EPSS leakage 49.5× → 5.7× point-in-time
  (was 45.7× → 5.0×) — same conclusion, protocol rule unchanged.
- **Also fixed while auditing Study A:** no experiment had been logged to MLflow
  (rule 6) — `src/tracking.py` now logs each A-step, store at
  `experiments/mlflow.db`; notebooks had never been executed and wrote to
  `notebooks/data/` when run from `notebooks/` (they now `chdir` to the root);
  `data/snapshots/` was not gitignored (rule 10).

### Environment

- **Python 3.12, not the system default.** `lightgbm`, `xgboost` and
  `umap-learn` ship no cp314 wheels, so 3.14 would force source builds.
  Use `/opt/homebrew/bin/python3.12 -m venv .venv`.
- **pandas pinned to 2.2.3, not 3.x** — `dice-ml` and `interpret` are not yet
  validated against pandas 3. `dice-ml` must be ≥0.12 (0.11 pins `pandas<2`),
  and `scikit-learn` ≥1.6 (required by `hdbscan` and `umap-learn`).

---

## Current Status

**STUDY A IS COMPLETE** (audited and re-run end to end 2026-11-23). The
evaluation protocol is locked and the analysis plan is pre-registered; the
protocol was amended once (November L1 refresh, counts only). Do not change
either without recording the deviation.

**Active study:** Study B — Exploitation Prediction & Baselines
**Current experiment:** B1 — baseline battery
**Current task:** CVSS ordering, severity label, recency, EPSS-current, random

**Read before writing any B1 code:**
- `reports/evaluation_protocol.md` — locked metrics, splits, EPSS rule
- `reports/analysis_plan_preregistration.md` — hypotheses fixed in advance,
  plus the KEV label-timing rule (registered 2026-11-24): train labels are KEV
  *as of the split date*, today's KEV only as a sensitivity check. Always
  build labels with `src/evaluation/labels.py::temporal_labels`.
- Use `src/evaluation/metrics.py` (`evaluate()`) and `src/evaluation/splits.py`
  so every model is scored and split identically.

**L3 label decision: RESOLVED 2026-09-09** (was the one open blocker for Study
C). EPSS percentile + explicit `misconfig_scale` + recalibrated thresholds;
the `source_layer` shortcut fell from AMI 0.289 to 0.095 and vulnerabilities
now spread 80.6/15.6/3.3/0.4 instead of 99.4% `Low`. Evidence table and full
reasoning in the A5 Deviations Log entry. **Worth walking supervisors through
at the next meeting** — it is a documented, defensible amendment to a formula
this file had declared fixed, not a silent change.

**A6 result:** EPSS leakage quantified and fixed via point-in-time archives
(49.5× → 5.7× on the November L1). L1 imbalance 1:240 (1,790 KEV of 430,678).
MLflow: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`.

**A3 result (carry into later studies):** Prowler vs ScoutSuite severity
agreement is only *slight* — Cohen's κ = 0.179, 95% CI [0.015, 0.330] on 161
semantically matched pairs, and κ stays in the "slight" band across every
matching threshold from 0.50 to 0.80. Checkov was excluded entirely: its
open-source repo ships no severities (they live in the commercial Bridgecrew
platform). Disagreement is systematic, not noise — e.g. Prowler rates all 13
matched CloudWatch alarm checks `medium` while ScoutSuite rates all 13
`danger`. **Implication: tool-assigned severity is not a reliable label
source**, which strengthens the CLAUDE.md rule that Study B labels come only
from CISA KEV. See `src/misconfig/normaliser.py` for the severity mapping and
`outputs/a3_*` for figures and tables.

**Completed:**
- [x] Project folder structure
- [x] requirements.txt
- [x] .gitignore
- [x] CLAUDE.md (this file)
- [x] A1: l1_vulnerabilities.csv
- [x] A2: l2_misconfig_catalog.csv
- [x] A3: disagreement analysis
- [x] A4: remediation clusters
- [x] A5: estate simulator
- [x] A6: EDA + evaluation protocol locked

**Update this checklist as experiments complete.**

---

## Supervisor Contact and Reporting

- **Mikhail Glazunov:** 20-minute meeting every 3 weeks — send agenda + 1-page progress note 24h before
- **Romulus-Cristinel Viesanu:** daily contact at Deutsche Bank office
- **Tripartite meeting:** every 6 weeks
- **Interim report #1:** end of September (Studies A + B complete)
- **Interim report #2:** end of October (Study C complete, SQ1 answered)
- **Full thesis draft:** mid-December
- **Final submission:** 1 February 2027
