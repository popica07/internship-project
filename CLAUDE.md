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
│   │   ├── prowler_extractor.py ← extract checks from Prowler repo
│   │   ├── checkov_extractor.py ← extract checks from Checkov repo
│   │   └── normaliser.py        ← unified schema normalisation
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

---

## Data Sources — URLs and Access

```python
# L1 — Vulnerability Intelligence
NVD_BULK_URL = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
NVD_API_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
EPSS_URL     = "https://epss.cyentia.com/epss_scores-current.csv.gz"
KEV_URL      = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# L2 — Cloud Misconfiguration Catalogs (clone from GitHub)
PROWLER_REPO   = "https://github.com/prowler-cloud/prowler"
CHECKOV_REPO   = "https://github.com/bridgecrewio/checkov"
SCOUTSUITE_REPO = "https://github.com/nccgroup/ScoutSuite"
```

---

## L3 Simulator — Label Formula (documented and fixed)

```python
# Priority score = 60% real component + 40% context component
# Real component (from L1 — external, not invented):
#   For vulnerabilities: 0.40 * epss_score + 0.35 * in_kev + 0.25 * public_exploit
#   For misconfigurations: severity_map = {critical:1.0, high:0.75, medium:0.5, low:0.25}

# Context component (from L3 — simulated, documented):
#   0.30 * tier_weight[criticality_tier]   # Tier 0=1.0, Tier 4=0.0
#   0.20 * internet_facing                  # boolean
#   0.20 * (pci_scope OR dora_scope)        # regulatory boolean
#   0.15 * sla_breached                     # boolean
#   0.15 * min(recurrence_count / 5, 1.0)  # capped at 1.0

# Label thresholds:
#   score >= 0.80 → Critical
#   score >= 0.60 → High
#   score >= 0.40 → Medium
#   score <  0.40 → Low

# All weights are tunable parameters swept in experiment C7
# Label noise injection controlled by noise_level parameter (0.0 to 0.40)
```

---

## Current Status

**Active study:** Study A — Data Foundation
**Current experiment:** A1 — NVD + EPSS + KEV ingestion
**Current task:** Set up project structure and pull first data snapshot

**Completed:**
- [ ] Project folder structure
- [ ] requirements.txt
- [ ] .gitignore
- [ ] CLAUDE.md (this file)
- [ ] A1: l1_vulnerabilities.csv
- [ ] A2: l2_misconfig_catalog.csv
- [ ] A3: disagreement analysis
- [ ] A4: remediation clusters
- [ ] A5: estate simulator
- [ ] A6: EDA + evaluation protocol locked

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
