# Prioritising Cloud Security Findings with ML and Explainable AI

Security teams in large organisations get far more vulnerability and
misconfiguration findings than they can fix. Most are never exploited, a few
are, and the hard part is telling them apart early enough to act. This
repository is my Master's internship project, done at the Deutsche Bank
Chief Security Office and Maastricht University (DACS). It asks a simple
question:

> **Can machine learning, together with explainable AI, help security
> engineers decide which findings to fix first, and can they understand why
> a model ranks a finding high?**

Everything here runs on **public data only**. No bank data, assets or
configurations were used or are needed. The pipeline is designed so that an
organisation could run it inside its own environment and plug in its own
context, without any data leaving.

---

## What I found, in short

**1. Point-in-time EPSS is very hard to beat.** EPSS is a public exploit
prediction score. Used naively, it leaks the future: today's score already
"knows" which CVEs were later exploited, which inflated results about 49.5×.
Using EPSS as it was *on the day* removes most of that leak (5.7×). Once that
is done honestly:
- learned models beat CVSS-based prioritisation;
- but they only match point-in-time EPSS month by month.

**2. Simple models are enough.** Logistic regression matched or beat XGBoost,
LightGBM, CatBoost, learning-to-rank objectives and deep tabular models
(TabNet, FT-Transformer). The deep models cost about 100× the compute for no
measurable gain.

**3. Explanations disagree, and they can mislead.**
- SHAP and LIME rarely agree on which features matter for a given finding
  (median Kendall τ 0.35).
- With a training-era SHAP background, about half of each explanation
  describes *when* a CVE was published (its CVSS version) rather than the
  CVE itself.
- On simulated data, where the true labelling rule is known, SHAP gave a
  correlated feature that the rule never uses twice the credit of the
  feature the rule actually uses.
- Practical lesson: draw the SHAP background from the data you are scoring,
  and treat explanations as hypotheses, not facts.

**4. Interpretability is free here.** Glass-box models (EBM, RuleFit) are
statistically indistinguishable from the black boxes. That's partly because
the best model, logistic regression, is already transparent.

**5. Security tools disagree with each other.** Prowler and ScoutSuite agree
only slightly on severity for the same issue (Cohen's κ = 0.18). This is why
labels come from real exploitation data (the CISA KEV catalogue) rather than
tool severities.

**Hypotheses were written down before the experiments that test them**
(`reports/analysis_plan_preregistration.md`). Negative results are reported
as negative.

---

## The data

| Layer | What it is | Source |
|---|---|---|
| **L1** Vulnerabilities | 430k CVEs with CVSS, descriptions, EPSS history and exploitation labels | NVD, FIRST EPSS, CISA KEV (all public) |
| **L2** Misconfigurations | about 3,900 cloud checks from three open-source scanners | Prowler, Checkov, ScoutSuite repos |
| **L3** Cloud estate | assets, exposure, IAM, SLAs and regulatory scope for a fictional bank | **simulated**, with a documented formula |

L3 stands in for the internal context a real organisation would have. It is
simulated because no internal data could be used. Its labelling formula is
fixed and documented, which also makes it useful for checking whether
explanations recover the true rule.

Models are trained on CVEs published before **1 January 2023** and tested on
everything after (226k CVEs, 309 later exploited). The split is never
shuffled, because a real team never gets to see the future.

---

## The four studies

| Study | Question | Notebooks |
|---|---|---|
| **A** Data foundation | Build the three data layers and lock the evaluation protocol | `A1`–`A6` |
| **B** Exploitation prediction | Can we predict which CVEs get exploited, and beat CVSS? | `B1`–`B6` |
| **C** Model comparison | Which model family works best, and does it hold up statistically? | `C1`–`C6`, `C1R` |
| **D** Explainability | Do explanations agree, are they stable, and are they actionable? | `D1`–`D6` |

D6 is a planned study with security engineers. The protocol and materials
are ready (`reports/d6_human_study_protocol.md`), but **no participant data
exists yet**, and none is simulated.

---

## Running it

You need **Python 3.12**. Some pinned packages, including `lightgbm`,
`xgboost` and `umap-learn`, have no wheels yet for newer versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost and LightGBM also need OpenMP: `brew install libomp`.

Optionally, copy `.env.example` to `.env` and add a free
[NVD API key](https://nvd.nist.gov/developers/request-an-api-key). It makes
the NVD download about 10× faster.

**Getting the data (Study A):**

```bash
python -m src.ingestion.join          # A1: NVD + EPSS + KEV  ->  data/processed/l1_vulnerabilities.csv
python -m src.misconfig.normaliser    # A2: clone the scanners and extract their checks
```

The public feeds change every day, so a fresh download gives slightly
different numbers from the ones in the reports. The datasets used for the
reported results are frozen in `data/snapshots/`.

**Experiments:** each one is a module under `src/`, for example:

```bash
python -m src.models.b2_kev
python -m src.models.c1_revised --stage all
python -m src.explainability.shap_global
```

The notebooks in `notebooks/` read the saved outputs and walk through each
result. They don't retrain anything, so they open quickly. Runs are tracked
in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```

A full re-run from the frozen data takes about 30 hours on a laptop. The
deep tabular models and the sparse decision trees are the slow parts.

---

## Repository layout

```
src/
  ingestion/       NVD, EPSS and KEV download and join
  misconfig/       scanner check extraction, cross-tool matching, clustering
  simulator/       the simulated cloud estate (L3)
  evaluation/      metrics, temporal splits, labels, statistical tests
  models/          one module per experiment (b2_kev.py, c3_deep.py, ...)
  explainability/  SHAP, LIME, counterfactuals, stability, human-study materials
notebooks/         one notebook per experiment, reading saved results
reports/           evaluation protocol, pre-registration, interim reports
scripts/           notebook builders
data/, outputs/, experiments/   generated locally, not tracked in git
```

[`CLAUDE.md`](./CLAUDE.md) is the detailed working log. It covers the
locked protocol and the project rules, and every deviation from the plan
with the reason for it. If you want to know *why* something was done a
certain way, it's probably explained there.

---

## Limitations, honestly

- **The estate layer (L3) is simulated.** Findings about it show that the
  method works, not how well it would work on real data. Real labels, such
  as analyst decisions or incident history, would be noisier.
- **Exploitation labels are incomplete.** CISA KEV lists exploitation that
  has been *observed*; a CVE missing from KEV is not necessarily safe.
- **The human evaluation (D6) has not been run yet**, so the question of
  whether explanations help engineers is only partly answered.

---

*Master AI internship, Maastricht University (DACS), in collaboration with
the Deutsche Bank Chief Security Office, 2026–2027. This is academic
research, not a Deutsche Bank product, and it contains no Deutsche Bank
data.*
