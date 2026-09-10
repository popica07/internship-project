# Cloud Security Finding Prioritisation — ML & XAI

Master AI internship (Deutsche Bank CSO / Maastricht University DACS)
investigating whether ML and explainable AI can support prioritisation of
cloud security and vulnerability findings in an enterprise financial
environment.

**Read [`CLAUDE.md`](./CLAUDE.md) first** — it is the single source of truth
for project identity, the four studies (24 experiments), the locked
evaluation protocol, and every non-negotiable rule (temporal splits only,
KEV as the only real label source, etc.).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 is required (not the OS default) — some pinned packages
(`lightgbm`, `xgboost`, `umap-learn`) do not yet ship prebuilt wheels for
newer Python versions.

Optional: copy `.env.example` to `.env` and set `NVD_API_KEY` to raise the
NVD API rate limit from ~5 to ~50 requests/30s.

## Study A — A1: Vulnerability Ingestion

```bash
python -m src.ingestion.join
```

Downloads NVD (API 2.0), EPSS, and CISA KEV, joins them on CVE ID, and
writes `data/processed/l1_vulnerabilities.csv`, sorted by CVE publication
date. Logs total CVEs, KEV count, class imbalance ratio, and date range on
completion. See `notebooks/A1_ingestion.ipynb` for an interactive run.
