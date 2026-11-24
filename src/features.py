"""Point-in-time L1 feature construction shared by Study B (B2-B6).

Implements the Study B design addendum in
reports/analysis_plan_preregistration.md. Two ideas carry the weight:

1. A SCORING DATE per CVE. Training CVEs are scored as of the split date T
   (everything a model built at T could see). Test CVEs are scored as of the
   first monthly EPSS snapshot on/after their publication - when an analyst
   would actually first triage them.
2. Features are read AS OF that date. EPSS comes from the archived daily file
   for that date, never from today's snapshot (A6: today's EPSS leaks KEV).

Primary features: CVSS score / severity / version, description text (TF-IDF ->
SVD, fitted on training text only), point-in-time EPSS. NVD reference-tag
features (public_exploit, has_patch) are opt-in: they reflect today's tags and
are a sensitivity check only.
"""

import gzip
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation.labels import temporal_labels
from src.evaluation.splits import temporal_split
from src.ingestion.epss import fetch_epss_asof

logger = logging.getLogger(__name__)

L1_PATH = Path("data/processed/l1_vulnerabilities.csv")
EPSS_DIR = Path("data/raw/epss")
# The most recent EPSS file on disk: the one A1 downloaded with the L1 refresh.
LATEST_EPSS = ("2026-11-23", EPSS_DIR / "epss_scores-current-downloaded-2026-11-23.csv.gz")
FIRST_MONTHLY_SNAPSHOT = "2021-05-01"

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
VERSION_CODE = {"3.1": 3.1, "3.0": 3.0, "2.0": 2.0}
L1_COLUMNS = [
    "cve_id", "published_date", "description", "cvss_score", "cvss_severity",
    "cvss_version", "public_exploit", "has_patch", "epss_score", "in_kev", "kev_date_added",
]


def load_l1() -> pd.DataFrame:
    df = pd.read_csv(L1_PATH, usecols=L1_COLUMNS, dtype={"cvss_version": str}, low_memory=False)
    df["published_date"] = pd.to_datetime(df["published_date"], format="ISO8601", utc=True)
    df["kev_date_added"] = pd.to_datetime(df["kev_date_added"], format="ISO8601", utc=True)
    if df["published_date"].isna().any():
        raise ValueError("unparsed published_date in L1")
    if df["kev_date_added"].notna().sum() != df["in_kev"].sum():
        raise ValueError("kev_date_added does not match in_kev")
    return df


@lru_cache(maxsize=1)
def snapshot_dates() -> pd.DatetimeIndex:
    """1st of each month, or the next day that has an archive (2024-12-01 is
    missing upstream). Resolved once; downloads only what is not cached."""
    resolved = []
    for month in pd.date_range(FIRST_MONTHLY_SNAPSHOT, LATEST_EPSS[0], freq="MS", tz="UTC"):
        for offset in range(5):
            day = (month + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                fetch_epss_asof(day)
            except Exception:  # noqa: BLE001 - a missing day upstream; try the next
                continue
            resolved.append(pd.Timestamp(day, tz="UTC"))
            break
        else:
            logger.warning("No EPSS archive in the first 5 days of %s", month.date())
    resolved.append(pd.Timestamp(LATEST_EPSS[0], tz="UTC"))
    return pd.DatetimeIndex(resolved)


def scoring_date_after_publication(published: pd.Series) -> pd.Series:
    """First EPSS snapshot on or after publication; NaT if none exists yet."""
    snapshots = snapshot_dates()
    positions = snapshots.searchsorted(published.to_numpy(), side="left")
    dates = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns, UTC]")
    valid = positions < len(snapshots)
    dates[valid] = snapshots[positions[valid]]
    return dates


@lru_cache(maxsize=None)
def _epss_on(date: str) -> pd.DataFrame:
    if date == LATEST_EPSS[0]:
        with gzip.open(LATEST_EPSS[1]) as handle:
            df = pd.read_csv(handle, comment="#").rename(
                columns={"cve": "cve_id", "epss": "epss_score", "percentile": "epss_percentile"}
            )
        return df[["cve_id", "epss_score", "epss_percentile"]]
    return fetch_epss_asof(date)[["cve_id", "epss_score", "epss_percentile"]]


def epss_as_of(cve_ids: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """EPSS score/percentile for each CVE as published on its own date.

    A CVE absent from that day's file (not yet scored, or a date before EPSS
    existed) gets NaN - missing, never back-filled from a later date.
    """
    out = pd.DataFrame(
        {"epss_pit": np.nan, "epss_pit_pct": np.nan}, index=cve_ids.index
    )
    for date, index in dates.dropna().groupby(dates.dropna()).groups.items():
        if date.year < 2021:
            continue  # no EPSS archive exists before 2021: stays missing
        snapshot = _epss_on(date.strftime("%Y-%m-%d")).set_index("cve_id")
        found = snapshot.reindex(cve_ids.loc[index])
        out.loc[index, "epss_pit"] = found["epss_score"].to_numpy()
        out.loc[index, "epss_pit_pct"] = found["epss_percentile"].to_numpy()
    return out


def kev_known_by(df: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """1 if the CVE was already in KEV on its scoring date."""
    return (df["kev_date_added"].notna() & (df["kev_date_added"] <= dates)).astype(int)


class FeatureBuilder:
    """Text model fitted on training rows only; numeric features are stateless."""

    def __init__(self, use_epss: bool = True, use_reference_tags: bool = False,
                 n_text_components: int = 64, seed: int = 0):
        self.use_epss = use_epss
        self.use_reference_tags = use_reference_tags
        self.n_text_components = n_text_components
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.tfidf = TfidfVectorizer(
            max_features=30_000, min_df=5, ngram_range=(1, 2), sublinear_tf=True,
            stop_words="english", dtype=np.float64,
        )
        text = self.tfidf.fit_transform(train["description"].fillna(""))
        self.svd = TruncatedSVD(self.n_text_components, random_state=self.seed).fit(text)
        return self

    def transform(self, df: pd.DataFrame, scoring_dates: pd.Series | None = None) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["cvss_score"] = df["cvss_score"]
        X["cvss_severity"] = df["cvss_severity"].map(SEVERITY_ORDER)
        X["cvss_version"] = df["cvss_version"].map(VERSION_CODE)
        text = self.svd.transform(self.tfidf.transform(df["description"].fillna("")))
        for i in range(text.shape[1]):
            X[f"text_{i:02d}"] = text[:, i]
        if self.use_epss:
            if scoring_dates is None:
                raise ValueError("EPSS features need scoring dates")
            X = X.join(epss_as_of(df["cve_id"], scoring_dates))
        if self.use_reference_tags:
            X["public_exploit"] = df["public_exploit"]
            X["has_patch"] = df["has_patch"]
        return X


def temporal_task(
    df: pd.DataFrame, split_date: str, test_end: str | None = None
) -> dict:
    """Train/test partitions with scoring dates and the 'already known' filter.

    Test rows already in KEV on their scoring date, or with no snapshot yet,
    are dropped (counts returned) - there is nothing left to predict for them.
    """
    train, test = temporal_split(df, split_date)  # CLAUDE.md rule 2, enforced in one place
    train, test = train.copy(), test.copy()
    if test_end is not None:
        test = test[test["published_date"] < pd.Timestamp(test_end, tz="UTC")]
    if train["published_date"].max() >= test["published_date"].min():
        raise ValueError("temporal leakage between train and test")

    train_dates = pd.Series(pd.Timestamp(split_date, tz="UTC"), index=train.index)
    test_dates = scoring_date_after_publication(test["published_date"])
    no_snapshot = test_dates.isna()
    already_known = kev_known_by(test, test_dates).astype(bool) & ~no_snapshot
    keep = ~(no_snapshot | already_known)

    # Labels only via the pre-registered rule (src/evaluation/labels.py).
    y_train, y_test = temporal_labels(train, test[keep], split_date, mode="asof_split")
    y_train_today, _ = temporal_labels(train, test[keep], split_date, mode="today")

    return {
        "train": train, "train_dates": train_dates,
        "test": test[keep], "test_dates": test_dates[keep],
        "y_train": y_train,              # KEV as of T (primary)
        "y_train_today": y_train_today,  # sensitivity
        "y_test": y_test,
        "n_excluded_already_known": int(already_known.sum()),
        "n_excluded_kev_already_known": int((already_known & (test["in_kev"] == 1)).sum()),
        "n_excluded_no_snapshot": int(no_snapshot.sum()),
    }
