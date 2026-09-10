"""Data splitting, with the project's two hard rules enforced in code.

CLAUDE.md rule 2: L1 is ALWAYS split temporally - train on CVEs published
before date T, test on those after. Random shuffling time-ordered vulnerability
data leaks future exploitation information and invalidates the result.

L3 is split grouped-by-asset, so the same asset never appears in both train and
test (findings on one asset share its context features and are not independent).

Both functions refuse to return a silently broken split: they raise rather than
hand back an empty or leaking partition.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    split_date: str | pd.Timestamp,
    date_column: str = "published_date",
    min_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split L1 on publication date. Train strictly before, test on/after."""
    dates = pd.to_datetime(df[date_column], format="ISO8601", utc=True, errors="coerce")
    unparsed = dates.isna().mean()
    if unparsed > 0.01:
        raise ValueError(
            f"{unparsed:.1%} of {date_column} failed to parse - refusing to split on "
            "dates that may be silently NaT (see CLAUDE.md Deviations Log, A1)."
        )

    boundary = pd.Timestamp(split_date, tz="UTC") if not isinstance(split_date, pd.Timestamp) else split_date
    train = df[dates < boundary]
    test = df[dates >= boundary]

    if len(train) < min_rows or len(test) < min_rows:
        raise ValueError(
            f"Temporal split at {boundary.date()} gives train={len(train)}, test={len(test)}; "
            f"both must exceed {min_rows}."
        )

    logger.info(
        "Temporal split at %s: train=%d (to %s), test=%d (from %s)",
        boundary.date(), len(train), dates[dates < boundary].max().date(),
        len(test), dates[dates >= boundary].min().date(),
    )
    return train, test


def grouped_split(
    df: pd.DataFrame,
    group_column: str = "asset_id",
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split L3 by asset, so no asset spans train and test."""
    rng = np.random.default_rng(seed)
    groups = df[group_column].unique()
    rng.shuffle(groups)

    n_test = max(1, int(len(groups) * test_size))
    test_groups = set(groups[:n_test])

    is_test = df[group_column].isin(test_groups)
    train, test = df[~is_test], df[is_test]

    overlap = set(train[group_column]) & set(test[group_column])
    if overlap:
        raise ValueError(f"Grouped split leaked {len(overlap)} groups across train/test.")

    logger.info(
        "Grouped split on %s: train=%d rows / %d groups, test=%d rows / %d groups",
        group_column, len(train), len(groups) - n_test, len(test), n_test,
    )
    return train, test


def assert_no_temporal_leakage(
    train: pd.DataFrame, test: pd.DataFrame, date_column: str = "published_date"
) -> None:
    """Fail loudly if any training row is dated at/after the earliest test row."""
    train_dates = pd.to_datetime(train[date_column], format="ISO8601", utc=True, errors="coerce")
    test_dates = pd.to_datetime(test[date_column], format="ISO8601", utc=True, errors="coerce")
    if train_dates.max() >= test_dates.min():
        raise ValueError(
            f"Temporal leakage: latest train row ({train_dates.max()}) is not before "
            f"earliest test row ({test_dates.min()})."
        )
