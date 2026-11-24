"""KEV labels as of a point in time (pre-registered decision, 2026-11-24).

`in_kev` in l1_vulnerabilities.csv is KEV membership *today*. Under a temporal
split that is not what a model built at the split date could have known: 174
CVEs published before 2023-01-01 entered KEV only after it. Labelling them
positive in training uses knowledge from after the split.

PRIMARY (all Study B headline results):
    train label = in KEV with kev_date_added <  split date
    test label  = in KEV as of the data snapshot (today)
SENSITIVITY (reported alongside, never instead):
    train label = in KEV as of today

Test labels always use the latest KEV: the test asks "which of these CVEs
turned out to be exploited", and the more time KEV has had to observe that,
the better the ground truth.
"""

import pandas as pd


def kev_label_asof(df: pd.DataFrame, asof: str | pd.Timestamp) -> pd.Series:
    """1 if the CVE had been added to KEV strictly before `asof`, else 0."""
    boundary = pd.Timestamp(asof, tz="UTC") if not isinstance(asof, pd.Timestamp) else asof
    added = pd.to_datetime(df["kev_date_added"], format="ISO8601", utc=True, errors="coerce")
    # Same rule as everywhere else: a parse failure must be loud, not a silent 0.
    unparsed = added.isna() & df["kev_date_added"].notna()
    if unparsed.any():
        raise ValueError(f"{int(unparsed.sum())} kev_date_added values failed to parse")
    return (added < boundary).astype(int)


def temporal_labels(
    train: pd.DataFrame, test: pd.DataFrame, split_date: str, mode: str = "asof_split"
) -> tuple[pd.Series, pd.Series]:
    """Train/test KEV labels under the pre-registered rule.

    mode="asof_split" is the primary analysis; mode="today" is the sensitivity
    check. Anything else raises, so a typo cannot silently pick a labelling.
    """
    if mode == "asof_split":
        y_train = kev_label_asof(train, split_date)
    elif mode == "today":
        y_train = train["in_kev"].astype(int)
    else:
        raise ValueError(f"unknown label mode {mode!r}; use 'asof_split' or 'today'")
    return y_train, test["in_kev"].astype(int)
