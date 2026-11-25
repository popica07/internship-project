"""D6 - materials for the human evaluation study (protocol: reports/d6_human_study_protocol.md).

Generates, from PUBLIC L1 CVEs only:
- outputs/d6_stimuli.csv          24 stimulus CVEs (3 strata x 8, seed 0) + answer key
- outputs/d6_latin_square.csv     6x6 Latin square: participant group x block -> format
- outputs/d6_stimulus_cards.html  one card per CVE x format (144 cards, self-contained)

No participant data is produced or simulated here.

Usage:  python -m src.explainability.human_study   (after D2, D3, D4)
"""

import base64
import html
import io
import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.explainability import common as c  # noqa: E402
from src.explainability.shap_local import attributions, month_ranks  # noqa: E402

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR
FORMATS = {"F0": "Score only (control)", "F1": "SHAP waterfall", "F2": "LIME bar chart",
           "F3": "Counterfactual what-if", "F4": "EBM term contributions", "F5": "Plain-language summary"}
STRATA = {"top-10": (1, 10), "rank 11-500": (11, 500), "below 500": (501, 10 ** 9)}
PER_STRATUM, N_BLOCKS = 8, 6


def latin_square(n: int = 6) -> pd.DataFrame:
    """Williams-style balanced square for even n: rows = participant groups, cols = blocks."""
    first = [0]
    for k in range(1, n):
        first.append((first[-1] + (k if k % 2 else n - k)) % n)
    rows = [[(f + r) % n for f in first] for r in range(n)]
    keys = list(FORMATS)
    return pd.DataFrame([[keys[v] for v in row] for row in rows],
                        index=[f"group_{g + 1}" for g in range(n)], columns=[f"block_{b + 1}" for b in range(n)])


def stimuli() -> pd.DataFrame:
    attr = attributions()
    r = month_ranks().iloc[attr["index"]].copy()
    r["pos"] = np.arange(len(r))
    rng = np.random.default_rng(0)
    picks = []
    for name, (lo, hi) in STRATA.items():
        pool = r[(r.rank_in_month >= lo) & (r.rank_in_month <= hi)]
        take = pool.iloc[np.sort(rng.choice(len(pool), PER_STRATUM, replace=False))]
        picks.append(take.assign(stratum=name))
    s = pd.concat(picks)
    # blocks of four mixing strata: round-robin over the shuffled strata
    order = np.concatenate([rng.permutation(np.flatnonzero(s.stratum.to_numpy() == k)) for k in STRATA])
    s = s.iloc[order].copy()
    s["block"] = [f"block_{i % N_BLOCKS + 1}" for i in range(len(s))]
    return s.sort_values(["block", "rank_in_month"])


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _bars(values, names, title, unit) -> str:
    order = np.argsort(-np.abs(values))[:8][::-1]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(range(len(order)), values[order], color=["#eb6834" if values[j] > 0 else "#2a78d6" for j in order],
            height=0.6)
    ax.set_yticks(range(len(order)), [c.label(names[j]) for j in order], fontsize=8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel(f"{unit} (orange raises the score, blue lowers it)", fontsize=8)
    ax.set_title(title, fontsize=9)
    return _png(fig)


def _top3_sentence(phi, names, rank, size) -> str:
    order = np.argsort(-np.abs(phi))[:3]
    parts = [f"{c.label(names[j])} ({'raises' if phi[j] > 0 else 'lowers'} the score)" for j in order]
    return (f"Ranked #{rank} of {size} new CVEs this month. Main reason: {parts[0]}; "
            f"then {parts[1]}; then {parts[2]}.")


def build() -> dict:
    attr = attributions()
    names = c.feature_names()
    d = c.data()
    s = stimuli()
    from src.explainability.counterfactuals import epss_triggers

    trig, _ = epss_triggers(s.index.to_numpy())  # same D3 L1 rule, for the stimulus CVEs
    trig = trig.set_index("cve_id")
    ebm = pd.read_pickle(c.CACHE_DIR / "d4_ebm_s0.pkl").m
    ebm_pred = pd.read_pickle(c.CACHE_DIR / "d4_test_predictions.pkl")["ebm__s0"].to_numpy()
    ebm_rank = pd.Series(ebm_pred).groupby(d["group_eval"]).rank(ascending=False, method="first").to_numpy()
    r_all = month_ranks()
    tau100_month = r_all.groupby("month").score.apply(lambda v: np.sort(v.to_numpy())[::-1][min(99, len(v) - 1)])

    key, cards = [], []
    for i, row in s.iterrows():
        k = row.pos
        x = d["X_eval"].iloc[i]
        phi, lime = attr["shap_xgboost"][k], attr["lime_xgboost"][k]
        grp = pd.Series(phi, index=names).groupby([c.group_of(n) for n in names]).sum()
        local = ebm.explain_local(d["X_eval"].iloc[[i]])
        ebm_terms = np.asarray(local.data(0)["scores"], float)
        ebm_names = list(local.data(0)["names"])
        key.append({"cve_id": row.cve_id, "month": row.month, "stratum": row.stratum, "block": row.block,
                    "kev": int(row.y), "xgb_rank": int(row.rank_in_month), "month_size": int(row.month_size),
                    "T1_in_top100_xgb": bool(row.rank_in_month <= 100),
                    "T1_in_top100_ebm": bool(ebm_rank[i] <= 100), "ebm_rank": int(ebm_rank[i]),
                    "T2_top_factor_F1_F5": c.label(names[int(np.argmax(np.abs(phi)))]),
                    "T2_top_factor_F2": c.label(names[int(np.argmax(np.abs(lime)))]),
                    "T2_top_factor_F4": ebm_names[int(np.argmax(np.abs(ebm_terms)))],
                    "T2_top_group_F0_F3": grp.abs().idxmax()})
        head = (f"<b>{html.escape(row.cve_id)}</b> · scoring month {row.month} · CVSS {x.cvss_score:.1f} "
                f"(v{x.cvss_version:.1f}) · EPSS {x.epss_pit:.4f} (percentile {x.epss_pit_pct:.2f})")
        score = f"Model score {row.score:.4f} — rank {int(row.rank_in_month)} of {int(row.month_size)} this month"
        body = {
            "F0": f"<p>{score}</p>",
            "F1": f"<p>{score}</p><img src='data:image/png;base64,"
                  f"{_bars(phi, names, 'Why this score (SHAP, model log-odds)', 'log-odds')}'>",
            "F2": f"<p>{score}</p><img src='data:image/png;base64,"
                  f"{_bars(lime, names, 'Why this score (LIME, local surrogate)', 'P(KEV) weight')}'>",
            "F3": f"<p>{score}</p><p>" + (
                "Already in the top-100 this month." if row.rank_in_month <= 100 else
                (f"Would enter this month's top-100 if its EPSS rose from {trig.loc[row.cve_id, 'epss_now']:.4f} to "
                 f"{trig.loc[row.cve_id, 'epss_needed']:.4f}." if row.cve_id in trig.index and
                 trig.loc[row.cve_id, "reachable"] else
                 "No EPSS value would bring it into this month's top-100: its other properties hold it down."
                 if row.cve_id in trig.index else
                 f"Top-100 cut-off this month: score {tau100_month[row.month]:.4f}.")) + "</p>",
            "F4": f"<p>EBM score {ebm_pred[i]:.4f} — rank {int(ebm_rank[i])} of {int(row.month_size)} this month "
                  f"(a different, glass-box model)</p><img src='data:image/png;base64,"
                  f"{_bars(ebm_terms, ebm_names, 'Why this score (EBM terms, log-odds)', 'log-odds')}'>",
            "F5": f"<p>{_top3_sentence(phi, names, int(row.rank_in_month), int(row.month_size))}</p>",
        }
        for f, content in body.items():
            cards.append(f"<section class='card'><h3>{f} · {FORMATS[f]}</h3><p class='head'>{head}</p>{content}"
                         "<ol class='q'><li>Will the model place this CVE in this month's top-100? (yes / no)</li>"
                         "<li>Which factor contributed most to this score?</li>"
                         "<li>Would you escalate this CVE for patching this month? (yes / no)</li>"
                         "<li>How much do you trust this ranking? (1–7)</li></ol></section>")
    key = pd.DataFrame(key)
    key.to_csv(OUT / "d6_stimuli.csv", index=False)
    square = latin_square()
    square.to_csv(OUT / "d6_latin_square.csv")
    page = ("<!doctype html><meta charset='utf-8'><title>D6 stimulus cards</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222}"
            ".card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.2rem 0;page-break-inside:avoid}"
            ".head{color:#555;font-size:.9rem}img{max-width:100%}.q{font-size:.85rem;color:#444}</style>"
            "<h1>D6 stimulus cards (public CVEs only)</h1><p>24 CVEs × 6 formats. Each participant sees each CVE "
            "once, in the format given by <code>d6_latin_square.csv</code> for their group and the CVE's block. "
            "Answer key: <code>d6_stimuli.csv</code>.</p>" + "".join(cards))
    (OUT / "d6_stimulus_cards.html").write_text(page, encoding="utf-8")
    return {"stimuli": key, "square": square, "n_cards": len(cards)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = build()
    print(out["square"].to_string())
    print(out["stimuli"].groupby(["block", "stratum"]).size().unstack(fill_value=0))
    print("cards:", out["n_cards"])
