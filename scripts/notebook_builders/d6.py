from common import *

write("notebooks/D6_human_study.ipynb", [
md("""# D6 — Human evaluation of explanation formats (materials only)

> **STATUS: NO PARTICIPANT DATA EXISTS.** This notebook shows the *materials* for the pre-registered human study. Nothing here simulates, imputes or invents participant responses. The analysis section stays empty until real sessions with CSO engineers have taken place, under the prerequisites in `reports/d6_human_study_protocol.md` §2: ethics approval, supervisor approval and informed consent.

**Design** (Study D addendum, commit c0a3ee1; full protocol in `reports/d6_human_study_protocol.md`):
- **Within-subject**, six explanation formats:
  - F0: score only (control);
  - F1: SHAP waterfall;
  - F2: LIME bars;
  - F3: group-level summary;
  - F4: EBM shape terms;
  - F5: counterfactual / EPSS trigger.
- 24 public CVEs in 3 rank strata (top-10, rank 11–500, below 500) × 8, split into 6 blocks.
- A 6×6 Williams Latin square balances format order and carry-over.
- **Measures:**
  - T1: forward simulation, "will the model place it in the top-100?";
  - T2: top factor;
  - escalation decision;
  - trust (1–7);
  - decision time.
- **Planned n = 12.** The minimum detectable paired effect is d_z 0.91 at α 0.05 and 1.19 at α 0.01 (power 0.8).
- **No internal data:** every stimulus is a public CVE (NVD + EPSS + CISA KEV).

Run: `python -m src.explainability.human_study` (needs D2 attributions and the D4 EBM)."""),
SETUP,
md("## (1) Williams Latin square: format per participant group × CVE block"),
code('pd.read_csv("outputs/d6_latin_square.csv", index_col=0)'),
md("## (2) Stimuli and answer key (never shown to participants)"),
code("""s = pd.read_csv("outputs/d6_stimuli.csv")
print(pd.crosstab(s.block, s.stratum))
s"""),
md("""Strata are balanced across blocks: each block has 4 CVEs, and every stratum appears 8 times in total. The answer key gives the correct T1 answer separately for the XGBoost model (used by formats F0–F3 and F5) and for the EBM (F4), because F4 explains a different model. The cards do not show KEV status."""),
md("## (3) One example card per format"),
code("""import re
from IPython.display import HTML
h = open("outputs/d6_stimulus_cards.html").read()
cards = re.findall(r"<section class='card'>.*?</section>", h, flags=re.S)
print(f"{len(cards)} cards in outputs/d6_stimulus_cards.html")
first = cards[:6]  # the first CVE in all six formats
HTML("<style>.card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}img{max-width:100%}</style>" + "".join(first))"""),
md("""## (4) Analysis — NOT RUN (no data)

Fixed in advance in protocol §7. Per participant × format, scores are averaged over that format's four trials.
- **Primary outcome:** T1 forward-simulation accuracy.
- **Across the six formats:** Friedman test, then Nemenyi.
- **Each format vs F0:** Wilcoxon signed-rank, Holm-adjusted over the five comparisons.
- **Effect sizes:** matched-pairs rank-biserial r, with 95% CI.
- **Secondary, labelled exploratory:** T2 accuracy, decision time, trust, appropriate reliance (T3 agreement with the KEV outcome), and trust calibration. Same tests, not Holm-adjusted across measures.
- **Exclusions:** a participant who fails both training checks or completes fewer than 18 of 24 trials.
- **Negative result** (protocol §8): no format beats F0 on T1 or T2. That would be a finding for SQ2, not a failure of the study.

This section is filled in only once real responses exist."""),
])
