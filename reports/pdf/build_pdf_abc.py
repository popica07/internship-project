"""Progress report PDF: Studies A, B and C, mapped against the internship plan."""
import os
import sys

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

PROJECT, SCRATCH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FIG = os.path.join(PROJECT, "outputs")

fonts = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fonts, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(fonts, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fonts, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, INK2, ACCENT, RULE, SOFT = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#1f5fa8", "#d9d8d3", "#f3f2ee"))
GOOD, PART, TODO = colors.HexColor("#1c7a4a"), colors.HexColor("#a86400"), colors.HexColor("#a33a3a")

base = dict(fontName="DV", textColor=INK, leading=13.2, fontSize=9.6)
S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=19, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=18, textColor=ACCENT, spaceBefore=12, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10.8, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", spaceAfter=5, alignment=TA_LEFT, **base),
    "bullet": ParagraphStyle("bullet", leftIndent=12, bulletIndent=2, spaceAfter=2.5, **base),
    "cap": ParagraphStyle("cap", fontName="DVI", fontSize=8.2, leading=10.5, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.3, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8.3, leading=10.4, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.4, leading=13, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, extra=None):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
             ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), SOFT))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def callout(text):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def figure(path, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, "cap")])


def status(word):
    c = {"Done": GOOD, "Partly": PART, "Not yet": TODO}[word]
    return Paragraph(f'<font color="{c.hexval()}"><b>{word}</b></font>', S["cell"])



def verdict(word):
    c = {"Supported": GOOD, "Not supported": TODO, "Partly": PART, "Pending": INK2}[word]
    return Paragraph(f'<font color="{c.hexval()}"><b>{word}</b></font>', S["cell"])


story = []
story += [
    P("Progress Report — Studies A, B and C", "title"),
    P("Cloud Security Finding Prioritisation with Machine Learning and Explainable AI", "sub"),
    P("Popa Stefan Andrei (6415250) · Master AI, Maastricht University (DACS) · Deutsche Bank CSO", "sub"),
    P("Supervisors: Mikhail Glazunov (UM DACS) · Romulus-Cristinel Viesanu (Deutsche Bank CSO)", "sub"),
    P("25 November 2026", "sub"),
    Spacer(1, 8),
]

# ---------------------------------------------------------------- 1. summary
story += [P("1. Summary", "h1")]
story += [callout(
    "<b>Studies A, B and C are complete.</b> Study A built and audited the data. Study B tested whether machine "
    "learning predicts which vulnerabilities get exploited. Study C compared 15 models from five method families "
    "under one protocol. Hypotheses and analysis plans were <b>pre-registered before each study ran</b>, and every "
    "result is the mean of 10 seeds on the same untouched test set (309 exploited CVEs among 226,463).<br/><br/>"
    "<b>Headline:</b> machine learning ranks exploited vulnerabilities clearly better than sorting by CVSS, today's "
    "default practice. A simple, transparent <b>logistic regression is as good as anything more complex</b>: "
    "gradient boosting, learning-to-rank objectives and deep networks add no measurable value. The signal comes "
    "mostly from <b>point-in-time EPSS</b>; the best model beats EPSS on overall PR-AUC, but not month by month "
    "once multiple comparisons are corrected. The explainability study (D) can therefore centre on a model that "
    "is accurate <i>and</i> interpretable.")]
story += [Spacer(1, 6)]
story += [table([
    ["Question", "Answer", "Evidence"],
    ["Does ML beat CVSS ordering? (H1)", "<b>Yes</b>", "PR-AUC +0.0095 [0.0033, 0.0227], Precision@100 +0.08 [0.02, 0.15]; holds for all 15 Study C models"],
    ["Which model family is best? (H2)", "<b>Logistic regression</b> (tied top group)", "LR 0.0235 vs XGBoost 0.0220: boosted trees do not beat LR (−0.0013 [−0.0070, 0.0031])"],
    ["Does a ranking objective help? (H3)", "<b>No</b>", "LambdaMART − pointwise, monthly NDCG@100 +0.0049 [−0.0061, 0.0158]"],
    ["Do deep models help? (H4)", "<b>No</b>", "FT-Transformer − XGBoost +0.0003 [−0.0043, 0.0087], at ~100× the compute"],
    ["Which features matter?", "<b>EPSS, then CVSS</b>", "Without EPSS every model falls to CVSS level; CVSS + EPSS ≈ all features"],
    ["Does it beat point-in-time EPSS?", "Overall yes; monthly no", "LR +0.0060 [0.0013, 0.0121] PR-AUC; no model better month by month after Holm"],
    ["Does performance decay over time? (H5)", "Same direction, <b>not significant</b>", "Spearman ρ = −0.53, p = 0.14 (9 cells)"],
    ["Are the probabilities usable?", "Yes", "Near-calibrated raw; Platt scaling repairs the rest (C5)"],
], [4.6 * cm, 3.6 * cm, 8.8 * cm])]

# ---------------------------------------------------------------- 2. plan alignment
story += [P("2. Alignment with the internship plan", "h1"),
          P("Each element of the internship plan, mapped to what Studies A–C delivered. Items marked "
            "<i>Partly</i> or <i>Not yet</i> are addressed in Study D or listed for discussion (section 8).")]
story += [table([
    ["Internship plan element", "Status", "Where / how"],
    ["Search, collect and pre-process cloud security and vulnerability data", status("Done"),
     "A1: NVD, EPSS and CISA KEV joined (430,678 CVEs). A2: Prowler, Checkov and ScoutSuite rule catalogs (3,870 checks)."],
    ["Clean data, handle missing or inconsistent values, combine sources", status("Done"),
     "Unified schema; silent date-parsing corruption and stale caches found and fixed (A1, logged)."],
    ["Exploratory analysis and visualisation", status("Done"), "A6 EDA; KEV trends; EPSS leakage analysis; tool-disagreement figures."],
    ["Risk classification", status("Done"), "B2 and C1–C3: binary KEV prediction under a strict temporal split."],
    ["Vulnerability prioritisation (ranking)", status("Done"),
     "B4 and C2: ranking each month's queue (NDCG, MAP, MRR, Precision@k), including learning-to-rank objectives."],
    ["Recurring issue detection", status("Partly"),
     "A4: 138 recurring remediation families across tools. Recurrence is simulated in L3; no dedicated recurrence model."],
    ["Remediation urgency estimation", status("Partly"),
     "B6: cost-sensitive thresholds and monthly budgets; C5: calibrated probabilities. SLA-based urgency on L3 not modelled."],
    ["Methods: LR, decision trees, random forests, gradient boosting, clustering", status("Done"),
     "All six classical families tuned and compared (C1); HDBSCAN clustering (A4)."],
    ["Deep learning compared with simpler models", status("Done"), "C3: MLP, TabNet, FT-Transformer vs gradient boosting."],
    ["Metrics: precision, recall, F1, ROC-AUC, confusion matrices; ranking metrics", status("Done"),
     "All reported. PR-AUC and Precision@k primary (1:240 imbalance); ROC-AUC secondary. Confusion matrices §4.5."],
    ["SQ: Which features are most predictive?", status("Done"), "C4 feature ablation (EPSS dominant). SHAP-based view follows in D1."],
    ["SQ: Which ML methods are most suitable?", status("Done"), "C1–C6: logistic regression in a statistically tied top group; see §5."],
    ["SQ: How can XAI make predictions understandable?", status("Not yet"), "Study D (SHAP, LIME, counterfactuals, interpretable models, user study)."],
    ["Cloud misconfiguration findings, not only CVEs", status("Partly"),
     "Catalogs (A2), severity disagreement (A3), simulated estate (A5) and an exploratory model on it (C4, reported separately)."],
], [5.6 * cm, 1.8 * cm, 9.6 * cm])]

# ---------------------------------------------------------------- 3. Study A
story += [P("3. Study A — Data foundation", "h1"),
          P("No Deutsche Bank internal data is available, so the project uses three layers. Real and simulated "
            "results are always reported separately, and labels for prediction come only from CISA KEV.")]
story += [table([
    ["Layer", "Content", "Size"],
    ["L1 (real)", "NVD vulnerability records + EPSS exploit scores + CISA KEV (known exploited)", "430,678 CVEs; 1,790 KEV (1 in 240)"],
    ["L2 (real)", "Cloud misconfiguration rule catalogs: Prowler, Checkov, ScoutSuite", "3,870 checks"],
    ["L3 (simulated)", "Parameterised financial-institution estate (assets, IAM, criticality, SLA, recurrence)", "39,989 findings"],
], [2.6 * cm, 9.4 * cm, 5.0 * cm])]
story += [P("Results", "h2")]
story += [table([
    ["Step", "Outcome"],
    ["A1 Ingestion", "NVD API 2.0 (legacy bulk feeds are retired), EPSS, KEV joined on CVE ID; refreshed to the November catalog."],
    ["A2 Misconfiguration catalogs", "3,870 checks in one schema (Checkov 1,684 · Prowler 1,608 · ScoutSuite 578)."],
    ["A3 Severity disagreement", "Prowler vs ScoutSuite κ = 0.179 [0.015, 0.330], slight agreement, stable across matching thresholds."],
    ["A4 Remediation families", "138 ± 3 families from check descriptions; 106 span several tools. Membership moderately stable (ARI 0.53)."],
    ["A5 Estate simulator", "Built; priority-label formula amended to remove a scale mismatch (below)."],
    ["A6 EDA and protocol lock", "EPSS leakage quantified and fixed; evaluation protocol and hypotheses pre-registered."],
], [4.2 * cm, 12.8 * cm])]
story += [P("Key findings", "h2")]
story += [figure(os.path.join(FIG, "a3_severity_crosstab.png"), 9.0,
                 "Figure 1. Prowler and ScoutSuite rate the same checks differently, and Checkov's open-source release has "
                 "no severities. Tool severity is therefore not a trustworthy label, which justifies using KEV.")]
story += bullets([
    "<b>Tool severity is not ground truth.</b> Disagreement is systematic: all 13 matched CloudWatch alarm checks "
    "are <i>medium</i> in Prowler and <i>danger</i> in ScoutSuite.",
    "<b>Today's EPSS leaks the answer.</b> CVEs that entered KEV after the split already carry a present-day EPSS "
    "49.5× that of never-exploited CVEs; archived EPSS as it stood at the time gives 5.7×, which is genuine signal. "
    "Present-day EPSS is banned as a feature (Figure 2).",
    "<b>Simulated label amendment (for discussion).</b> The original L3 label put CVEs and misconfigurations on "
    "incompatible scales (99.4% of vulnerabilities <i>Low</i>; finding type alone predicted the label, AMI 0.289). "
    "The amended formula reduces this to AMI 0.095. It is documented in the deviations log.",
    "<b>Silent data bugs found and fixed</b>, e.g. pandas date inference erasing 91% of publication dates on re-read.",
])
story += [figure(os.path.join(FIG, "a6_epss_leakage.png"), 13.5,
                 "Figure 2. Present-day EPSS (orange) has already seen exploitation activity; EPSS as of the split date "
                 "(blue) has not. All models use point-in-time EPSS only.")]

# ---------------------------------------------------------------- 4. Study B
story += [P("4. Study B — Exploitation prediction and baselines", "h1"),
          P("<b>Setup (pre-registered, used unchanged in Study C).</b> Train on CVEs published before 1 January 2023 "
            "and test on later ones; data are never shuffled. Each CVE is scored with what an analyst could know when "
            "first seeing it: CVSS, description text and point-in-time EPSS. Training labels are KEV <i>as of the "
            "split date</i>. Test CVEs already in KEV when first triaged are excluded (443 of 752), leaving "
            "<b>309 exploited CVEs among 226,463</b>.")]
story += [P("4.1 Baselines (B1)", "h2"),
          figure(os.path.join(FIG, "b1_baselines.png"), 15,
                 "Figure 3. Fixed orderings on the full test set. Present-day EPSS (grey) is a leaky reference. CVSS v3 is "
                 "the bar to beat: 96 of its top 100 are not exploited, yet its ROC-AUC is 0.786 — why PR-AUC and "
                 "Precision@k are primary and ROC-AUC secondary.")]
story += [KeepTogether([P("4.2 KEV prediction (B2) — hypothesis H1", "h2"),
          figure(os.path.join(SCRATCH, "b2_chart.png"), 14.5,
                 "Figure 4. Both learned models beat CVSS v3; logistic regression is best. Present-day EPSS (PR-AUC 0.351) "
                 "is omitted as a leaky reference.")])]
story += bullets([
    "<b>H1 supported against CVSS:</b> gradient boosting vs CVSS v3, PR-AUC +0.0095 [0.0033, 0.0227], "
    "Precision@100 +0.08 [0.02, 0.15]; better on all 10 seeds.",
    "<b>Not better than point-in-time EPSS</b> (−0.0017 [−0.0116, 0.0043]); EPSS is itself a model trained on "
    "exploit-activity data this project does not have.",
    "<b>Logistic regression beat gradient boosting</b> — the starting point for Study C.",
    "<b>B3:</b> public NVD fields reproduce EPSS only weakly (Spearman ρ = 0.36). <b>B4:</b> month by month, "
    "scorers differ significantly (Friedman p ≤ 3×10<super>−4</super>); boosting beats CVSS by Wilcoxon "
    "(p 0.003–0.013), not by the more conservative Nemenyi test.",
    "<b>B5 (H5):</b> the margin over CVSS shrinks with time (0.034 → 0.022 → 0.016 for gaps of 1–3 years) in every "
    "training year, but with 9 cells it is not significant (p = 0.14): consistent decay, underpowered.",
])
story += [P("4.3 Cost-sensitive evaluation and confusion matrices (B6)", "h2"),
          P("No bank cost data exists, so the cost of a missed exploited CVE relative to an unnecessary fix is swept over "
            "10–500. Under a fixed monthly remediation budget the model finds about twice the exploited CVEs that CVSS "
            "ordering finds for the same effort (309 exploited CVEs; mean of 10 seeds):")]
story += [table([
    ["Policy", "True pos.", "False pos.", "False neg.", "Precision", "Recall", "F1"],
    ["Model: top 10 per month", "24.2", "445.8", "284.8", "5.2%", "7.8%", "0.062"],
    ["CVSS v3: top 10 per month", "11.2", "458.8", "297.8", "2.4%", "3.6%", "0.029"],
    ["Model: top 50 per month", "53.7", "2,296.3", "255.3", "2.3%", "17.4%", "0.040"],
    ["CVSS v3: top 50 per month", "33.3", "2,316.7", "275.7", "1.4%", "10.8%", "0.025"],
    ["Rule: CVSS ≥ 9", "133", "21,254", "176", "0.6%", "43.0%", "0.012"],
    ["Rule: CVSS ≥ 7", "263", "101,555", "46", "0.3%", "85.1%", "0.005"],
], [5.0 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm],
    extra=[("ALIGN", (1, 1), (-1, -1), "RIGHT")])]
story += [P("The common 'CVSS ≥ 7' rule catches 85% of exploited CVEs but requires fixing over 100,000. The threshold "
            "chosen on a validation year was unstable because 2022 contains only 39 exploited CVEs — the same problem "
            "that shaped Study C's selection protocol.", "cap")]

# ---------------------------------------------------------------- 5. Study C
story += [PageBreak(), P("5. Study C — Model comparison", "h1"),
          P("<b>Design.</b> Every model uses Study B's data, features, labels and test set, so all numbers are directly "
            "comparable. Hyperparameters are chosen on a time-ordered validation fold, never on test; each selected "
            "model is refitted with 10 seeds and evaluated once. The C1 design and, later, the C2–C6 design were "
            "pre-registered before the corresponding models ran.")]
story += [P("5.1 Classical models (C1) and a lesson about model selection", "h2"),
          P("C1 tuned six families (logistic regression, decision tree, random forest, XGBoost, LightGBM, CatBoost) "
            "on a 2022 validation year with only <b>39 exploited CVEs</b>. Validation rank did not predict test rank: "
            "LightGBM was best on validation and weakest of the boosted models on test. C1's registered verdict "
            "stands (H2 not supported). Because the cause was a data property — too few validation positives — a "
            "revised protocol (v2) with a wider validation window and <b>145 positives</b> was pre-registered "
            "before re-running (C1-R) and made binding for C2–C6. A diagnostic that scores all 126 tried "
            "configurations on test shows v2 selects far more reliably:")]
story += [table([
    ["Spearman ρ, validation vs test PR-AUC", "C1 fold (39 KEV)", "v2 fold (145 KEV)"],
    ["All 126 configurations pooled", "0.68", "<b>0.78</b>"],
    ["LightGBM (the C1 failure)", "0.16", "<b>0.68</b>"],
    ["XGBoost", "0.40", "<b>0.77</b>"],
    ["CatBoost / random forest / decision tree", "0.74 / 0.82 / 0.52", "0.57 / 0.65 / 0.58"],
], [8.0 * cm, 4.5 * cm, 4.5 * cm], extra=[("ALIGN", (1, 1), (-1, -1), "CENTER")])]
story += [Spacer(1, 5), P("Under v2, XGBoost is the best-validation boosted model and logistic regression the best overall. "
            "<b>H2 is still not supported</b>: XGBoost − LR = −0.0013 [−0.0070, 0.0031]. Logistic regression beats "
            "point-in-time EPSS on PR-AUC (+0.0060 [0.0013, 0.0121]) but not on Precision@100 (+0.01 [−0.03, 0.05]); "
            "the registered test needs both, so this is a partial result, and post-hoc with respect to C1.", "body")]
story += [figure(os.path.join(SCRATCH, "c_overview.png"), 15.5,
                 "Figure 5. All Study C models and both baselines on the same test set. The top of the chart is a tight "
                 "group; logistic regression leads, CVSS v3 (current practice) is last. Error bars are seed-to-seed "
                 "spread (logistic regression and the baselines are deterministic).")]
story += [KeepTogether([P("5.2 Learning-to-rank (C2) — hypothesis H3", "h2"),
          P("Six boosted models that differ <i>only in the training objective</i>: pointwise (binary classification), "
            "pairwise (the RankNet loss) and listwise (two LambdaMART implementations, XE-NDCG). Each month's new CVEs "
            "form one ranking query. <b>H3 not supported:</b> the selected LambdaMART beats the selected pointwise "
            "model by +0.0049 monthly NDCG@100, 95% CI [−0.0061, 0.0158], Wilcoxon p = 0.68. Ranking objectives look "
            "slightly better on top-of-list metrics (pairwise has the best NDCG@100, 0.151) but no difference is "
            "significant; XE-NDCG clearly fails. With about seven exploited CVEs per monthly queue, there is little "
            "list structure for a ranking loss to exploit.")])]
story += [P("5.3 Deep tabular models (C3) — hypothesis H4", "h2"),
          P("MLP, TabNet and FT-Transformer, tuned and selected the same way. <b>H4 (deep models do not beat boosted "
            "trees) is supported</b>: FT-Transformer − XGBoost = +0.0003 [−0.0043, 0.0087]. This is "
            "'no difference detected' rather than 'boosting wins'. FT-Transformer is also the least stable across "
            "seeds and needs about 8 minutes per fit against about 5 seconds for XGBoost; MLP and TabNet are below "
            "the classical models (MLP significantly below logistic regression). At this data scale complexity is "
            "not justified.")]
story += [P("5.4 Feature ablation and class imbalance (C4)", "h2"),
          figure(os.path.join(SCRATCH, "c4_chart.png"), 15,
                 "Figure 6. Feature groups used alone and in pairs (models not re-tuned per subset). Without point-in-time "
                 "EPSS, every model is at the CVSS-ordering level.")]
story += bullets([
    "<b>EPSS carries the signal.</b> CVSS + EPSS is statistically indistinguishable from all features "
    "(XGBoost −0.0023 [−0.0068, 0.0012]). Description text adds little once CVSS and EPSS are present.",
    "<b>Imbalance handling does not help.</b> Class weighting and undersampling never change PR-AUC or "
    "Precision@100 significantly, but they wreck the probabilities (Brier 0.0014 → 0.064). Train unweighted.",
    "<b>Simulated estate (L3, reported separately).</b> Predicting High/Critical priority: finding-only features "
    "PR-AUC 0.39, asset-context-only 0.39, both together 0.998. The model recovers the documented label formula "
    "once it sees both halves, so L3 accuracy says nothing about a real estate. What it does show is that the "
    "simulated priority needs both finding and asset context.",
])
story += [KeepTogether([P("5.5 Calibration (C5)", "h2"),
          P("Can the model's numbers be read as probabilities? Platt scaling and isotonic regression were fitted on "
            "the validation fold and applied to test. At a 0.14% base rate the locked equal-width ECE puts almost "
            "everything in one bin, so an adaptive (equal-mass) ECE was added and labelled as such."), table([
    ["Model", "Brier raw", "Brier Platt", "Adaptive ECE raw", "Adaptive ECE Platt"],
    ["Logistic regression", "0.001349", "0.001349", "<b>0.00028</b>", "0.00065"],
    ["XGBoost", "0.001350", "0.001350", "0.00079", "0.00060"],
    ["Random forest", "0.001374", "<b>0.001360</b>", "0.00234", "<b>0.00035</b>"],
    ["FT-Transformer", "0.001358", "0.001351", "0.00096", "0.00074"],
    ["LambdaMART (scores, not probabilities)", "—", "0.001351", "—", "0.00055"],
    ["XGBoost, full class weighting (C4)", "0.063716", "<b>0.001350</b>", "0.20128", "<b>0.00056</b>"],
], [6.0 * cm, 2.6 * cm, 2.6 * cm, 2.9 * cm, 2.9 * cm], extra=[("ALIGN", (1, 1), (-1, -1), "RIGHT")]),
          P("Unweighted models are already close to calibrated (a constant base-rate prediction scores Brier 0.00136). "
            "Platt scaling significantly improves the models whose raw scores are off (decision tree, random forest, "
            "CatBoost, TabNet), fully repairs class-weighted models and turns ranking scores into probabilities. "
            "Logistic regression needs no recalibration. Isotonic regression lowers PR-AUC (it creates ties), so "
            "Platt is the recommended calibrator.", "cap")])]
story += [P("5.6 Statistical comparison across all methods (C6)", "h2"),
          P("All 15 models plus CVSS and point-in-time EPSS were ranked within each of the 46 monthly queues. The "
            "methods differ (Friedman p &lt; 10<super>−4</super> on MAP, Precision@10, NDCG@10 and NDCG@100). "
            "Figure 7 is the standard critical-difference diagram: methods joined by a bar are not significantly "
            "different (Nemenyi, α = 0.05).")]
story += [figure(os.path.join(FIG, "c6_cd_map.png"), 15.5,
                 "Figure 7. Mean rank per month on MAP (1 = best). Logistic regression has the best mean rank; the boosted "
                 "and learning-to-rank models sit in the same group. CVSS v3, the decision tree and XE-NDCG are at the "
                 "bottom. Nemenyi is conservative with 17 methods; Holm-corrected Wilcoxon tests are reported alongside.")]
story += bullets([
    "<b>The top tier is flat.</b> After Holm correction, logistic regression is indistinguishable from the eight "
    "other boosted models, the MLP and point-in-time EPSS; it is significantly better than the decision tree, "
    "random forest, XE-NDCG, FT-Transformer, TabNet and CVSS v3.",
    "<b>No model beats point-in-time EPSS month by month</b>, on any of the five metrics, once 16 comparisons are "
    "corrected. The overall PR-AUC advantage of logistic regression over EPSS (§5.1) does not survive as a "
    "per-month advantage.",
    "<b>CVSS v3 is in the bottom group on every metric</b>, so the Study B conclusion holds across all models.",
])

# ---------------------------------------------------------------- 6. hypotheses
story += [P("6. Pre-registered hypotheses — scorecard", "h1")]
story += [table([
    ["#", "Hypothesis", "Verdict", "Evidence"],
    ["H1", "A learned model beats CVSS ordering (PR-AUC and P@100)", verdict("Supported"), "B2; confirmed for every Study C model (C6)"],
    ["H1b", "The best model also beats point-in-time EPSS (PR-AUC and P@100)", verdict("Partly"), "PR-AUC yes (C1-R, post-hoc); P@100 and monthly comparisons no"],
    ["H2", "Gradient boosting beats LR and single trees", verdict("Not supported"), "LR ≥ boosting (C1, C1-R); boosting > single tree"],
    ["H3", "LambdaMART beats pointwise classification on NDCG@100", verdict("Not supported"), "+0.0049 [−0.0061, 0.0158] (C2)"],
    ["H4", "Deep tabular models do not beat gradient boosting", verdict("Supported"), "No difference detected (C3); not 'boosting wins'"],
    ["H5", "Performance decays with distance from the training year", verdict("Partly"), "Consistent direction, p = 0.14 (B5)"],
    ["H6", "SHAP and LIME disagree on per-finding attributions", verdict("Pending"), "Study D"],
], [1.1 * cm, 6.6 * cm, 2.6 * cm, 6.7 * cm])]
story += [P("Four of the six decided hypotheses failed or held only in part. They are reported as found, not reframed: "
            "the negative results (no gain from boosting, ranking objectives or deep learning) are themselves the "
            "practical answer to 'which method is most suitable'.", "cap")]

# ---------------------------------------------------------------- 7. rigour
story += [P("7. Rigour, threats to validity and deviations", "h1")]
story += bullets([
    "<b>Pre-registration:</b> the protocol and H1–H6 were locked before any model; Study B and two Study C addenda "
    "were registered before their models ran. The one revision made after seeing results (selection protocol v2) "
    "is labelled post-hoc for C1 and was registered before it was used.",
    "<b>Reproducibility:</b> version-controlled code, one executed notebook per experiment, every run in MLflow; "
    "re-running C5 reproduced every number exactly.",
    "<b>Bugs caught before reporting:</b> a statistics library (scikit-posthocs) silently overwrote its p-value "
    "table when drawing diagrams; torch crashed when loaded after XGBoost; an MLflow naming error. Each was "
    "diagnosed, fixed and the affected step re-run; all are in the deviations log.",
    "<b>KEV is a lagging, partial ground truth:</b> absence from KEV is not evidence that a CVE was never exploited.",
    "<b>Few positives:</b> 309 exploited CVEs in test, about seven per month, which limits power (B5, C2, C6).",
    "<b>No real enterprise data:</b> no result measures Deutsche Bank's estate; simulated (L3) results are reported "
    "separately and never as ground truth. The simulator's label noise is set to zero, so its registered "
    "noise-sensitivity check was uninformative.",
    "<b>Other logged deviations:</b> NVD bulk feeds retired (API used); Checkov has no open-source severities; L3 "
    "label amendment; present-day EPSS replaced by archived EPSS; B5 training years moved to 2021–2024 (KEV "
    "launched November 2021); smaller search budgets for the deep models, fixed in advance from a speed test.",
])

# ---------------------------------------------------------------- 8. next steps
story += [KeepTogether([P("8. Next steps and points for discussion", "h1"),
          P("Study D (explainability) is next, to be pre-registered before it runs. Prioritised against the "
            "1 February 2027 deadline:")])]
story += bullets([
    "<b>Primary model for explanation:</b> logistic regression (top of the tied group, transparent by design), with "
    "XGBoost alongside for TreeSHAP, so explanations of an interpretable and a black-box model can be compared.",
    "<b>D1–D5:</b> global SHAP, permutation importance and ALE; per-finding SHAP and LIME; DiCE counterfactuals; "
    "interpretable models (EBM, sparse trees); stability and SHAP–LIME agreement (H6).",
    "<b>D6 human evaluation with CSO engineers:</b> proposed as a small pilot; scope and timing to be agreed now.",
    "<b>Write-up</b> of Studies A–C in thesis form in parallel.",
])
story += [Spacer(1, 4), callout(
    "<b>For discussion:</b> (1) framing the main result as 'simple, explainable models are enough; ML beats CVSS, "
    "and matches but does not clearly beat EPSS'; (2) the amended simulated-label formula; (3) logistic regression "
    "as the primary model for Study D; (4) feasibility and timing of a pilot user study within Deutsche Bank CSO.")]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "Progress Report — Studies A, B and C · Popa Stefan Andrei")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.9 * cm, title="Progress Report — Studies A, B and C",
                        author="Popa Stefan Andrei", subject="Cloud Security Finding Prioritisation — ML & XAI")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("written", OUT)
