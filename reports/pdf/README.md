# Progress-report PDF build

Regenerates `Progress_Report_Studies_A_B_C.pdf` (saved outside the repo, in the
parent "Internship Project" folder) from the experiment outputs.

```sh
# reportlab + pypdfium2 live in a separate venv (not in requirements.txt)
python3.12 -m venv /tmp/pdfenv && /tmp/pdfenv/bin/pip install reportlab matplotlib pypdfium2
FIGS=$(mktemp -d)
.venv/bin/python reports/pdf/b2_chart.py . $FIGS/b2_chart.png
.venv/bin/python reports/pdf/c_charts.py . $FIGS          # c_overview.png, c4_chart.png
/tmp/pdfenv/bin/python reports/pdf/build_pdf_abc.py . $FIGS "../Progress_Report_Studies_A_B_C.pdf"
```

All numbers in the PDF are typed from `outputs/` and the Deviations Log in
CLAUDE.md; `reports/interim_report_2.md` is the same content as Markdown.
