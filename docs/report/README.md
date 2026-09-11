# Report documents

Three self-contained HTML files. No server, no build step, no account needed —
open them in any browser, email them, or attach them to a submission.

| File | What it is |
|---|---|
| `iudbms_handbook.html` | **Start here.** Complete reference: problem, literature, the four novelty claims from `IUDBMS_Project_Report.pdf` checked against our own ablations, architecture as designed vs. built, data, algorithm, results, full audit trail |
| `iudbms_review.html` | Progress report: what is built, measured results, the findings |
| `iudbms_algorithms.html` | Technical reference: all 23 algorithms, formulas, parameters, rationale |

## Diagrams

| File | What it shows |
|---|---|
| `architecture.png` / `.pdf` | **Review deliverable.** The full pipeline, kernel hook to enforcement — every arrow labelled with what moves along it |
| `gate_timeline.png` / `.pdf` | The k-of-n gate escalating on a real test session, not an illustration |

Regenerate both with `make diagrams` (needs `make eval` first). PDF is vector —
use it for print and for slides; the PNG is 220 dpi.

## Sharing them

Open the file and use the browser's **Print → Save as PDF** for a submission or
an appendix. Both carry print stylesheets, so the PDF drops the navigation,
starts each layer on a fresh page and avoids splitting tables and formula blocks
across page breaks.

Fonts load from Google Fonts when online and fall back to system serif/sans/mono
offline, so the files stay readable with no network.

## Keeping them current

`iudbms_handbook.html` is **hand-written**, cross-referenced against
`../IUDBMS_Project_Report.pdf` (the Review 1 submission) and the numbers in
`out/metrics.json` / `out/model_zoo.json` at the time it was written. Re-check
it manually after any change that moves the headline numbers or the N1/N2/N3/N4
scorecard in Section 3 — that scorecard is the document's most load-bearing
claim and the one most likely to go stale silently.

`iudbms_review.html` is **generated**, not hand-edited:

    make eval      # produces out/metrics.json and out/figure_data.json
    make report    # regenerates iudbms_review.html from them

Every figure and every chart in it comes from those two files, so it cannot
drift from the results. Edit `eval/build_report.py`, never the HTML.

`iudbms_algorithms.html` is still a hand-written reference — it documents method
rather than measurements, so it only needs revisiting when an algorithm changes.

The two diagrams are drawn by `eval/render_architecture.py` and never
hand-edited. Both read their constants from `iudbms.contract` and
`out/figure_data.json`, and `gate_timeline` is a real test session stepped
through `PolicyEngine` — so changing the threshold, `GATE_K`/`GATE_N`,
`SEQ_LEN` or the corpus redraws them instead of leaving them quietly stale.

## About the charts

They are inline SVG rather than embedded PNGs, drawn by `eval/render_svg.py`
from `out/figure_data.json`. That keeps them sharp on a projector, responsive to
the reader's light/dark theme, and correct in print. Every value is direct-
labelled or repeated in the adjacent table, so nothing depends on hover.
