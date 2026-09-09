# Report documents

Two self-contained HTML files. No server, no build step, no account needed —
open them in any browser, email them, or attach them to a submission.

| File | What it is |
|---|---|
| `iudbms_review.html` | Progress report: what is built, measured results, the eight findings |
| `iudbms_algorithms.html` | Technical reference: all 23 algorithms, formulas, parameters, rationale |

## Sharing them

Open the file and use the browser's **Print → Save as PDF** for a submission or
an appendix. Both carry print stylesheets, so the PDF drops the navigation,
starts each layer on a fresh page and avoids splitting tables and formula blocks
across page breaks.

Fonts load from Google Fonts when online and fall back to system serif/sans/mono
offline, so the files stay readable with no network.

## Keeping them current

They are exports, not sources. After changing any measured number, re-run

    make eval

and update the figures in the documents to match `out/metrics.json`. Nothing
regenerates these automatically — a stale number here is the same defect as a
stale number in a paper.
