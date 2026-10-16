"""
Check that a built paper contains every author-supplied paragraph verbatim.

    python3 scripts/check_paper_text.py [paper.docx]

Reads scripts/paper_supplied_text.json (extracted from the authors' own files by
scripts/extract_supplied_text.py) and the built .docx, and reports any supplied paragraph
that is missing or differs by even one character. Exit status 1 if any does.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parent.parent
SUPP = json.loads((ROOT / "scripts" / "paper_supplied_text.json").read_text())
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "IUDBMS_Project_Report_IEEE.docx"

#: how each supplied paragraph appears in the document: the text that precedes it in the
#: same paragraph (a list label, or the run-in "Abstract"/"Index Terms" head)
PREFIX = {"abstract": "Abstract—", "index_terms": "Index Terms—",
          **{f"contrib_{n}": f"{n})\t" for n in range(1, 6)},
          **{f"va_b{n}": "–\t" for n in range(1, 6)}}


def main() -> int:
    paras = [p.text for p in docx.Document(DEST).paragraphs]
    present = set(paras)
    bad = 0
    n = 0
    for key, text in SUPP.items():
        if key.startswith("_") or key.startswith("verify_"):
            continue
        n += 1
        want = PREFIX.get(key, "") + text
        if want in present:
            continue
        bad += 1
        near = difflib.get_close_matches(want, paras, n=1, cutoff=0.5)
        print(f"MISSING {key}")
        if near:
            sm = difflib.SequenceMatcher(None, want, near[0])
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"    {tag}: supplied {want[i1:i2]!r}  vs  built {near[0][j1:j2]!r}")
    # paragraphs the authors did not change must still match what they saw
    def norm(t):
        return re.sub(r"\s+", " ", t.replace("–", "-").replace("—", "-")).strip()
    items = [p for p in paras if re.match(r"^[1-7]\)\t", p)]
    lim = norm(" ".join(p.replace("\t", " ", 1) for p in items[5:]))   # the five contributions come first
    ok_lim = norm(SUPP["verify_limitations"]) in lim or lim in norm(SUPP["verify_limitations"])
    concl = next((p for p in paras if p.startswith("IUDBMS treats USB trust")), "")
    ok_con = norm(concl) == norm(SUPP["verify_conclusion"])
    print(f"{n - bad} of {n} supplied paragraphs present verbatim; "
          f"limitations {'match' if ok_lim else 'DIFFER'}, conclusion {'matches' if ok_con else 'DIFFERS'} the edited pages")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
