"""
Extract the author-supplied replacement text for the paper, verbatim, into
scripts/paper_supplied_text.json, which scripts/build_paper.py reads.

    python3 scripts/extract_supplied_text.py

Three source files, edited by the authors in Word (all in ~/Downloads unless noted):

  slides_4_5.docx                    Section IV-C, IV-D, V (A-D), first paragraph of VI-A.
                                     Read paragraph by paragraph with python-docx: exact.
  osdarsh.pdf                        Abstract, index terms, introduction, contributions,
                                     Section II-A. A PDF export, so lines are re-joined;
                                     a line that ends in "-" keeps its hyphen (the text is
                                     not auto-hyphenated, so a hyphen at a line end is one).
  IUDBMS_Project_Report_IEEE (1).pdf the edited pages 6-8: rest of VI-A, VI-B to VI-F,
                                     VII. Same line-joining rule.

Nothing is corrected, normalised or reworded: a typo in a source is a typo in the output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import docx
import pymupdf

HOME = Path.home() / "Downloads"
OUT = Path(__file__).resolve().parent / "paper_supplied_text.json"


def join_lines(lines: list[str]) -> str:
    out = ""
    for ln in (x.rstrip() for x in lines):
        if not ln.strip():
            continue
        if not out:
            out = ln
        elif out.endswith("-"):
            out += ln.lstrip()
        else:
            out += " " + ln.lstrip()
    return out


def block_text(txt: str) -> str:
    return join_lines(txt.split("\n"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def page_lines(pg):
    rows = []
    for b in pg.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            t = "".join(s["text"] for s in l["spans"])
            if t.strip():
                x0, y0, x1, y1 = l["bbox"]
                rows.append((y0, x0, t))
    rows.sort()
    return rows


def main() -> int:
    S: dict[str, str] = {}
    src = {}

    # ---------------------------------------------------------------- slides_4_5.docx
    p = HOME / "slides_4_5.docx"
    src[p.name] = sha(p)
    paras = [x.text for x in docx.Document(p).paragraphs]
    want = {  # key -> distinctive start of the paragraph
        "ivc_1": "A device alarms once 4 of its last 6", "ivc_2": "Two checks determine what gets judged",
        "ivd_1": "When a device is escalated",
        "va_intro": "No labelled USB attack traffic",
        "va_b1": "Typing speed:", "va_b2": "Key releases:", "va_b3": "Text:", "va_b4": "Idle periods:",
        "va_b5": "Session length and attack onset:",
        "va_three": "Three choices keep the attacks", "va_tableref": "Table IV compares the simulator",
        "va_illustration": "For illustration, two seconds",
        "vb_1": "Corpus A is the original corpus", "vc_1": "To test on real kernels",
        "vc_2": "Each distribution runs one 75-minute", "vd_1": "Rates carry 95% Wilson",
        "via_1": "On Corpus A, the full system detected",
    }
    for k, start in want.items():
        hits = [t for t in paras if t.startswith(start)]
        assert hits, f"{k}: no paragraph starts with {start!r}"
        assert len(set(hits)) == 1, f"{k}: {len(hits)} different paragraphs start with {start!r}"
        S[k] = hits[0]

    # ---------------------------------------------------------------- osdarsh.pdf
    p = HOME / "osdarsh.pdf"
    src[p.name] = sha(p)
    d = pymupdf.open(p)
    p1 = block_text(d[0].get_text("blocks")[0][4])
    p2 = block_text(d[1].get_text("blocks")[0][4])
    p3 = block_text(d[2].get_text("blocks")[0][4])
    i_abs = p1.index("Abstract—") + len("Abstract—")
    i_idx = p1.index("Index Terms—")
    i_usb = p1.index("USB request block (URB) through usbmon")
    i_c1 = p1.index("1) A working pipeline")
    S["abstract"] = p1[i_abs:i_idx].strip()
    S["index_terms"] = p1[i_idx + len("Index Terms—"):i_usb].strip()
    intro_tail = p1[i_usb:i_c1].strip()
    contrib = p1[i_c1:] + " " + p2[:p2.index("I. INTRODUCTION")]
    import re
    cs = re.split(r"(?:(?<=\s)|^)([1-5])\) ", contrib.strip())
    # cs = ['', '1', text, '2', text, ...]
    for n in range(1, 6):
        S[f"contrib_{n}"] = cs[cs.index(str(n)) + 1].strip()
    intro = p2[p2.index("I. INTRODUCTION") + len("I. INTRODUCTION"):p2.index("II. RELATED WORK")].strip()
    j2 = intro.index("Defences in use today")
    j3 = intro.index("IUDBMS is our attempt")
    S["intro_1"] = intro[:j2].strip()
    S["intro_2"] = intro[j2:j3].strip()
    S["intro_3"] = intro[j3:].strip() + " " + intro_tail
    rel = p2[p2.index("A. USB Peripheral Security") + len("A. USB Peripheral Security"):].strip() + " " + p3.strip()
    k2 = rel.index("Closer to our setting")
    S["related_a1"] = rel[:k2].strip()
    S["related_a2"] = rel[k2:].strip()

    # ---------------------------------------------------------------- the edited paper, pages 6-8
    p = HOME / "IUDBMS_Project_Report_IEEE (1).pdf"
    src[p.name] = sha(p)
    d = pymupdf.open(p)

    def blk(pi, ytop, col):
        for b in d[pi].get_text("blocks"):
            if abs(b[1] - ytop) < 1.5 and ((b[0] < 300) == (col == "L")):
                return block_text(b[4])
        raise KeyError((pi, ytop, col))

    def line(pi, start):
        hits = [t for y, x, t in page_lines(d[pi]) if t.strip().startswith(start)]
        assert len(hits) == 1, (start, hits)
        return hits[0].rstrip()

    S["via_2"] = blk(1, 54, "L")
    S["vib_1"] = blk(1, 184, "L")
    S["vib_2"] = join_lines([line(1, "We therefore choose the threshold on gate firings"), blk(1, 513, "L")])
    S["vic_1"] = blk(1, 632, "L")
    S["vic_2"] = blk(1, 165, "R")
    S["vic_3"] = join_lines([blk(1, 323, "R"), line(1, "covert exfiltration go from"),
                             line(1, "positive, because they inject"), line(1, "paced families stay where")])
    S["vid_1"] = blk(1, 531, "R")
    S["vid_t1"] = blk(2, 279, "L")
    S["vid_2"] = blk(2, 459, "L")
    S["vie_1"] = blk(2, 578, "L")
    S["vie_2"] = join_lines([blk(2, 693, "L"), blk(2, 54, "R")])
    S["vie_3"] = blk(2, 82, "R")
    S["vif_1"] = blk(2, 223, "R")
    S["vif_2"] = blk(2, 443, "R")
    S["vii_1"] = blk(2, 658, "R")
    S["vii_2"] = blk(3, 279, "L")
    # identical to the paper as built; kept for the verification test only
    S["verify_limitations"] = blk(3, 427, "L") + " " + blk(3, 54, "R")
    S["verify_conclusion"] = blk(3, 127, "R")

    OUT.write_text(json.dumps({"_sources": src, **S}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}  ({len(S)} paragraphs)")
    for k, v in S.items():
        print(f"{k:16s} {len(v):5d}  {v[:70]!r} ... {v[-40:]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
