"""
Check that the built paper has every block in the right place: each paragraph under its
heading, each table and the figure immediately after the paragraph that introduces it, and
the tables numbered in order.

    python3 scripts/check_paper_layout.py [paper.docx]

The expected order is the order of the authors' own pages (the edited PDF they supplied),
written out below. Exit status 1 if anything is missing or out of order.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import supplied  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUPP = json.loads((ROOT / "scripts" / "paper_supplied_text.json").read_text())
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "IUDBMS_Project_Report_IEEE.docx"


def K(key):                      # a supplied paragraph, as it is built
    return ("key", key)


def H(text):                     # a heading
    return ("head", text)


def T(n):                        # table n: its caption, then the table itself
    return ("table", n)


def P(start):                    # a paragraph of mine, by its opening words
    return ("start", start)


FIG = ("fig", "Fig. 1.")

EXPECTED = [
    K("abstract"), K("index_terms"),
    H("I. Introduction"), K("intro_1"), K("intro_2"), K("intro_3"),
    K("contrib_1"), K("contrib_2"), K("contrib_3"), K("contrib_4"), K("contrib_5"),
    P("Section II reviews related work"),
    H("II. Related Work"), H("A. USB Peripheral Security"), K("related_a1"), K("related_a2"),
    H("B. eBPF for Kernel Observability"), P("eBPF runs verified programs"),
    H("C. Temporal Models for Behavioural Anomalies"), P("LSTM networks"),
    H("III. Threat Model, Scope and Terminology"), H("A. Threat Model and Scope"),
    P("The attacker supplies a USB device"), T(1),
    H("B. Terminology"), P("These terms are used with one meaning throughout"),
    P("–\tWindow."), P("–\tLearned model alone"),
    H("IV. System"), H("A. Capture and Features"), P("Fig. 1 shows the whole pipeline"), T(2),
    H("B. Two Stage Detector"), P("Stage 1 is unsupervised"), T(3), FIG,
    H("C. Decision Rule"), K("ivc_1"), K("ivc_2"),
    H("D. Enforcement and Deployment"), K("ivd_1"),
    H("V. Evaluation Method"), H("A. Simulated Traffic"), K("va_intro"),
    K("va_b1"), K("va_b2"), K("va_b3"), K("va_b4"), K("va_b5"),
    K("va_three"), K("va_tableref"), T(4), K("va_illustration"),
    H("B. Corpora and Splits"), K("vb_1"), T(5),
    H("C. Real Kernel Testbed"), K("vc_1"), K("vc_2"),
    H("D. Statistics, Pre registration and Availability"), K("vd_1"),
    H("VI. Results"), H("A. Detection in Simulation"), K("via_1"), T(6), K("via_2"),
    H("B. From a Window Threshold to False Alarms per Device Hour"), K("vib_1"), T(7), K("vib_2"),
    H("C. Generalisation to Unseen Attack Families"), K("vic_1"), T(8), K("vic_2"), K("vic_3"),
    H("D. Real Kernels"), K("vid_1"), T(9), K("vid_t1"), T(10), K("vid_2"),
    H("E. What Did Not Help"), K("vie_1"), K("vie_2"), K("vie_3"),
    H("F. Cost and Overhead"), K("vif_1"), K("vif_2"), T(11),
    H("VII. What the Evidence Supports"), K("vii_1"), T(12), K("vii_2"),
    H("VIII. Limitations and Threats to Validity"), P("1)\tSimulated training data"),
    P("7)\tPost hoc design"),
    H("IX. Conclusion"), P("IUDBMS treats USB trust as a question"),
    H("References"),
]


def main() -> int:
    d = docx.Document(DEST)
    seq = []                                   # (kind, text) in document order
    for ch in d.element.body.iterchildren():
        if ch.tag == qn("w:p"):
            t = "".join(("\t" if x.tag == qn("w:tab") else (x.text or ""))
                        for x in ch.iter()
                        if x.tag == qn("w:t") or (x.tag == qn("w:tab") and x.getparent().tag == qn("w:r")))
            has_pic = any(True for _ in ch.iter(qn("w:drawing")))
            if has_pic:
                seq.append(("pic", ""))
            if t.strip():
                seq.append(("p", t))
        elif ch.tag == qn("w:tbl"):
            seq.append(("tbl", ""))
    problems, pos = [], 0

    def find(pred, what):
        nonlocal pos
        for i in range(pos, len(seq)):
            if pred(seq[i]):
                pos = i + 1
                return i
        problems.append(f"not found in order (after position {pos}): {what}")
        return None

    for kind, val in EXPECTED:
        if kind == "key":
            want = supplied(val, SUPP[val])
            find(lambda e, w=want: e[0] == "p" and (e[1] == w or e[1].split("\t", 1)[-1] == w
                                                    or e[1].split("—", 1)[-1] == w), f"supplied {val}")
        elif kind == "head":
            find(lambda e, v=val: e[0] == "p" and e[1].strip() == v, f"heading {val!r}")
        elif kind == "start":
            find(lambda e, v=val: e[0] == "p" and e[1].startswith(v), f"paragraph {val!r}")
        elif kind == "table":
            i = find(lambda e, n=val: e[0] == "p" and re.match(rf"^Table {['','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII'][n]}\.\s", e[1]), f"caption of table {val}")
            if i is not None and not (i + 1 < len(seq) and seq[i + 1][0] == "tbl"):
                problems.append(f"table {val}: its caption is not immediately followed by the table")
            pos = (i or 0) + 2
        elif kind == "fig":
            i = find(lambda e: e[0] == "pic", "the architecture figure")
            if i is not None and not (i + 1 < len(seq) and seq[i + 1][1].startswith("Fig. 1.")):
                problems.append("the figure is not immediately followed by its caption")
            pos = (i or 0) + 2
    caps = [e[1][:12] for e in seq if e[0] == "p" and re.match(r"^Table [IVX]+\.\s", e[1])]
    if len(caps) != 12:
        problems.append(f"{len(caps)} table captions, expected 12")
    # every table and the figure is referred to in the text before it appears
    text_so_far, first_ref = "", {}
    for kind, t in seq:
        if kind == "p" and not re.match(r"^Table [IVX]+\.\s", t) and not t.startswith("Fig. 1."):
            for m in re.finditer(r"\bTable ([IVX]+)\b|\bFig\. 1\b", t):
                first_ref.setdefault(m.group(0), True)
        if kind == "p":
            m = re.match(r"^(Table [IVX]+)\.\s", t)
            if m and m.group(1) not in first_ref:
                problems.append(f"{m.group(1)} appears before any sentence refers to it")
            if t.startswith("Fig. 1.") and "Fig. 1" not in first_ref:
                problems.append("Fig. 1 appears before any sentence refers to it")
    for pr in problems:
        print("PROBLEM:", pr)
    print(f"layout: {len(EXPECTED)} anchors checked in order; {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
