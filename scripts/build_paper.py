"""
Build the IUDBMS paper as an IEEE conference-format .docx.

    python3 scripts/build_paper.py [out.docx]

**Every number in the Results section, and in the abstract, is read from the eval
artefacts** (``out/metrics.json``, ``out/fusion_ablation.json``, ``out/baseline_ablation.json``,
``out/cross_device.json``, ``out/real_*.json``, ``out/ebpf/*.json`` and the
``out/analysis_*.json`` files) rather than typed in. That is the same rule the rest
of the project runs on -- no number goes into a report that was not printed by
code -- and it means the paper cannot drift from the measurements. Re-run
``make eval`` and the ablations, then re-run this, and the tables follow.

IEEE Conference Template (Word), as implemented here:
  page   US Letter; margins top 0.75", bottom 1", left/right 0.625"
  cols   two, 3.5" each, 0.25" gutter; title block and wide figures full-width
  font   Times New Roman throughout, all text black
  title  24 pt; authors 11 pt; abstract / index terms 9 pt bold; body 10 pt
         justified with 0.2" first-line indent; section heads 10 pt small caps
         centred with Roman numerals; subsections 10 pt italic lettered;
         table captions 8 pt small caps above the table; figure captions 8 pt
         below; references 8 pt with a hanging indent.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "Downloads" / "IUDBMS_Project_Report_IEEE.docx"
FIG = OUT_DIR / "architecture_ieee.png"
#: Author-supplied replacement text, extracted verbatim from the authors' Word files by
#: scripts/extract_supplied_text.py. It is used as written: nothing in it is reworded,
#: re-punctuated or re-numbered (it already cites by FINAL reference number).
SUPP = json.loads((ROOT / "scripts" / "paper_supplied_text.json").read_text())

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)


def load(name: str) -> dict | None:
    p = OUT_DIR / name
    if not p.exists():
        print(f"  WARNING: {name} missing; its section will say so")
        return None
    return json.loads(p.read_text())


M = load("metrics.json")
FUSION = load("fusion_ablation.json")
BASE = load("baseline_ablation.json")
CROSS = load("cross_device.json")
REALCAP = load("real_capture_summary.json")
REALTYPE = load("real_typing.json")
EBPF_IDLE = load("ebpf/idle.json")
EBPF_BUSY = load("ebpf/busy.json")
LIVE_O = load("live_original.json")
LIVE_R = load("live_retrained.json")
DIAG = load("analysis_diagnostics.json")
RULEJ = load("analysis_rules.json")
BASEJ = load("analysis_baseline.json")

doc = docx.Document()

# ---------------------------------------------------------------- fonts
def force_fonts(rpr, name=None):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in list(rf.attrib):
        del rf.attrib[a]
    for a in ("ascii", "hAnsi", "cs", "eastAsia"):
        rf.set(qn(f"w:{a}"), name or FONT)


dd = doc.styles.element.find(qn("w:docDefaults"))
force_fonts(dd.find(qn("w:rPrDefault")).find(qn("w:rPr")))

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = FONT, Pt(10)
normal.font.color.rgb = BLACK
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.line_spacing = 1.0
force_fonts(normal.element.get_or_add_rPr())


def make_style(name, size, *, align=None, bold=False, italic=False, small_caps=False,
               before=0, after=0, first=None, left=None, spacing=1.0, outline=None,
               keep_next=False):
    st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = normal
    f = st.font
    f.name, f.size, f.bold, f.italic, f.small_caps = FONT, Pt(size), bold, italic, small_caps
    f.color.rgb = BLACK
    force_fonts(st.element.get_or_add_rPr())
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = spacing
    pf.keep_with_next = keep_next
    if align is not None:
        pf.alignment = align
    if first is not None:
        pf.first_line_indent = Inches(first)
    if left is not None:
        pf.left_indent = Inches(left)
    if outline is not None:
        ppr = st.element.get_or_add_pPr()
        ol = OxmlElement("w:outlineLvl")
        ol.set(qn("w:val"), str(outline))
        ppr.append(ol)
    return st


J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
make_style("IEEE Title", 24, align=C, after=10)
make_style("IEEE Author", 11, align=C, after=2)
make_style("IEEE Affiliation", 10, align=C, italic=True, after=10)
make_style("IEEE Abstract", 9, align=J, bold=True, after=6, spacing=0.95)
make_style("IEEE Body", 10, align=J, first=0.2, after=6, spacing=0.95)
make_style("IEEE List", 10, align=J, first=-0.25, left=0.25, after=3, spacing=0.95)
make_style("IEEE H1", 10, align=C, small_caps=True, before=12, after=6, keep_next=True, outline=0)
make_style("IEEE H2", 10, align=L, italic=True, before=6, after=3, keep_next=True, outline=1)
make_style("IEEE Caption", 8, align=C, after=10, spacing=0.95)
make_style("IEEE TableCaption", 8, align=C, small_caps=True, before=8, after=3,
           keep_next=True, spacing=0.95)
make_style("IEEE TableCell", 8, align=C, after=0, spacing=0.95)
make_style("IEEE Figure", 10, align=C, after=4, keep_next=True)
make_style("IEEE Ref", 8, align=J, first=-0.3, left=0.3, after=2, spacing=0.95)


def run(par, text, *, italic=None, bold=None, small_caps=None, size=None):
    # a hyphen before a digit that does not follow a word character is a minus sign
    text = re.sub(r"(?<![\w\-])-(?=\d)", "\u2212", text)
    r = par.add_run(text)
    force_fonts(r._r.get_or_add_rPr())
    r.font.color.rgb = BLACK
    if italic is not None:
        r.italic = italic
    if bold is not None:
        r.bold = bold
    if small_caps is not None:
        r.font.small_caps = small_caps
    if size is not None:
        r.font.size = Pt(size)
    return r


#: Source ids (the numbers used while writing) -> reference keys. The numbers actually
#: printed are assigned in order of FIRST CITATION, as IEEE requires. The text used to be
#: numbered by hand, and cited [5] before [3] and [4].
OLD2KEY = {1: "shen", 2: "bintang", 3: "kosto", 4: "rajurkar", 5: "song", 6: "baek",
           7: "hnamte", 8: "saini", 9: "raja", 10: "anwar", 11: "shannon", 12: "nohl",
           13: "goodusb", 14: "usbfilter", 15: "usblock", 16: "huang", 17: "iforest",
           18: "lstm", 19: "dhakal", 20: "killourhy", 21: "balabit", 22: "hidspec"}
#: The supplied text cites references by their FINAL numbers, which are the order of first
#: citation in the paper as it stood. Pin that numbering so text that cites by final number
#: and text that still cites by source id (number_cites) agree. The end of the file checks
#: that the numbers really are in order of first citation.
REF_ORDER = ("nohl", "shen", "goodusb", "usbfilter", "bintang", "usblock", "baek", "hnamte",
             "saini", "raja", "anwar", "huang", "song", "kosto", "rajurkar", "lstm", "iforest",
             "hidspec", "shannon", "dhakal", "killourhy", "balabit")
REF_NO: dict[str, int] = {k: i + 1 for i, k in enumerate(REF_ORDER)}
NO2KEY = {n: k for k, n in REF_NO.items()}
CITED: set[str] = set()
CITED_ORDER: list[str] = []


def _cite_no(old):
    key = OLD2KEY[old]
    if key not in REF_NO:
        REF_NO[key] = len(REF_NO) + 1
    if key not in CITED:
        CITED.add(key)
        CITED_ORDER.append(key)
    return REF_NO[key]


def mark_cited(text):
    """Record the citations in text that already carries final numbers."""
    for m in re.finditer(r"\[(\d+)\](?:–\[(\d+)\])?", text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        for n in range(a, b + 1):
            key = NO2KEY[n]                      # KeyError: the text cites a number with no reference
            if key not in CITED:
                CITED.add(key)
                CITED_ORDER.append(key)


def number_cites(text):
    """Rewrite [n] and [a]–[b] as final numbers; compress consecutive runs of three or more."""
    def sub(m):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        nums = [_cite_no(i) for i in range(a, b + 1)]
        if len(nums) >= 3 and nums == list(range(nums[0], nums[0] + len(nums))):
            return f"[{nums[0]}]–[{nums[-1]}]"
        return ", ".join(f"[{n}]" for n in nums)
    return re.sub(r"\[(\d+)\](?:–\[(\d+)\])?", sub, text)


def rich(par, text, cites=True, **base):
    italic = False
    code = False
    if cites:
        text = number_cites(text)
    for tok in re.split(r"(</?i>|</?code>)", text):
        if tok == "<i>":
            italic = True
        elif tok == "</i>":
            italic = False
        elif tok == "<code>":
            code = True
        elif tok == "</code>":
            code = False
        elif tok:
            r = run(par, tok, italic=True if italic else base.get("italic"),
                    bold=base.get("bold"))
            if code:
                force_fonts(r._r.get_or_add_rPr(), name="Courier New")
                r.font.size = Pt(9)


def para(style, text="", **kw):
    p = doc.add_paragraph(style=style)
    if text:
        rich(p, text, **kw)
    return p


def h1(roman, title):
    p = doc.add_paragraph(style="IEEE H1")
    run(p, f"{roman}. ")
    run(p, title, small_caps=True)
    return p


def h2(letter, title):
    p = doc.add_paragraph(style="IEEE H2")
    run(p, f"{letter}. {title}")
    return p


def body(text):
    return para("IEEE Body", text)


def body_lit(text):
    """A paragraph of supplied text, used exactly as written (see SUPP)."""
    mark_cited(text)
    p = doc.add_paragraph(style="IEEE Body")
    rich(p, text, cites=False)
    return p


def lead(text, lead_in):
    """Italicise the run-in heading a supplied paragraph starts with; the words are untouched."""
    assert text.startswith(lead_in), f"{text[:40]!r} does not start with {lead_in!r}"
    return f"<i>{lead_in}</i>" + text[len(lead_in):]


def item(label, text):
    p = doc.add_paragraph(style="IEEE List")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.25), WD_TAB_ALIGNMENT.LEFT)
    run(p, f"{label}\t")
    rich(p, text)
    return p


def item_lit(label, text):
    """A list item of supplied text, used exactly as written (see SUPP)."""
    mark_cited(text)
    p = doc.add_paragraph(style="IEEE List")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.25), WD_TAB_ALIGNMENT.LEFT)
    run(p, f"{label}\t")
    rich(p, text, cites=False)
    return p


_TABLE_N = [0]
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
#: Tables in the order they appear. Text refers to them by key through ``tref`` and
#: ``table`` asserts the order, so a "Table V" in the prose cannot point at the wrong table.
TABLE_ORDER = ["families", "headline", "perclass", "n1", "n2", "realism", "rules", "parser",
               "keyboard", "live", "unseen", "perturb", "ebpf"]


def tref(key):
    return f"Table {ROMAN[TABLE_ORDER.index(key) + 1]}"


def table(caption, header, rows, *, widths=None, key=None, left_cols=(0,)):
    """An IEEE table: caption above in small caps, horizontal rules only."""
    _TABLE_N[0] += 1
    if key is not None:
        assert TABLE_ORDER.index(key) + 1 == _TABLE_N[0], \
            f"table {key!r} is #{_TABLE_N[0]} but TABLE_ORDER puts it at #{TABLE_ORDER.index(key) + 1}"
    p = doc.add_paragraph(style="IEEE TableCaption")
    run(p, f"Table {ROMAN[_TABLE_N[0]]}.  ", small_caps=True)
    run(p, caption, small_caps=True)

    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for j, htxt in enumerate(header):
        cell = t.rows[0].cells[j]
        cp = cell.paragraphs[0]
        cp.style = doc.styles["IEEE TableCell"]
        run(cp, htxt, bold=True)
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            cell = t.rows[i].cells[j]
            cp = cell.paragraphs[0]
            cp.style = doc.styles["IEEE TableCell"]
            if j in left_cols:
                cp.alignment = L
            run(cp, str(v))
    # Keep the table on one page: every row but the last holds with the next, and no
    # row may split. IEEE tables are short; a table broken across pages reads badly.
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if ri < len(t.rows) - 1:
            for cell in row.cells:
                for cp in cell.paragraphs:
                    cp.paragraph_format.keep_with_next = True

    # rules: above header, below header, below last row
    for idx, row in enumerate(t.rows):
        for cell in row.cells:
            tcpr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "bottom"):
                show = (edge == "top" and idx in (0,)) or \
                       (edge == "bottom" and idx in (0, len(t.rows) - 1))
                e = OxmlElement(f"w:{edge}")
                e.set(qn("w:val"), "single" if show else "nil")
                if show:
                    e.set(qn("w:sz"), "6")
                    e.set(qn("w:color"), "000000")
                borders.append(e)
            tcpr.append(borders)
    doc.add_paragraph(style="IEEE Caption")
    return t


def set_columns(section, n):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")):
        sp.remove(c)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), "360")
    grid = sp.find(qn("w:docGrid"))
    (grid.addprevious if grid is not None else sp.append)(cols)


def new_section(n_cols):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec, n_cols)
    brk = doc.paragraphs[-1]
    brk.paragraph_format.space_after = Pt(0)
    brk.paragraph_format.line_spacing = Pt(1)
    ppr = brk._p.get_or_add_pPr()
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "2")
    rpr.append(sz)
    sect = ppr.find(qn("w:sectPr"))
    (sect.addprevious if sect is not None else ppr.append)(rpr)
    return sec


def pct(x, nd=1):
    return "—" if x is None else f"{100 * float(x):.{nd}f}%"


# ---------------------------------------------------------------- page setup
sec0 = doc.sections[0]
sec0.page_width, sec0.page_height = Inches(8.5), Inches(11)
sec0.left_margin = sec0.right_margin = Inches(0.625)
sec0.top_margin, sec0.bottom_margin = Inches(0.75), Inches(1.0)
sec0.header_distance = sec0.footer_distance = Inches(0.4)
set_columns(sec0, 1)

TITLE = ("IUDBMS: Behaviour-Based Runtime Trust for USB Peripherals, a Simulation-Based "
         "Proof of Concept Evaluated on Real Linux Kernels")
para("IEEE Title", TITLE)

AUTHORS = [("Darsh Sharma", "24BDS0082"),
           ("Devansh Bramha", "24BCE0646"),
           ("Siddhant Gaddi", "24BCT0273")]


def author_line(style, cells):
    p = doc.add_paragraph(style=style)
    p.alignment = L
    third = 7.25 / len(AUTHORS)
    for i in range(len(AUTHORS)):
        p.paragraph_format.tab_stops.add_tab_stop(Inches(third * (i + 0.5)),
                                                  WD_TAB_ALIGNMENT.CENTER)
    for c in cells:
        run(p, "\t")
        run(p, c)
    return p


author_line("IEEE Author", [n for n, _ in AUTHORS])
author_line("IEEE Affiliation", [f"Reg. No. {r}" for _, r in AUTHORS])

new_section(2)

# ---------------------------------------------------------------- numbers
CAL = load("analysis_calibration.json")
SFP = load("analysis_session_fp.json")
LONG = load("analysis_long_sessions.json")
ROBF = load("analysis_robust_features.json")
HYB = load("analysis_hybrid.json")
TR = load("analysis_transfer.json")
HT = load("analysis_human_timing.json")
POLL = load("analysis_polling.json")
LOPO = load("analysis_lopo.json")
RB = load("analysis_realistic_v2.json")
MC = load("model_card.json")
TB = load("testbed/report.json")
OVH = load("testbed/overhead.json")
TABLE_ORDER[:] = (["families", "features", "settings", "timing", "data", "simdetect",
                   "operating", "transfer", "envs", "realkernel"]
                  + (["overhead"] if OVH else []) + ["claims"])
DEP = json.loads((OUT_DIR / "models" / "deploy.json").read_text())

FAMS = ["hid_injection", "staged_delay", "identity_switch", "class_exfil", "mimicry",
        "mimicry_adaptive"]
FAST = ["hid_injection", "staged_delay", "identity_switch"]
before, after = CAL["fresh_before"], CAL["fresh_after"]
cb, ca = before["corpus_c"], after["corpus_c"]
kb_b, kb_a = before["keyboard"], after["keyboard"]
sfp_old, sfp_new = SFP["rows"][0], SFP["rows"][-1]
lopo_m, lopo_h = HYB["lopo_mean"]["model"], HYB["lopo_mean"]["hybrid"]
seen_mean = sum(LOPO[f]["full"][0] for f in FAMS) / len(FAMS)
tb_tot = TB["benign_total"]
tb_att = TB["attack_summary"]
n_att_tb = sum(v["n"] for v in tb_att.values())
fast_n = sum(tb_att[f]["n"] for f in FAST)
fast_dep = sum(tb_att[f]["deployed"] for f in FAST)
fast_rule = sum(tb_att[f]["rule"] for f in FAST)
fast_prev = sum(tb_att[f]["model_previous"] for f in FAST)
model_cal_all = sum(v["model_calibrated"] for v in tb_att.values())
mim_n = sum(tb_att[f]["n"] for f in ("mimicry", "mimicry_adaptive"))
mim_dep = sum(tb_att[f]["deployed"] for f in ("mimicry", "mimicry_adaptive"))
kbd_role = TB["benign_by_role"]["keyboard"]
tb_lat = sorted(a["latency_s"] for a in TB["attacks"] if a["latency_s"] is not None)
tb_lat_med = tb_lat[len(tb_lat) // 2]
dists = TB["distributions"]
tb_minutes = sum(d["minutes"] for d in dists.values())
tb_lines = sum(d["usbmon_lines"] for d in dists.values())
tb_unparsed = sum(d["unparsed"] for d in dists.values())
tb_keys = sum(d["keys_typed"] for d in dists.values())
long_kbd = LONG["benign"]["keyboard"]["alarms_per_device_hour"]
det_a = M["full_system"]["session"]["attack_detect"]
s1_a = M["stage1_only"]["session"]["attack_detect"]
orig_fp_b = RB["overall"]["original"]["fp"]
human = HT["aalto"]


def kn(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "—"



# ---------------------------------------------------------------- supplied text vs results
#: The supplied text states many numbers literally. Each one that a result file also holds is
#: checked here, so a rebuild after new results fails instead of printing a stale number.
NOTES: list[str] = []


def _need(cond, what):
    assert cond, f"supplied text says {what}, but the result files disagree"


_need(ca["detect_k"] == 448 and ca["n_attack"] == 600 and ca["fp_k"] == 0 and ca["n_benign"] == 3000
      and pct(ca["detect"]) == "74.7%", "448 of 600 attacks (74.7%), no false alarm in 3,000 (calibrated)")
_need(cb["detect_k"] == 496 and cb["fp_k"] == 2 and pct(cb["detect"]) == "82.7%",
      "496 of 600 (82.7%) with 2 false positives (per-window threshold)")
_need(BASEJ["headline"]["detect_k"] == 107 and BASEJ["headline"]["n_attack"] == 120
      and BASEJ["headline"]["n_benign"] == 178, "107 of 120 attacks, 178 benign test sessions (Corpus A)")
_need(pct(det_a) == "89.2%" and pct(s1_a) == "15.8%" and pct(orig_fp_b) == "46.6%",
      "89.2% / 15.8% / 46.6% (Corpus A, without the LSTM, original model on Corpus B)")
_need(pct(lopo_m) == "34.8%" and pct(lopo_h) == "68.2%", "34.8% -> 68.2% (unseen families)")
_need(int(tb_tot["devices"]) == 114 and f"{tb_tot['hours']:.1f}" == "41.4",
      "114 benign device sessions over 41.4 device-hours")
_need(fast_dep == 24 and fast_n == 24 and model_cal_all == 1 and n_att_tb == 40 and mim_n == 16
      and mim_dep == 0, "24/24 fast attacks by the deployed rule, 1/40 by the model alone, 0/16 human-paced")
_need(int(kbd_role["devices"]) == 12 and int(kbd_role["previous_threshold_alarmed"]) == 6
      and int(kbd_role["deployed_alarmed"]) == 0, "0 of 12 keyboards alarmed, 6 of 12 at the old threshold")
_need(int(tb_tot["judging_all_alarmed"]) == 27 and int(tb_tot["corpus_a_alarmed"]) == 82
      and int(tb_tot["previous_threshold_alarmed"]) == 41, "27 / 82 / 41 of 114 benign devices alarmed")
_need(f"{tb_lat_med:.1f}" == "2.6", "median latency 2.6 s")
_need(f"{kb_b['rate']:.3f}" == "0.225" and f"{kb_a['rate']:.3f}" == "0.013" and kb_b["alarms"] == 18
      and kb_a["alarms"] == 1 and 4 <= 1 / kb_b["rate"] <= 5, "keyboard alarms 0.225 -> 0.013 per hour, one per 4-5 h")
_need(f"{sfp_old['p_window_over']:.2e}" == "1.05e-03" and round(100 * sfp_old["q_next_over_given_over"]) == 38
      and f"{sfp_old['naive_alarms_per_hour']:.1e}" == "6.5e-08", "1.05e-03 / 38% / 6.5e-08")
if f"{sfp_old['observed_alarms_per_hour']:.3f}" != "0.255":
    NOTES.append("Section VI-B says the observed alarms per hour is 0.255; the result file and "
                 f"Table VII say {sfp_old['observed_alarms_per_hour']:.3f}.")
_need(f"{long_kbd:.3f}" == "0.149" and round(LONG["total_device_hours"]) == 101, "0.149/h over about 101 device-hours")
_need(CAL["calibration"]["hours"] == 80, "80 hours of calibration sessions")
_need(HT["aalto"]["sessions"] == 1972 and HT["aalto"]["reports_per_s_sustained_max"] == 26
      and DEP["capability_rule"]["human_max_rps"] == 30, "1,972 people, sustained maximum 26, gate at 30")
_need(f"{DIAG['cost']['ms_per_device_window']:.1f}" == "3.8" and DIAG["cost"]["devices_per_core"] == 264,
      "3.8 ms per device-window, 264 devices per core")
_need(EBPF_BUSY["ebpf"]["completions_at_giveback_hook"] == 1144
      and EBPF_BUSY["usbmon"]["completions_and_errors"] == 1144, "1,144 of 1,144 completions")
_need(tb_lines == 2901742 and tb_unparsed == 0, "2,901,742 lines, 0 unparsed")
_need(pct(POLL["polling_v3"]["corpus_c_v2"]["detect"]) == "71.5%"
      and pct(POLL["polling_v3"]["corpus_c_v2"]["per_family"]["mimicry"], 0) == "29%",
      "polling-limited retraining: 71.5%, replay loop 29%")
_need(f"{ROBF['summary']['R_both']['kbd_alarms_per_hour'][0]:.3f}" == "0.296"
      and f"{ROBF['summary']['base']['kbd_alarms_per_hour'][0]:.3f}" == "0.242", "0.296 against 0.242 per hour")
_need(f"{TR['overlap']['strong']['spearman_jaccard']:+.2f}" == "+0.61"
      and f"{TR['overlap']['top5']['spearman_jaccard']:+.2f}" == "-0.22", "Spearman +0.61 (top five: -0.22)")
_need(HYB["corpus_c"]["rule"]["detect_k"] == 400, "the capability gate alone reaches 400/600")
_need(pct(cb["per_profile"]["mimicry"], 0) == "95%" and pct(ca["per_profile"]["mimicry"], 0) == "48%",
      "replay loop 95% -> 48%")
_need(len(MC["features"]) == 30, "30 features")
_need([tref(k) for k in ("timing", "data", "simdetect", "operating", "transfer", "envs", "realkernel",
                         "overhead", "claims")]
      == ["Table IV", "Table V", "Table VI", "Table VII", "Table VIII", "Table IX", "Table X",
          "Table XI", "Table XII"], "Tables IV to XII in this order")

# ---------------------------------------------------------------- abstract
ABSTRACT = SUPP["abstract"]
mark_cited(ABSTRACT)
n_words = len(re.findall(r"\S+", ABSTRACT))
print(f"  abstract: {n_words} words" + ("   (IEEE allows at most 200)" if n_words > 200 else ""))

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Abstract", italic=True)
run(p, "—" + ABSTRACT)
p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Index Terms", italic=True)
run(p, "—" + SUPP["index_terms"])

# ---------------------------------------------------------------- I
h1("I", "Introduction")
body_lit(SUPP["intro_1"])
body_lit(SUPP["intro_2"])
body_lit(SUPP["intro_3"])
item_lit("1)", lead(SUPP["contrib_1"], "A working pipeline"))
item_lit("2)", lead(SUPP["contrib_2"], "Setting the alarm threshold"))
item_lit("3)", SUPP["contrib_3"])
item_lit("4)", SUPP["contrib_4"])
item_lit("5)", lead(SUPP["contrib_5"], "Negative results with their pre registered criteria"))
body("Section II reviews related work. Section III fixes the threat model and the terms used "
     "throughout. Section IV describes the system, Section V the evaluation method and "
     "Section VI the results. Section VII separates what is proposed, implemented and "
     "supported, Section VIII lists the limitations, and Section IX concludes.")

# ---------------------------------------------------------------- II
h1("II", "Related Work")
h2("A", "USB Peripheral Security")
body_lit(SUPP["related_a1"])
body_lit(SUPP["related_a2"])
h2("B", "eBPF for Kernel Observability")
body("eBPF runs verified programs inside the Linux kernel. Song [5] surveys its mechanics "
     "and limits, Kostopoulos et al. [3] build a real-time eBPF and machine-learning network "
     "intrusion system, and Rajurkar and Deshmukh [4] identify the common pattern of "
     "extracting features in the kernel and inferring in userspace. This work is about "
     "networks; we use the same pattern on USB, with usbmon as the deployed capture and an "
     "eBPF counting prototype measured against it.")
h2("C", "Temporal Models for Behavioural Anomalies")
body("LSTM networks [18] suit anomalies that lie in temporal structure. Hnamte et al. [7] "
     "combine an LSTM with an auto-encoder for network intrusion detection, Baek and Jung [6] "
     "separate anomaly detection from attack classification in two stages, and Saini et "
     "al. [8] use CNN–LSTM hybrids. Chithik Raja and Al Mahri [9] and Mohamed Anwar et al. "
     "[10] report LSTM-based detection of zero-day and IIoT attacks. Isolation Forests [17] "
     "are a standard unsupervised first stage. None of this work applies a learned temporal "
     "model to one USB device’s URB stream, and none tests transfer from simulated to real "
     "kernel traffic, which is where Section VI-D finds the largest gap.")

# ---------------------------------------------------------------- III
h1("III", "Threat Model, Scope and Terminology")
h2("A", "Threat Model and Scope")
body("The attacker supplies a USB device that presents a HID interface able to send "
     "keystrokes, alone or beside other interfaces, and uses it to type on the host. The "
     f"six attack families ({tref('families')}) cover machine-speed injection, injection "
     "after a long benign dwell, a storage device that comes back as a keyboard, a keyboard "
     "that sends covert bulk data, and two human-paced attacks. The defender runs Linux "
     "with usbmon available and no special hardware. Out of scope: attacks that never send "
     "keystrokes (a network adapter that hijacks traffic, a drive carrying malware), "
     "attacks below the USB protocol, and physical or electrical attacks. Section IV-C "
     "shows how the system reports devices outside this scope rather than judging them.")
table("The six attack families", ["Family", "What the device does", "Main signal"],
      [["hid_injection", "Types a scripted payload at machine speed soon after plug-in",
        "Speed, regularity"],
       ["staged_delay", "Types like a person for about half the session, then injects",
        "Burst after a dwell"],
       ["identity_switch", "Works as storage, then a keyboard appears and injects",
        "New interface or device"],
       ["class_exfil", "A declared keyboard that sends covert bulk data out",
        "Transfer type outside its class"],
       ["mimicry", "Replays a fixed short script in a loop at human speed",
        "Exact repeats across windows"],
       ["mimicry_adaptive", "Types a finite payload once at human speed, then goes silent",
        "None strong (Section VI-C)"]],
      key="families", left_cols=(0, 1, 2))
h2("B", "Terminology")
body("These terms are used with one meaning throughout.")
for term, meaning in (
        ("Window", "one device’s traffic in one second; the unit the models score."),
        ("Session", "one device from attachment to removal, or one simulated device run."),
        ("Window-level recall / false-positive rate", "the fraction of post-onset attack "
         "windows, or of benign windows, scored at or above the threshold."),
        ("Session-level detection rate", "the fraction of attack sessions in which the "
         "4-of-6 gate fires at least once; always given with its count, k/n."),
        ("Session-level false-positive rate", "the fraction of benign sessions in which the "
         "gate fires at least once, with its count."),
        ("False alarms per device-hour", "gate firings on benign devices divided by the "
         "hours they were observed; the unit that matters for continuous monitoring."),
        ("Latency", "time from attack onset to the end of the window that completed the "
         "gate."),
        ("Learned model alone / deployed decision rule", "the model’s own gate, versus the "
         "full rule of Section IV-C, which adds the capability gate and competence check.")):
    item("–", f"<i>{term}:</i> {meaning}")

# ---------------------------------------------------------------- IV
h1("IV", "System")
h2("A", "Capture and Features")
body("usbmon, a passive kernel tap, writes one text line per URB submission, completion or "
     "error. A parser turns each line into a typed event; completions and errors are used "
     "for features, submissions are not. Events are bucketed by kernel timestamp into "
     "one-second windows per device. The text timestamp is the monotonic clock with its "
     "seconds truncated to 12 bits, so it wraps every 4,096 s; the parser undoes that wrap "
     "(Section VI-E records the defect we found here). Each window becomes 30 features in "
     f"five modalities ({tref('features')}). Two are not given to the model: time since "
     "enumeration, which a Kolmogorov–Smirnov test showed to encode session position and "
     "hence the label, and the host-level rate, which is constant on single-device training "
     "data. Payload bytes are compared only for equality, so a stored trace replaces each "
     "payload with a keyed, length-preserving pseudonym and every feature stays exact.")
FD = MC["features"]
rows = [[f["index"], f["name"].replace("_", " "), f["unit"],
         "yes" if f["used_by_deployed_model"] else "no"] for f in FD]
table("Input features in their frozen order (definitions and normalisation in the model card)",
      ["#", "Feature", "Unit", "Used"], rows, key="features", left_cols=(1,))
h2("B", "Two-Stage Detector")
body("<i>Stage 1</i> is unsupervised and trained on benign windows only. Each feature is "
     "z-scored with the benign training mean and standard deviation, with the deviation "
     "floored at 0.01 and the result clipped to ±10, so a feature that is constant in benign "
     "traffic cannot explode when it changes. An Isolation Forest per declared class group "
     "scores the window against devices of the same kind, and a per-device exponentially "
     "weighted baseline scores it against the device’s own history; the two are combined "
     f"0.55/0.45 into one score. <i>Stage 2</i> is a one-layer LSTM over the last ten "
     "windows, each holding the deployed features plus the Stage-1 score; its last hidden "
     "state feeds one linear unit and a sigmoid, the probability that the newest window is "
     f"post-onset. {tref('settings')} lists every setting. Sequences are left-padded by "
     "repeating the earliest row, so scoring is strictly causal, and a test pins the "
     "streaming detector to the offline scores exactly.")
S1C, S2C = MC["stage1"], MC["stage2"]
rows = [["Stage 1", "Isolation Forest", f"{S1C['n_estimators']} trees, contamination "
         f"{S1C['contamination']}, max samples min(512, n)"],
        ["", "Device baseline", "EWMA α 0.15, warm-up 4 windows, update gate 0.55"],
        ["", "Normalisation", "z-score, SD floor 0.01, clip ±10"],
        ["Stage 2", "Input", f"10 windows × {DEP['feature_idx'].__len__() + 1} values"],
        ["", "Architecture", "LSTM, 1 layer, 32 hidden; Linear(32→1); sigmoid"],
        ["", "Loss", "binary cross-entropy (with logits)"],
        ["", "Class weighting", "positive weight = negatives / positives"],
        ["", "Optimiser", "Adam, lr 2×10⁻³, no weight decay, grad-norm clip 5"],
        ["", "Batch, epochs", f"{S2C['batch_size']}, {S2C['epochs']} (fixed; no early stopping)"],
        ["", "Seed", "7 (seeds 8, 9 for spread)"],
        ["Decision", "Threshold", f"{DEP['threshold']:.6f}, ≤ 0.01 false alarms/device-h"],
        ["", "Gate", "4 of last 6 windows; 8 qualifying → revoke"],
        ["", "Capability gate", f"keyboard ≥ {DEP['capability_rule']['human_max_rps']} "
                                "reports/s, own 4-of-6"]]
table("Model and decision settings", ["Stage", "Item", "Setting"], rows, key="settings",
      left_cols=(0, 1, 2))

new_section(1)
p = doc.add_paragraph(style="IEEE Figure")
p.add_run().add_picture(str(FIG), width=Inches(7.16))
para("IEEE Caption",
     "Fig. 1.  The IUDBMS pipeline. Stage 1’s score is an input to Stage 2, not a separate "
     "vote. The policy engine alarms on the model’s calibrated threshold or on the "
     "capability gate, and never escalates on device types outside the model’s competence. "
     "Dashed blocks are a prototype (eBPF) or generated but never executed (enforcement).")
new_section(2)

h2("C", "Decision Rule")
body_lit(SUPP["ivc_1"])
body_lit(SUPP["ivc_2"])
h2("D", "Enforcement and Deployment")
body_lit(SUPP["ivd_1"])

# ---------------------------------------------------------------- V
h1("V", "Evaluation Method")
h2("A", "Simulated Traffic")
body_lit(SUPP["va_intro"])
item_lit("–", lead(SUPP["va_b1"], "Typing speed:"))
item_lit("–", lead(SUPP["va_b2"], "Key releases:"))
item_lit("–", lead(SUPP["va_b3"], "Text:"))
item_lit("–", lead(SUPP["va_b4"], "Idle periods:"))
item_lit("–", lead(SUPP["va_b5"], "Session length and attack onset:"))
body_lit(SUPP["va_three"])
body_lit(SUPP["va_tableref"])
H = HT
rows = [["People (sessions)", f"{H['aalto']['sessions']:,}", f"51 ({H['cmu']['sessions']})",
         f"({H['simulator']['sessions']})"]]
for k, lbl, f in (("iki_median_ms", "Inter-key median (ms)", "{:.0f}"),
                  ("iki_p10_ms", "Inter-key 10th pct. (ms)", "{:.0f}"),
                  ("hold_median_ms", "Key hold median (ms)", "{:.0f}"),
                  ("rollover_share", "Rollover share", "{:.1%}"),
                  ("gaps_over_1s_share", "Gaps over 1 s", "{:.1%}"),
                  ("reports_per_s_sustained_max", "Fastest sustained reports/s", "{:d}")):
    rows.append([lbl] + [f.format(H[s][k]) for s in ("aalto", "cmu", "simulator")])
table("Simulated typing against two real populations", ["Measure", f"Aalto [{REF_NO['dhakal']}]", f"CMU [{REF_NO['killourhy']}]",
      "Simulator"], rows, key="timing")
_ill = SUPP["va_illustration"]
_snip = "C Ii:3:003:1 0:10 8 = 00000b00…"
assert _snip in _ill
body_lit(_ill.replace(_snip, "<code>" + _snip + "</code>"))
h2("B", "Corpora and Splits")
body_lit(SUPP["vb_1"])
B = MC["data"]["corpus_B"]["splits"]
A = MC["data"]["corpus_A"]["splits"]
rows = []
for nm, S in (("A", A), ("B", B)):
    for sp in ("fit (train)", "validation", "test"):
        v = S[sp]
        rows.append([f"{nm} {sp.split()[0]}", v["sessions"], v["benign"], v["attack"],
                     f"{v['windows']:,}", f"{v['hours']:.1f}"])
rows.append(["C (all)", 3600, 3000, 600, "—", "—"])
table("Sessions, windows and hours per split", ["Split", "Sessions", "Benign", "Attack",
      "Windows", "Hours"], rows, key="data")
h2("C", "Real-Kernel Testbed")
body_lit(SUPP["vc_1"])
body_lit(SUPP["vc_2"])
h2("D", "Statistics, Pre-Registration and Availability")
body_lit(SUPP["vd_1"])

# ---------------------------------------------------------------- VI
h1("VI", "Results")
h2("A", "Detection in Simulation")
body_lit(SUPP["via_1"])
R = RULEJ
H_ = HYB["corpus_c"]
rows = []
for f in FAMS:
    rows.append([f.replace("_", " "), f"{pct(cb['per_profile'][f], 0)}",
                 f"{pct(ca['per_profile'][f], 0)}", f"{pct(H_['rule']['per_profile'][f]['rate'], 0)}",
                 f"{pct(H_['hybrid']['per_profile'][f]['rate'], 0)}"])
rows.append(["All (n = 600)", f"{cb['detect_k']}", f"{ca['detect_k']}", f"{H_['rule']['detect_k']}",
             f"{H_['hybrid']['detect_k']}"])
rows.append(["Benign FP (n = 3,000)", f"{cb['fp_k']}", f"{ca['fp_k']}", f"{H_['rule']['fp_k']}",
             f"{H_['hybrid']['fp_k']}"])
table("Session-level detection on Corpus C (100 sessions per family)",
      ["Family", "Per-window thr.", "Calibrated", "Capability gate", "Deployed rule"],
      rows, key="simdetect")
body_lit(SUPP["via_2"])
h2("B", "From a Window Threshold to False Alarms per Device-Hour")
body_lit(SUPP["vib_1"])
rows = [["Keyboard windows over threshold", f"{sfp_old['p_window_over']:.1e}",
         f"{sfp_new['p_window_over']:.1e}"],
        ["P(next over | over)", f"{sfp_old['q_next_over_given_over']:.2f}",
         f"{sfp_new['q_next_over_given_over']:.2f}"],
        ["Alarms/h predicted if independent", f"{sfp_old['naive_alarms_per_hour']:.1e}",
         f"{sfp_new['naive_alarms_per_hour']:.1e}"],
        ["Alarms/h observed (40 h)", f"{sfp_old['observed_alarms_per_hour']:.3f}",
         f"{sfp_new['observed_alarms_per_hour']:.3f}"],
        ["Fresh keyboard alarms (80 h)", f"{kb_b['alarms']} ({kb_b['rate']:.3f}/h)",
         f"{kb_a['alarms']} ({kb_a['rate']:.3f}/h)"],
        ["Corpus C detection", kn(cb["detect_k"], cb["n_attack"]), kn(ca["detect_k"], ca["n_attack"])],
        ["Corpus C benign FP", f"{cb['fp_k']}/3,000", f"{ca['fp_k']}/3,000"],
        ["Median latency", f"{cb['median_latency']:.1f} s", f"{ca['median_latency']:.1f} s"]]
table("Two operating points of the same model", ["Measure",
      f"Per-window ({CAL['previous_threshold']:.4f})", f"Calibrated ({CAL['chosen_threshold']:.4f})"],
      rows, key="operating")
body_lit(SUPP["vib_2"])
h2("C", "Generalisation to Unseen Attack Families")
body_lit(SUPP["vic_1"])
fl = {"hid_injection": "hid", "staged_delay": "stag", "identity_switch": "iden",
      "class_exfil": "exfil", "mimicry": "mim", "mimicry_adaptive": "mim-a"}
rows = []
for F in FAMS:
    rows.append([fl[F]] + [f"{100 * TR['matrix'][F][G]['rate']:.0f}" for G in FAMS])
table("Detection (%) of each family (columns) by a model trained on one family (rows)",
      ["Trained on"] + [fl[g] for g in FAMS], rows, key="transfer")
st = TR["overlap"]["strong"]
body_lit(SUPP["vic_2"])
body_lit(SUPP["vic_3"])
h2("D", "Real Kernels")
rows = []
for k in ("ubuntu-24.04", "debian-13", "fedora-44", "alma-9"):
    d = dists[k]
    drv = "all classes" if "cdc_ether" in d["drivers"] else "HID, storage"
    rows.append([d["os"].replace(" (Olive Jaguar)", "").replace(" (Cloud Edition)", "")
                 .replace(" GNU/Linux", "").replace(" (trixie)", ""),
                 d["kernel"].split("-")[0], f"{d['usbmon_lines']:,}", drv])
table(f"Testbed environments ({tb_minutes / len(dists):.0f} min each, {tb_minutes / 60:.0f} h in all)",
      ["Distribution", "Kernel", "usbmon lines", "Drivers bound"], rows, key="envs",
      left_cols=(0, 3))
body_lit(SUPP["vid_1"])
BR = TB["benign_by_role"]
assert tb_tot["rule_alarmed"] == 0       # so the model-alone benign column equals the deployed one
rows = [["Benign devices alarmed", f"{int(tb_tot['deployed_alarmed'])}/{int(tb_tot['devices'])}",
         f"{int(tb_tot['deployed_alarmed'])}/{int(tb_tot['devices'])}",
         f"{int(tb_tot['previous_threshold_alarmed'])}/{int(tb_tot['devices'])}",
         f"{int(tb_tot['rule_alarmed'])}/{int(tb_tot['devices'])}",
         f"{int(tb_tot['corpus_a_alarmed'])}/{int(tb_tot['devices'])}"],
        ["  of which keyboards (human input)", f"{int(kbd_role['deployed_alarmed'])}/{int(kbd_role['devices'])}",
         f"{int(kbd_role['deployed_alarmed'])}/{int(kbd_role['devices'])}",
         f"{int(kbd_role['previous_threshold_alarmed'])}/{int(kbd_role['devices'])}",
         f"{int(kbd_role['rule_alarmed'])}/{int(kbd_role['devices'])}",
         f"{int(kbd_role['corpus_a_alarmed'])}/{int(kbd_role['devices'])}"]]
for f in FAST + ["mimicry", "mimicry_adaptive"]:
    v = tb_att[f]
    rows.append([f.replace("_", " "), f"{v['deployed']}/{v['n']}", f"{v['model_calibrated']}/{v['n']}",
                 f"{v['model_previous']}/{v['n']}", f"{v['rule']}/{v['n']}", f"{v['corpus_a']}/{v['n']}"])
table(f"Real kernels: benign devices ({tb_tot['hours']:.1f} device-hours) and attack episodes. "
      "† model alone, calibrated threshold, competence checks on; ‡ model alone, per-window "
      "threshold, every device judged",
      ["", "Deployed", "Model, cal.†", "Model, per-win.‡", "Gate only", "Corpus A"],
      rows, key="realkernel", left_cols=(0,))
body_lit(SUPP["vid_t1"])
body_lit(SUPP["vid_2"])
h2("E", "What Did Not Help")
rs = ROBF["summary"]
body_lit(lead(SUPP["vie_1"], "Sample-size-robust features."))
body_lit(lead(SUPP["vie_2"], "Polling-limited injection."))
body_lit(lead(SUPP["vie_3"], "Earlier negative results."))
h2("F", "Cost and Overhead")
body_lit(SUPP["vif_1"])
if OVH:
    OS_ = OVH["summary"]
    rows = []
    for c, lbl in (("usbmon", "usbmon (deployed)"), ("ebpf_count", "eBPF, counting"),
                   ("ebpf_event", "eBPF, per-event"), ("both", "usbmon + eBPF")):
        s_ = OS_[c]
        wl, wh = s_["write_MBps_vs_none_ci"]
        rl, rh = s_["read_MBps_vs_none_ci"]
        ns = s_.get("ebpf_ns_per_call")
        rows.append([lbl, f"{100 * s_['write_MBps_vs_none']:+.0f}% [{100 * wl:+.0f}, {100 * wh:+.0f}]",
                     f"{100 * s_['read_MBps_vs_none']:+.0f}% [{100 * rl:+.0f}, {100 * rh:+.0f}]",
                     f"{s_['cpu_jiffies'][0]:.0f}", f"{ns[0] / 1000:.1f}" if ns else "—"])
    nn = OS_["none"]
    rows.append(["None (reference)", f"{nn['write_MBps'][0]:.1f} MB/s", f"{nn['read_MBps'][0]:.1f} MB/s",
                 f"{nn['cpu_jiffies'][0]:.0f}", "—"])
    table(f"Monitoring cost under USB storage load ({OVH['stick_mb']} MB in {OVH['bs_kib']} KiB "
          f"direct I/Os, {OVH['reps']} paired repetitions; change vs. none, 95% CI)",
          ["Monitor", "Write", "Read", "CPU (jiffies)", "eBPF µs/URB"], rows,
          key="overhead", left_cols=(0,))
    ns_max = max(OS_[c]["ebpf_ns_per_call"][0] for c in ("ebpf_count", "ebpf_event", "both"))
    cis = [OS_[c][k] for c in ("usbmon", "ebpf_count", "ebpf_event", "both")
           for k in ("write_MBps_vs_none_ci", "read_MBps_vs_none_ci")]
    n_ci, n_contain = len(cis), sum(lo <= 0 <= hi for lo, hi in cis)
    # an interval that excludes zero must do so on the FASTER side, or this text is wrong
    assert all(lo > 0 for lo, hi in cis if not lo <= 0 <= hi), "a monitor measurably slowed I/O"
    _need(n_contain == 7 and n_ci == 8 and f"{ns_max / 1000:.1f}" == "3.8",
          "7 of 8 intervals contain zero; 3.8 us per URB completion")
    body_lit(SUPP["vif_2"])

# ---------------------------------------------------------------- VII
h1("VII", "What the Evidence Supports")
body_lit(SUPP["vii_1"])
rows = [["N1 Multi-modal fusion", "Built", "Partly: helps at moderate attack strength, loses at "
         "the subtlest (simulation)"],
        ["N2 Per-device baseline", "Built", "Not supported (simulation)"],
        ["N3 Closed-loop enforcement", "Generated, not executed", "Untested"],
        ["N4 Temporal modelling", "Built", f"Supported in simulation ({pct(s1_a)} → {pct(det_a)}); "
         "did not transfer to real kernels at a usable threshold"],
        ["Device-hour calibration", "Built, deployed", "Supported: simulation and real kernels"],
        ["Competence checks", "Built, deployed", "Supported on real kernels"],
        ["Capability gate", "Built, deployed", "Supported: unseen families, real kernels"],
        ["Zero-day detection", "Not claimed", "Unseen human-paced attacks are missed"]]
table("Claims, implementation status and evidence", ["Claim", "Status", "Evidence"], rows,
      key="claims", left_cols=(0, 1, 2))
body_lit(SUPP["vii_2"])

# ---------------------------------------------------------------- VIII
h1("VIII", "Limitations and Threats to Validity")
item("1)", "<i>Simulated training data.</i> Every model is trained on simulated traffic, and "
           "the real-kernel results show that the learned component does not transfer at a "
           "usable operating point.")
item("2)", "<i>Emulated devices and recorded users.</i> The testbed’s kernels and drivers are "
           "real, but the devices are QEMU’s and the input is replayed, not live. No physical "
           "device, real attack hardware or live user has been tested. A capture tool that "
           "pseudonymises payloads as it records exists for contributors with their own "
           "machines and devices; no contribution has been analysed yet.")
item("3)", "<i>Small real-kernel attack set.</i> Forty episodes, eight per family, on one "
           "QEMU keyboard model with a 10 ms polling interval. A device declaring a slower "
           "interval, or typing just under 30 reports per second, would evade the "
           "capability gate.")
item("4)", "<i>Human-paced attacks are not detected.</i> Neither the model nor the gate "
           "catches a payload typed at human speed.")
item("5)", "<i>Enforcement is not applied.</i> Actions are generated and logged only, so the "
           "closed loop and its latency are untested.")
item("6)", "<i>Assumptions in the generator.</i> Idle mixes, text model and onsets are our "
           "choices; the replay-loop result swings from 97% to 1% on one attacker choice.")
item("7)", "<i>Post-hoc design.</i> The HID competence check and the 0.8 overlap threshold "
           "were introduced after seeing data; both are reported as such, and the check was "
           "validated only on sessions run after it was designed.")

# ---------------------------------------------------------------- IX
h1("IX", "Conclusion")
body("IUDBMS treats USB trust as a question about behaviour over time. As built, it is a "
     "working Linux monitor whose alarm threshold is set in false alarms per device-hour, "
     "which declines to judge device types its model never saw, and which pairs a learned "
     "sequence model with a bound on human typing speed. In simulation it detects "
     f"{kn(ca['detect_k'], ca['n_attack'])} of fresh attack sessions at no false positive in "
     f"3,000 benign ones. On four real kernels it raised no false alarm in "
     f"{tb_tot['hours']:.0f} device-hours and caught every injection-speed attack, but those "
     "detections came from the typing-speed bound, and human-paced attacks went undetected. "
     "The work is a simulation-based proof of concept with a real-kernel testbed, not a "
     "tested security product. The next steps are recordings of real users on physical "
     "devices, real attack hardware, a review of the enforcement path before it touches a "
     "kernel, and features that can see a human-paced payload.")

# ---------------------------------------------------------------- refs
p = doc.add_paragraph(style="IEEE H1")
run(p, "References", small_caps=True)

REFS = {
    "shen": "X. Shen, N. Li, C. Liu, and M. Cai, “WMUD: A Whitelist Method of USB Devices Redirection for Virtual Desktop,” in 2019 IEEE 21st Int. Conf. on High Performance Computing and Communications; IEEE 17th Int. Conf. on Smart City; IEEE 5th Int. Conf. on Data Science and Systems (HPCC/SmartCity/DSS), 2019, pp. 527–533, doi: 10.1109/HPCC/SmartCity/DSS.2019.00083.",
    "bintang": "R. M. G. S. Bintang, Arizal, R. Purwoko, M. Syahral, S. S. Zen, I. G. M. Putra, and R. A. Wijayanti, “SafeUSB: Implementation of Keystroke Monitoring System to Prevent BadUSB HID Injection Attack,” in 2025 IEEE 2nd Int. Conf. on Cryptography, Informatics, and Cybersecurity (ICoCICs), 2025, pp. 286–291, doi: 10.1109/ICOCICS68032.2025.11383963.",
    "kosto": "S. Kostopoulos, D. Papatsaroucha, I. Kefaloukos, and E. K. Markakis, “eIDPS: A real-time eBPF-based and Machine Learning-powered Network Intrusion Detection and Prevention Solution,” in 2025 6th Int. Conf. in Electronic Engineering & Information Technology (EEITE), 2025, doi: 10.1109/EEITE65381.2025.11165804.",
    "rajurkar": "R. Rajurkar and N. Deshmukh, “From Packets to Defense: Architectural Patterns in eBPF-Based Intrusion Detection,” in 2025 Int. Conf. on Intelligent Computing and Next Generation Networks (ICNGN), 2025, doi: 10.1109/ICNGN67480.2025.11413805.",
    "song": "L. Song, “eBPF: Pioneering Kernel Programmability and System Observability – Past, Present, and Future Insights,” in 2024 3rd Int. Conf. on Artificial Intelligence and Computer Information Technology (AICIT), 2024, doi: 10.1109/AICIT62434.2024.10730620.",
    "baek": "M. S. Baek and Y.-A. Jung, “A Two-Stage Deep Learning-based Framework for High-Accuracy Network Intrusion Detection,” in 2025 16th Int. Conf. on Information and Communication Technology Convergence (ICTC), 2025, doi: 10.1109/ICTC66702.2025.11388157.",
    "hnamte": "V. Hnamte, H. Nhung-Nguyen, J. Hussain, and Y. Hwa-Kim, “A Novel Two-Stage Deep Learning Model for Network Intrusion Detection: LSTM-AE,” IEEE Access, vol. 11, pp. 37131–37148, 2023, doi: 10.1109/ACCESS.2023.3266979.",
    "saini": "A. Saini, T. K. Jaiswal, Parteek, A. Thakur, A. Rai, and S. Saini, “Hybrid CNN–LSTM Architecture for Cybersecurity Threat Detection in Network Traffic,” in 2026 Int. Conf. on Signal, Systems, and Computing for Next-Gen Automation (ICSSCNA), 2026, doi: 10.1109/ICSSCNA68616.2026.11546877.",
    "raja": "M. Chithik Raja and M. M. B. Al Mahri, “Behavioral Anomaly Detection in Big Data Streams: An Attention-Enhanced LSTM Approach for Privacy-Aware Zero-Day Attack Detection,” in 2025 7th Int. Conf. on Innovative Data Communication Technologies and Application (ICIDCA), 2025, doi: 10.1109/ICIDCA66325.2025.11280566.",
    "anwar": "A. Mohamed Anwar, D. Vikram, Thiraviaselvi G, and S. Ganapathy, “LSTM-Powered Anomaly Detection for Securing IIoT Systems Against DDoS, Spoofing, and Ransomware Attacks,” in 2025 Int. Conf. on Emerging Research in Computational Science (ICERCS), 2025, doi: 10.1109/ICERCS65898.2025.11580832.",
    "shannon": "C. E. Shannon, “Prediction and Entropy of Printed English,” Bell Syst. Tech. J., vol. 30, no. 1, pp. 50–64, Jan. 1951, doi: 10.1002/j.1538-7305.1951.tb01366.x.",
    # added after the audit; each was checked against the publisher or proceedings page
    "nohl": ("K. Nohl and J. Lell, “BadUSB—On accessories that turn evil,” presented at ",
             "Black Hat USA", ", Las Vegas, NV, USA, Aug. 2014."),
    "goodusb": ("D. Tian, A. Bates, and K. Butler, “Defending against malicious USB firmware with GoodUSB,” in ",
                "Proc. 31st Annu. Comput. Security Appl. Conf. (ACSAC)",
                ", Los Angeles, CA, USA, Dec. 2015, pp. 261–270."),
    "usbfilter": ("D. Tian, N. Scaife, A. Bates, K. Butler, and P. Traynor, “Making USB great again with USBFILTER,” in ",
                  "Proc. 25th USENIX Security Symp.", ", Austin, TX, USA, Aug. 2016."),
    "usblock": ("S. Neuner, A. G. Voyiatzis, S. Fotopoulos, C. Mulliner, and E. R. Weippl, “USBlock: Blocking USB-based keypress injection attacks,” in ",
                "Proc. 32nd Annu. IFIP WG 11.3 Conf. Data Appl. Security Privacy (DBSec)",
                ", Bergamo, Italy, Jul. 2018, Lecture Notes in Computer Science, vol. 10980."),
    "huang": ("C.-Y. Huang, H.-M. Lee, J.-C. Wang, and C.-H. Mao, “Identifying HID-based attacks through process event graph using guilt-by-association analysis,” in ",
              "Proc. 3rd Int. Conf. Cryptography, Security Privacy (ICCSP)", ", Jan. 2019."),
    "iforest": ("F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in ",
                "Proc. 8th IEEE Int. Conf. Data Mining (ICDM)", ", 2008, pp. 413–422."),
    "lstm": ("S. Hochreiter and J. Schmidhuber, “Long short-term memory,” ", "Neural Comput.",
             ", vol. 9, no. 8, pp. 1735–1780, 1997."),
    "dhakal": ("V. Dhakal, A. M. Feit, P. O. Kristensson, and A. Oulasvirta, “Observations on "
               "typing from 136 million keystrokes,” in ",
               "Proc. CHI Conf. Human Factors in Computing Systems",
               ", Montréal, QC, Canada, 2018, doi: 10.1145/3173574.3174220."),
    "killourhy": ("K. S. Killourhy and R. A. Maxion, “Comparing anomaly-detection algorithms "
                  "for keystroke dynamics,” in ",
                  "Proc. IEEE/IFIP Int. Conf. Dependable Systems and Networks (DSN)",
                  ", Lisbon, Portugal, 2009, pp. 125–134, doi: 10.1109/DSN.2009.5270346."),
    "balabit": ("Á. Fülöp, L. Kovács, T. Kurics, and E. Windhager-Pokol, ",
                "Balabit Mouse Dynamics Challenge data set",
                ", 2016. [Online]. Available: https://github.com/balabit/Mouse-Dynamics-Challenge"),
    "hidspec": ("USB Implementers Forum, ",
                "Device Class Definition for Human Interface Devices (HID), Version 1.11",
                ", Jun. 2001."),
}

missing = [k for k in REFS if k not in CITED]
assert not missing, f"references never cited: {missing}"
assert set(REF_NO) == set(REFS), (set(REF_NO) ^ set(REFS))
_order = [REF_NO[k] for k in CITED_ORDER]
assert _order == sorted(_order), f"references are not numbered in order of first citation: {_order}"
for key, n in sorted(REF_NO.items(), key=lambda kv: kv[1]):
    ref = REFS[key]
    p = doc.add_paragraph(style="IEEE Ref")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3), WD_TAB_ALIGNMENT.LEFT)
    run(p, f"[{n}]\t")
    if isinstance(ref, tuple):
        run(p, ref[0]); run(p, ref[1], italic=True); run(p, ref[2])
        continue
    m = re.search(r"” in (.+?), (20\d\d)(?=[,.])", ref)
    if m:
        run(p, ref[:m.start(1)])
        run(p, m.group(1), italic=True)
        run(p, ref[m.end(1):])
    else:
        journal = "IEEE Access" if "IEEE Access" in ref else "Bell Syst. Tech. J."
        h_, t_ = ref.split(journal, 1)
        run(p, h_)
        run(p, journal, italic=True)
        run(p, t_)

cp = doc.core_properties
cp.title = TITLE
cp.author = ", ".join(n for n, _ in AUTHORS)
cp.subject = "IEEE conference format"
# python-docx's default template carries a 2013 creation date; stamp the real build time,
# so Word shows when this version was produced.
import datetime as _dt
cp.created = cp.modified = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
cp.last_modified_by = cp.author
cp.revision = 1
cp.keywords = "BadUSB, behavioural anomaly detection, false alarms per device-hour, LSTM, USB security"

DEST.parent.mkdir(parents=True, exist_ok=True)
# Replace the file instead of overwriting it in place: macOS keeps a file's original
# creation date across in-place writes, so the first build's date stayed on it.
DEST.unlink(missing_ok=True)
doc.save(DEST)
print(f"  wrote {DEST}")
print(f"  tables: {_TABLE_N[0]}   figure: {FIG.name}")
for _n in NOTES:
    print(f"  NOTE (supplied text vs results): {_n}")
