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
REF_NO: dict[str, int] = {}


def _cite_no(old):
    key = OLD2KEY[old]
    if key not in REF_NO:
        REF_NO[key] = len(REF_NO) + 1
    return REF_NO[key]


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


def rich(par, text, **base):
    italic = False
    code = False
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


def item(label, text):
    p = doc.add_paragraph(style="IEEE List")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.25), WD_TAB_ALIGNMENT.LEFT)
    run(p, f"{label}\t")
    rich(p, text)
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


# ---------------------------------------------------------------- abstract
ABSTRACT = (
    "USB hosts trust what a peripheral declares, so a BadUSB device that declares itself a "
    "keyboard can type a payload seconds after it is plugged in. IUDBMS judges devices by "
    "behaviour instead. It reads the Linux usbmon stream, summarises each device-second as "
    "30 features, scores them with an Isolation Forest and an LSTM over ten-second "
    "sequences, and alarms when four of six seconds look malicious. Enforcement is generated "
    "but not applied, and the models are trained on simulated traffic, so the work is a "
    "simulation-based proof of concept. On 3,600 fresh simulated sessions, with the alarm "
    "threshold calibrated to 0.01 false alarms per device-hour, it detects "
    f"{ca['detect_k']} of {ca['n_attack']} attack sessions ({pct(ca['detect'])}) with no false "
    f"alarm in {ca['n_benign']:,} benign ones. A typing-speed bound measured on 1,972 people raises "
    f"detection of attack families withheld from training from {pct(lopo_m)} to "
    f"{pct(lopo_h)}. On four Linux distributions with emulated devices driven by recorded "
    f"human input, none of {int(tb_tot['devices'])} benign devices alarmed in "
    f"{tb_tot['hours']:.0f} device-hours and "
    f"{fast_dep} of {fast_n} fast injection attacks were caught, but by the speed bound: "
    f"the learned model alone caught {model_cal_all} of {n_att_tb}, and human-paced attacks "
    "went undetected."
)
n_words = len(re.findall(r"\S+", ABSTRACT))
assert n_words <= 200, f"abstract is {n_words} words; IEEE allows at most 200"
print(f"  abstract: {n_words} words")

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Abstract", italic=True)
run(p, "—" + ABSTRACT)
p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Index Terms", italic=True)
run(p, "—BadUSB, behavioural anomaly detection, false alarms per device-hour, HID, "
       "Isolation Forest, LSTM, USB security, usbmon.")

# ---------------------------------------------------------------- I
h1("I", "Introduction")
body("A USB host believes what a device says about itself. During enumeration the device "
     "reports its vendor and product identifiers and its interfaces, and the host binds "
     "drivers to them with no attestation. BadUSB [12] turns this into an attack: "
     "reprogrammed firmware lets a flash drive or a cable declare a keyboard and type a "
     "stored payload as soon as it is plugged in. The operating system treats those "
     "keystrokes as a person’s, so the payload runs with the user’s privileges.")
body("Defences in use today check identity. Allowlists admit devices by class and "
     "identifier [1], GoodUSB asks the user which functions a device should have [13], and "
     "USBFILTER enforces packet rules in the USB stack [14]. A device that copies an allowed "
     "identity passes all of them. Behavioural monitors exist for keystrokes [2], [15], but "
     "they are rule- or threshold-based, and the learned temporal models that work well "
     "for network intrusion detection [6]–[10] have not been applied to the event stream of a "
     "single USB device. Where learned detectors are evaluated, the figures are per window "
     "or per short session on synthetic or benchmark traffic. A monitor watches each device "
     "for a working day, so the number that matters to an operator is false alarms per "
     "device-hour, and a model trained on simulated USB traffic has to be checked against "
     "real kernel output before any result means much.")
body("IUDBMS is our attempt to close that gap, and this paper reports how far it gets. It "
     "is a Linux monitor that reads every USB request block (URB) through usbmon, scores "
     "each device once a second, and escalates through a persistence gate. The "
     "contributions are listed once here, each with the scope of its evidence:")
item("1)", "<i>A working pipeline</i> (implemented): capture, features, a two-stage "
           "detector, a gated policy, storage, dashboard and systemd services, running on "
           "real kernels. Enforcement is generated but not applied.")
item("2)", "<i>Operating-point calibration in false alarms per device-hour</i> (simulation): "
           "a per-window false-positive target does not predict session-level false alarms "
           "(the independence model is off by a factor of millions), so the threshold is "
           "chosen on gate firings over long benign sessions.")
item("3)", "<i>A human-capability bound</i> (simulation and real kernels): the fastest "
           "sustained typing of 1,972 people, used as a second gate beside the model. It "
           "explains and partly repairs the failure on unseen attack families.")
item("4)", "<i>A real-kernel testbed</i> (four distributions, 11 emulated USB device types, "
           "recorded human input) and what it showed: the learned model did not transfer at "
           "a usable operating point, and declining to judge device types the model never "
           "saw removes false alarms without losing keystroke coverage.")
item("5)", "<i>Negative results with their pre-registered criteria</i>, including two "
           "candidate improvements that were measured and not adopted.")
body("Section II reviews related work. Section III fixes the threat model and the terms used "
     "throughout. Section IV describes the system, Section V the evaluation method and "
     "Section VI the results. Section VII separates what is proposed, implemented and "
     "supported, Section VIII lists the limitations, and Section IX concludes.")

# ---------------------------------------------------------------- II
h1("II", "Related Work")
h2("A", "USB Peripheral Security")
body("Shen et al. [1] proposed WMUD, a server-side allowlist for USB redirection to virtual "
     "desktops. It keys policy on class and identifier, which shows the value of "
     "tamper-resistant policy but admits any device that presents an allowed identity. "
     "GoodUSB [13] mediates the Linux USB stack and grants a device only the functions the "
     "user expects. USBFILTER [14] filters packets between drivers and the host controller "
     "at a cost of microseconds per packet. These are identity- or rule-based.")
body("Closer to our setting, SafeUSB [2] monitors keystrokes on Windows with an "
     "inter-keystroke timing threshold and keyword matching, and reports roughly one second "
     "between connection and the start of monitoring. USBlock [15] detects injection devices "
     "from the timing of the packets they send, and Huang et al. [16] identify HID attacks "
     "from host process events. As far as we can tell from their descriptions, none of "
     "these learns a per-device model or models a device’s behaviour over a sequence of "
     "windows, and none reports false alarms per device-hour.")
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
body("A device alarms when 4 of its last 6 windows reach the threshold. The threshold is "
     "chosen in false alarms per device-hour (Section VI-B): the smallest value at which "
     "benign devices raise at most 0.01 alarms per hour over 80 hours of long simulated "
     "sessions. A second gate, with its own 4-of-6 count, fires when a keyboard sends at "
     f"least {DEP['capability_rule']['human_max_rps']} reports in a second; no human in a "
     f"population of 1,972 sustained more than {human['reports_per_s_sustained_max']} "
     "(Section V-A). The two gates are combined with OR exactly as they were evaluated.")
body("Two checks decide what is judged at all. The model records the interface classes of "
     "its benign training devices, and a device that declares any other class (a network "
     "adapter, a serial converter, a smart-card reader) is reported as <i>unmodelled</i>: "
     "scored and stored, never escalated. For HID interfaces the class is not enough, "
     "because a keyboard and a tablet share it. What a HID interface can send is declared "
     "by its report descriptor [22], which the kernel exposes in sysfs; keystrokes need the "
     "Keyboard page and applications can be launched through the Consumer page. A device "
     "that may send keystrokes is always judged, including when its descriptor cannot be "
     "read; a pure pointer is outside the threat model and reported as unmodelled.")
h2("D", "Enforcement and Deployment")
body("Escalation generates THROTTLE and then DEAUTHORIZE actions with the kernel command and "
     "the evidence behind them, and stores them. Nothing is written to the kernel’s "
     "authorization interface: execution is gated behind a manual safety review that has "
     "not been done, and the claim of closed-loop enforcement is therefore untested. The "
     "monitor and an operator API run as hardened systemd services, and checkpoints carry a "
     "fixed-input fingerprint that the loader checks.")

# ---------------------------------------------------------------- V
h1("V", "Evaluation Method")
h2("A", "Simulated Traffic")
body("No labelled USB attack traffic is public, so training data come from a simulator that "
     "writes sessions in usbmon’s exact text format; the same parser reads them back. Benign "
     "devices come from five profiles (keyboard, mouse, mass storage, webcam, audio). The "
     "choices that matter, and their sources:")
item("–", "<i>Typing speed:</i> each device draws its own median inter-key time, "
          "log-uniform over 60 to 300 ms, with log-normal spread.")
item("–", "<i>Key releases:</i> as the HID boot protocol specifies, every keystroke sends a "
          "key-down and a key-up report, and overlapping keys produce multi-key reports. "
          "Key holds are log-normal with median 90 ms.")
item("–", "<i>Text:</i> a Zipf-weighted vocabulary whose digram entropy is tuned to 3.56 "
          "bits, Shannon’s estimate for English [11].")
item("–", "<i>Idle periods:</i> 40% of benign devices are always active, 15% pause for 3 to "
          "8 s between bursts, 15% for 8 to 20 s, 20% for 20 to 60 s, and 10% idle after "
          "enumeration.")
item("–", "<i>Attack duration and onset:</i> every session, benign or attack, lasts 40 to "
          "85 s. Onset is 1.5 to 4 s for injection, 15 to 30 s for the identity switch, 45 "
          "to 62% of the session for the staged attack, 5 to 12 s for covert exfiltration "
          "and 4 to 12 s for the mimicry families; a human-paced payload is 60 to 220 keys.")
body("Three choices keep attacks from being easy. Attack devices draw their timing persona "
     "from the same prior as benign ones, the mimicry families draw keystroke timing from the "
     "exact human parameters, and benign and attack sessions share one duration "
     "distribution. A leakage audit tests duration, window count and identifier per family, "
     "and found the time-since-enumeration leak that is now excluded. "
     f"{tref('timing')} measures the simulator against real people: 1,972 participants of "
     "the 136M Keystrokes study [19] and the 51 subjects of the CMU benchmark [20]. The "
     "simulator types at a realistic median pace and hold time, pauses more than "
     "transcription typists, and has no very fast typists, which is why the capability "
     "bound is taken from the real data.")
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
table("Simulated typing against two real populations", ["Measure", "Aalto [19]", "CMU [20]",
      "Simulator"], rows, key="timing")
body("For illustration, two seconds of a benign keyboard emit lines such as "
     "<code>C Ii:3:003:1 0:10 8 = 00000b00…</code> a few dozen milliseconds apart, a key-down "
     "followed by its all-zero release; machine-speed injection emits the same pairs about "
     "2 ms apart for as long as its payload lasts. The model card lists real excerpts of "
     "both.")
h2("B", "Corpora and Splits")
body("Corpus A (1,000 sessions, 600 benign) is the original corpus, in which every benign "
     "device is always active. Corpus B has the same size and seed with the realistic "
     "generator of Section V-A, and the deployed model is trained on it. Corpus C is a "
     "fresh set of 3,600 sessions (3,000 benign, 100 per attack family) from a different "
     "seed, never used to train, calibrate or select. Splits are by session, never by "
     "window: 30% of each profile is held out for test, and 20% of the rest is drawn for "
     f"validation, which is where thresholds are set ({tref('data')}). Long-session sets for "
     "calibration and evaluation use their own seeds.")
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
body("To test on real kernels we boot four distributions headless under QEMU with hardware "
     "acceleration: Ubuntu 24.04 (kernel 6.8), Debian 13 (6.12), Fedora 44 (6.19) and "
     "AlmaLinux 9 (5.14). Each guest has 11 emulated USB devices: boot keyboard, mouse, "
     "absolute tablet, pen tablet, bulk-only and UAS storage, audio, network adapter, "
     "serial converter, smart-card reader and hub. The kernels and drivers are real; the "
     "device side is emulated. The keyboard replays key-down and key-up times of real "
     "participants [19], the mouse and tablet replay the Balabit recordings of ten users "
     "[21], and the guest copies files to the drives, sends network packets, writes to the "
     "serial port and plays audio on a seeded schedule. Each distribution runs a 75-minute "
     "benign session and two 30-minute attack sessions, in which six episodes are played "
     "on newly attached keyboards: the five attack families that a keyboard can carry out "
     "and a benign control in which a new keyboard is used by a replayed person. Attack "
     "payloads are innocuous text without the Enter key; what makes them attacks is the "
     "traffic the kernel sees. Captures are scored through the deployed monitor loop itself, "
     "replayed as fast as the file can be read; a test shows that replay gives the same "
     "verdicts as replay at wall-clock pace.")
h2("D", "Statistics, Pre-Registration and Availability")
body("Rates carry 95% Wilson intervals or counts. LSTM training is seed-sensitive, so "
     "comparisons that could hinge on the seed use three seeds, and a difference counts only "
     "if it exceeds both the seed spread and one session. Every candidate improvement had "
     "its adoption criterion written into its script before the first run, and two failed "
     "(Section VI-E). Every number here is read from a file written by a seeded script in "
     "the project repository, and this document is generated from those files. Seeds: 7 "
     "for Corpora A and B and model training (8 and 9 for spread), 2025 for Corpus C, 4049 "
     "for threshold calibration, 2027 to 2031 for the long-session evaluation sets, and 1 "
     "and 2 for the testbed rounds. The code, data formats and generator settings are in "
     "the repository and its model card; the human-input datasets are fetched from their "
     "publishers under their licences and are not redistributed.")

# ---------------------------------------------------------------- VI
h1("VI", "Results")
h2("A", "Detection in Simulation")
body("On Corpus A the full system detected "
     f"{BASEJ['headline']['detect_k']}/{BASEJ['headline']['n_attack']} attack sessions "
     f"({pct(det_a)}) with no false positive in {BASEJ['headline']['n_benign']} benign test "
     f"sessions, against {pct(s1_a)} without the LSTM. That result did not survive "
     "realism: Corpus A’s benign devices never go idle, and once they did, the same model "
     f"flagged {pct(orig_fp_b)} of Corpus B’s benign sessions. Retrained on Corpus B, the "
     f"model at its original per-window threshold detects {kn(cb['detect_k'], cb['n_attack'])} "
     f"of Corpus C’s attacks with {cb['fp_k']} false positives in {cb['n_benign']:,} benign "
     f"sessions; at the deployed, calibrated threshold it detects "
     f"{kn(ca['detect_k'], ca['n_attack'])} with {ca['fp_k']} ({tref('simdetect')}).")
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
body(f"Four families are detected every time at either threshold. The cost of calibration "
     f"falls on one family, the replay loop, which drops from {pct(cb['per_profile']['mimicry'], 0)} "
     f"to {pct(ca['per_profile']['mimicry'], 0)}; the single human-paced payload is not "
     "detected by anything. The capability gate alone reaches "
     f"{kn(H_['rule']['detect_k'], 600)} with no false positive, and adds nothing to the "
     "deployed rule on families the model was trained on. Three hand-written rules from our "
     f"earlier study reach the same {pct(R['all three rules']['mean'])}: in simulation, what "
     "learning adds over rules is the replay loop.")
h2("B", "From a Window Threshold to False Alarms per Device-Hour")
body("The threshold was first set at a per-window false-positive rate of 10⁻⁴ on "
     "validation windows. On sessions of up to an hour (about 101 device-hours) that model "
     f"raised {long_kbd:.3f} false alarms per keyboard-hour, about one per keyboard per working "
     "day, and none on other classes; short sessions had too few hours to show it. "
     f"{tref('operating')} explains why the per-window target says little. On 40 benign "
     f"keyboard-hours, {sfp_old['p_window_over']:.2e} of windows were over that threshold, ten "
     "times the target, because the pooled validation target hides the keyboard’s own rate. "
     f"An over-threshold window was followed by another {100 * sfp_old['q_next_over_given_over']:.0f}% "
     "of the time, since consecutive LSTM inputs share nine of ten rows. Treating windows as "
     f"independent predicts {sfp_old['naive_alarms_per_hour']:.1e} alarms per hour against "
     f"{sfp_old['observed_alarms_per_hour']:.3f} observed.")
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
body("We therefore choose the threshold on gate firings. Each session has a critical "
     "threshold, the largest at which its gate fires, so the whole curve of false alarms "
     "against threshold is a sort; a test checks this against the policy engine. On 80 "
     "hours of long benign sessions from a calibration seed, the smallest threshold with at "
     "most 0.01 alarms per device-hour was adopted, as pre-registered, and evaluated once on "
     f"a fresh seed: keyboard false alarms fell from {kb_b['rate']:.3f} to {kb_a['rate']:.3f} "
     "per hour, at the cost in Table "
     f"{ROMAN[TABLE_ORDER.index('simdetect') + 1]}. The full threshold curve is in the "
     "result file, so a deployment can trade the other way.")
h2("C", "Generalisation to Unseen Attack Families")
body("Stage 2 learns the families it is shown. Trained without a family and tested on it, "
     f"mean detection on Corpus C falls from {pct(seen_mean)} to {pct(lopo_m)}. To see why, we trained on "
     f"benign traffic and one family at a time and tested every family ({tref('transfer')}).")
fl = {"hid_injection": "hid", "staged_delay": "stag", "identity_switch": "iden",
      "class_exfil": "exfil", "mimicry": "mim", "mimicry_adaptive": "mim-a"}
rows = []
for F in FAMS:
    rows.append([fl[F]] + [f"{100 * TR['matrix'][F][G]['rate']:.0f}" for G in FAMS])
table("Detection (%) of each family (columns) by a model trained on one family (rows)",
      ["Trained on"] + [fl[g] for g in FAMS], rows, key="transfer")
st = TR["overlap"]["strong"]
body("The four injection-speed families transfer among themselves almost completely, and "
     "the two human-paced families transfer to and from nothing. We measured, for each "
     "family, the features that separate its attack windows from benign windows of the same "
     "class by a Kolmogorov–Smirnov statistic of at least 0.8. The overlap of those sets "
     f"predicts transfer (Spearman ρ = {st['spearman_jaccard']:+.2f}; a version using the top "
     f"five features, stated before the run, gave {TR['overlap']['top5']['spearman_jaccard']:+.2f} "
     "because many features tie at 1.00). The injection families share 9 to 11 such "
     "features, mostly rate and timing. The human-paced families have none: their best "
     "feature reaches 0.45. A supervised model can only learn weak, family-specific cues for "
     "them, and no other family carries those cues, so a new human-paced attack will not be "
     "caught by training on old ones.")
body("The capability gate needs no training family. Combined with the model under the same "
     "leave-one-family-out protocol, mean detection rises from "
     f"{pct(lopo_m)} to {pct(lopo_h)}: identity switch and covert exfiltration go from 0% "
     "to 100%, with no added false positive, because they inject at machine speed. The "
     "human-paced families stay where they were.")
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
body(f"The parser read {tb_lines:,} lines from four kernels with {tb_unparsed} unparsed "
     f"({tref('envs')}). The environments differ in a way simulation cannot show: Ubuntu’s "
     "and Fedora’s cloud kernels ship without the audio, network and serial class drivers, "
     "so the same emulated devices fall silent after enumeration there, and Ubuntu’s also "
     "lacks usbmon itself until an extra module package is installed.")
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
body(f"{tref('realkernel')} holds the main real-kernel result. The deployed rule raised no "
     f"alarm on any of {int(tb_tot['devices'])} benign device sessions over "
     f"{tb_tot['hours']:.1f} device-hours, including {int(kbd_role['devices'])} keyboards "
     f"driven by real people’s keystrokes, and detected all {fast_n} injection-speed attacks "
     f"with a median latency of {tb_lat_med:.1f} s. Those detections are the capability "
     f"gate’s. The learned model alone (†, competence checks on) caught {model_cal_all} of "
     f"{n_att_tb} attacks at its calibrated threshold. At the per-window threshold "
     f"(‡, every device judged) it caught all {fast_prev} injection attacks but also alarmed "
     f"on {int(kbd_role['previous_threshold_alarmed'])} of the {int(kbd_role['devices'])} "
     "human-typed keyboards. Real injection through a keyboard polled every 10 ms produces "
     "about 93 reports per second, not the simulator’s 390, and the model’s score reached "
     "the calibrated threshold too briefly to complete the gate. Nothing detected the "
     f"human-paced attacks ({mim_dep}/{mim_n}) except the original Corpus-A model, which "
     f"alarmed on {int(tb_tot['corpus_a_alarmed'])} of {int(tb_tot['devices'])} benign devices.")
body("The competence checks were needed. Judging every device, the same model alarmed on "
     f"{int(tb_tot['judging_all_alarmed'])} benign devices: a network adapter whose "
     "link-status endpoint sends an identical eight-byte report every 32 ms (perfectly "
     "regular and repetitive, which is what injection looks like to a model that has only "
     "seen keyboards), a serial converter, hubs, and the absolute tablet, whose six-byte "
     "reports no training device had. The HID check was designed after seeing the tablet "
     "alarm and validated on the sessions run afterwards.")
h2("E", "What Did Not Help")
rs = ROBF["summary"]
body("<i>Sample-size-robust features.</i> The long-session false alarms came from windows "
     "holding one keystroke, where the coefficient of variation and entropy of inter-arrival "
     "times are exactly zero, the signature of a script. Freezing either feature removed "
     "every such alarm in the trained model. Retraining with versions shrunk toward a "
     "Poisson prior, and with a novelty feature on a 60-second horizon, did not: pooled over "
     f"three seeds the candidate raised more keyboard alarms than the base model "
     f"({rs['R_both']['kbd_alarms_per_hour'][0]:.3f} against "
     f"{rs['base']['kbd_alarms_per_hour'][0]:.3f} per hour), so it was not adopted.")
pv2 = POLL["polling_v3"]["corpus_c_v2"]
body("<i>Polling-limited injection.</i> Training on injection bounded by a device polling "
     "interval of 1 to 10 ms, after the real-kernel miss, lowered Corpus C detection to "
     f"{pct(pv2['detect'])} (replay loop {pct(pv2['per_family']['mimicry'], 0)}) and caught "
     "almost none of the real-kernel attacks, failing two of its four criteria. The "
     "sim-to-real gap is not only speed.")
body("<i>Earlier negative results.</i> The per-device baseline contributes nothing measurable, "
     "even when devices are made heterogeneous; 13 alternative alerting rules, a long-repeat "
     "feature, longer context and a five-seed ensemble all fall within seed noise; a "
     "host-level feature for concurrent mouse activity did not help, because simulated "
     "devices on one host are independent. Two real defects surfaced only on real kernels: "
     "the usbmon timestamp wraps at 4,096 s and the parser had assumed 2³² µs, shifting "
     "every event after a wrap 199 s into the future, and the descriptor scanner had marked "
     "every real interface as late-arriving.")
h2("F", "Cost and Overhead")
body(f"The streaming detector costs {DIAG['cost']['ms_per_device_window']:.1f} ms per "
     f"device-window, about {DIAG['cost']['devices_per_core']} devices per core. An eBPF probe "
     "on the URB completion hook counts exactly the same completions as usbmon "
     f"({EBPF_BUSY['ebpf']['completions_at_giveback_hook']:,} of "
     f"{EBPF_BUSY['usbmon']['completions_and_errors']:,} under load), but reads the URB status "
     "too early to be correct, emits no records and is not a capture backend.")
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
    body(f"{tref('overhead')} measures each monitor on a guest whose storage stick is driven "
         f"with {OVH['bs_kib']} KiB direct I/Os, so that every few kilobytes costs three URBs, "
         "the worst case for a per-URB monitor. Each condition is compared with the "
         "unmonitored run of the same repetition, because a first version of the benchmark "
         "drifted by more than any difference between conditions. No monitor’s cost is "
         f"detectable. {n_contain} of {n_ci} intervals contain zero; "
         + (f"the {'other' if n_ci - n_contain == 1 else 'others'} lie on the side where "
            "monitoring would make transfers faster, which no monitor can cause, and with "
            f"{n_ci} intervals at 95% about one miss is expected. " if n_contain < n_ci else "")
         + "The resolution is coarse, ±10 to 30%, "
         "because the device is emulated, so the honest statement is an upper bound, not a "
         "zero. The kernel’s own BPF statistics put the probe at most "
         f"{ns_max / 1000:.1f} µs per URB completion in this guest, the same for counting and "
         "per-event output, which suggests the measurement’s own timing dominates; at a "
         "keyboard’s ten or so URBs a second that is negligible.")

# ---------------------------------------------------------------- VII
h1("VII", "What the Evidence Supports")
body(f"{tref('claims')} separates what was proposed, what was built and what the "
     "experiments support. The four novelty claims of the original proposal are N1 to N4.")
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
body("Three readings follow. First, the strongest result is operational: a monitor that "
     "raised no false alarm on 114 real-kernel device sessions and caught every "
     "injection-speed attack, with the detections coming from a physical bound and the "
     "learned model contributing its calibrated restraint rather than its detections. "
     "Second, the learned model’s advantage is narrow even in simulation, the replay loop, "
     "and it did not carry over to real kernel traffic, so this paper does not present "
     "IUDBMS as a detector of unknown attacks. Third, a human-paced payload typed once is "
     "statistically the same process as a person typing within one device’s traffic; no "
     "feature we have separates it, and no detector here catches it.")

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

missing = [k for k in REFS if k not in REF_NO]
assert not missing, f"references never cited: {missing}"
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
cp.keywords = "BadUSB, behavioural anomaly detection, false alarms per device-hour, LSTM, USB security"

DEST.parent.mkdir(parents=True, exist_ok=True)
doc.save(DEST)
print(f"  wrote {DEST}")
print(f"  tables: {_TABLE_N[0]}   figure: {FIG.name}")
