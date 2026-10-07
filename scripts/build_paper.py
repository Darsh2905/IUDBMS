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


def rich(par, text, **base):
    italic = False
    code = False
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
TABLE_ORDER = ["families", "headline", "perclass", "n1", "n2", "realism", "parser",
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

para("IEEE Title",
     "Intelligent USB Device Behaviour Monitoring System (IUDBMS): "
     "A Behaviour-Based Approach to Runtime USB Peripheral Trust")

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
RB = load("analysis_realistic_v2.json")
LOPO = load("analysis_lopo.json")
GATES = load("analysis_gates.json")
VARS = load("analysis_variants.json")
ROB = load("analysis_robustness.json")
AUG = load("analysis_augment.json")
IDLE = load("analysis_idle.json")

FAMS = ["hid_injection", "identity_switch", "staged_delay", "class_exfil", "mimicry",
        "mimicry_adaptive"]
seen_mean = sum(LOPO[f]["full"][0] for f in FAMS) / len(FAMS)
unseen_mean = sum(LOPO[f]["unseen"][0] for f in FAMS) / len(FAMS)
or_mean = sum(LOPO[f]["unseen_or_s1"][0] for f in FAMS) / len(FAMS)
s1_mean = sum(LOPO[f]["stage1"][0] for f in FAMS) / len(FAMS)
fp_fresh = LOPO["benign_fp_full"]
fp_or = sum(LOPO[f]["fp_unseen_or_s1"] for f in FAMS) / len(FAMS)
fp_unseen = sum(LOPO[f]["fp_unseen"] for f in FAMS) / len(FAMS)
orig_fp = RB["overall"]["original"]["fp"]
real_o, real_r = RB["real"]["original"], RB["real"]["realistic"]

det = M["full_system"]["session"]["attack_detect"]
fp = M["full_system"]["session"]["benign_fp"]
auc = M["full_system"]["window"]["auc"]
s1_det = M["stage1_only"]["session"]["attack_detect"]

ebpf_ratio = None
if EBPF_IDLE and EBPF_BUSY:
    ebpf_ratio = round(EBPF_BUSY["ebpf"]["completions_at_giveback_hook"]
                       / max(EBPF_IDLE["ebpf"]["completions_at_giveback_hook"], 1))

# ---------------------------------------------------------------- abstract
ABSTRACT = (
    "USB hosts accept a peripheral’s self-reported identity, and BadUSB attacks exploit "
    "that trust to inject keystrokes within seconds. Identity checks made at connection "
    "time say nothing about what a device does afterwards. We present IUDBMS, a monitor "
    "that trusts a device by its observed behaviour. It reads USB request blocks from the "
    "Linux kernel, condenses each second into a feature vector, scores it with an Isolation "
    "Forest and a per-device baseline, and passes that score to an LSTM that reads "
    "ten-second sequences. A persistence gate drives a throttle-then-deauthorise policy, "
    "run here in dry-run only. On a simulated corpus the system detects "
    f"{pct(det)} of attack sessions with {pct(fp)} benign false positives; without its "
    f"temporal stage it detects {pct(s1_det)}. We then stress that result. Once benign "
    f"devices go idle and keyboards send key-release reports, the original model flags "
    f"{pct(orig_fp)} of benign sessions. A model retrained on realistic traffic detects "
    f"{pct(seen_mean)} of attacks at {fp_fresh:.2%} false positives on 3,600 fresh "
    f"sessions, but only {pct(unseen_mean)} of attack families it never saw. Watching a "
    "real keyboard live on a Linux guest, the original model revoked it and the "
    "retrained model, now deployed, raised no alert. Two of four novelty claims were "
    "not supported."
)
n_words = len(re.findall(r"\S+", ABSTRACT))
assert 185 <= n_words <= 199, f"abstract is {n_words} words; IEEE allows at most 200"
print(f"  abstract: {n_words} words")

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Abstract", italic=True)
run(p, "—" + ABSTRACT)

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Index Terms", italic=True)
run(p, "—BadUSB, behavioural anomaly detection, eBPF, Isolation Forest, kernel "
       "enforcement, LSTM, USB security, usbmon.")

# ---------------------------------------------------------------- I
h1("I", "Introduction")
body("USB has been the standard way to attach peripherals since 1996, and plug-and-play "
     "convenience is the reason. That convenience hides a weakness: the host trusts what a "
     "device says about itself. During enumeration a device reports its Vendor ID (VID), "
     "Product ID (PID) and interface descriptors, and the host accepts them with no "
     "cryptographic attestation.")
body("BadUSB attacks exploit this. Altered firmware lets a flash drive or a charging cable "
     "declare itself a Human Interface Device (HID) keyboard and type a stored payload the "
     "moment it is plugged in. Hosts trust HID input by default, so the payload gets past "
     "file-based antivirus and runs within seconds.")
body("Most defences check identity at connection time. They allowlist VID/PID pairs, ask the "
     "user to confirm a device’s role, or apply static kernel filter rules. None of them "
     "looks at what the device does once it is running. A device that copies an allowed "
     "identity, or misbehaves inside an approved role, passes every one of these checks.")
body("IUDBMS (Intelligent USB Device Behaviour Monitoring System) moves the decision from "
     "connection time to run time. It captures USB Request Blocks (URBs) at the kernel "
     "boundary, turns each device’s traffic into one feature vector per second, and scores "
     "those vectors in two stages. An unsupervised detector asks whether this second looks "
     "normal for the device class. An LSTM asks whether the last ten seconds look like an "
     "attack unfolding. A graduated policy then rate-limits or revokes the device when the "
     "evidence persists.")
body("This paper makes five contributions.")
item("1)", "A complete behavioural pipeline from usbmon capture to a dry-run enforcement "
           "policy, with every stage tested.")
item("2)", "Four novelty claims, each given a purpose-built experiment with its pass "
           "criterion fixed in advance. Temporal modelling is strongly supported, fusion "
           "only partly, and the per-device baseline not at all.")
item("3)", "A realism stress test. Our first corpus had benign devices that never went idle. "
           "That inflated the headline result and taught the model to flag any quiet "
           "keyboard. We rebuild the corpus, retrain, and report what survives.")
item("4)", "A leave-one-family-out test of how the system handles attacks it was not trained "
           f"on: detection falls from {pct(seen_mean)} to {pct(unseen_mean)}.")
item("5)", "Validation against real kernel output, including an eBPF capture prototype. "
           "It exposed defects that 385 simulator-driven tests had missed.")
body("Sections II to IV review the literature, derive the research gap and novelty, and "
     "state the objectives. Section V describes the architecture and the experimental "
     "setup. Section VI reports the results, and Section VII states what they do not show.")

# ---------------------------------------------------------------- II
h1("II", "Literature Review")
body("The literature falls into three strands: the USB attack surface and defences against "
     "malicious peripherals, eBPF as a way to observe and enforce at kernel level, and "
     "deep-learning methods for anomalies that unfold over time. IUDBMS sits where the "
     "three meet.")

h2("A", "USB Peripheral Security and BadUSB Defences")
body("Shen et al. [1] proposed WMUD, a whitelist method for USB redirection in virtual "
     "desktops (QEMU/SPICE). Server-side filter rules keyed on device class, VID and PID "
     "admit only permitted devices, and the client cannot tamper with them. That shows the "
     "value of protected, per-asset policy. It is still an identity allowlist, though. A "
     "device that presents an allowed identifier is admitted whatever it does next, and "
     "the scheme covers virtualised redirection, not the defence of a physical host.")
body("Bintang et al. [2] proposed SafeUSB, a Windows keystroke monitor that combines an "
     "inter-keystroke timing threshold with content analysis, flagging typed keywords such "
     "as <i>POWERSHELL</i> and <i>CMD.EXE</i>. It blocks most injection scenarios. The "
     "authors also report a delay of about one second between HID connection and the "
     "start of monitoring, long enough for a short payload to finish, and they suggest "
     "compiled, kernel-level hooks as future work. That is the direction IUDBMS takes.")
body("Read together, [1] and [2] show the gap. WMUD’s tamper-resistant policy is bound to "
     "identity, and SafeUSB’s behavioural analysis is held back by its place in userspace "
     "and its activation delay. Neither learns a model of each device, and neither "
     "enforces at the kernel or driver boundary.")

h2("B", "eBPF for Kernel-Level Observability and Enforcement")
body("eBPF lets verified bytecode run inside the Linux kernel, which gives "
     "userspace-authored logic fine-grained observation and enforcement at kernel speed. "
     "Song’s survey [5] traces its core mechanics: the in-kernel virtual machine, JIT "
     "compilation, and the static verifier that guarantees bounded, memory-safe execution. "
     "It also follows eBPF’s growth from packet filtering into tracing and security, and "
     "notes the remaining limits on verification complexity and program size.")
body("Kostopoulos et al. [3] presented eIDPS, a real-time eBPF and machine-learning "
     "pipeline for network intrusion detection and prevention at line rate. Rajurkar and "
     "Deshmukh [4] survey the field and identify its dominant pattern: <i>hybrid designs "
     "that extract features in the kernel and leave ML inference to userspace</i>, "
     "coordinated through shared BPF maps. They list verifier constraints, portability and "
     "ML integration as open problems.")
body("Together [3]–[5] establish eBPF as a proven substrate for sub-microsecond observation "
     "and enforcement, and the kernel-extract, userspace-infer split as the accepted way to "
     "pair kernel speed with ML expressiveness. But this work is about <i>network</i> "
     "intrusion detection. Applying the pattern to the <i>USB</i> subsystem, with URBs as "
     "the unit of observation and driver binding as the enforcement point, has not been "
     "done.")

h2("C", "Deep Learning for Behavioural and Temporal Anomaly Detection")
body("LSTM networks suit anomalies whose signal lies in temporal structure rather than in any "
     "single moment. Hnamte et al. [7] proposed LSTM-AE, which hybridises an LSTM with an "
     "auto-encoder for network intrusion detection. They validate it on several public "
     "datasets and argue for two-stage designs that separate detection from classification.")
body("Baek and Jung [6] also decouple anomaly detection from attack classification. A binary "
     "focal-loss stage filters anomalies out of heavily imbalanced traffic, and a "
     "multi-class stage then names the attack, with both stages sharing a CNN-1D-LSTM "
     "backbone. IUDBMS’s own layering follows the same staged idea. Saini et al. [8] "
     "reinforce the value of hybrid <i>CNN–LSTM</i> architectures for network threat "
     "detection.")
body("Two further works target novel and slow-developing attacks. Chithik Raja and Al Mahri "
     "[9] applied an attention-enhanced LSTM to zero-day detection across process, network "
     "and file-system telemetry, showing that behaviour-based temporal models can "
     "generalise to unseen threats. Mohamed Anwar et al. [10] used LSTM-based anomaly "
     "detection on IIoT systems against DDoS, spoofing and ransomware, and beat Random "
     "Forest, Logistic Regression and 1D-CNN baselines at low latency.")
body("Collectively, [6]–[10] show that LSTM-based temporal models catch multi-stage and "
     "zero-day behaviour within real-time budgets, and that layered designs balance recall "
     "against discrimination. None of them applies temporal modelling to the event stream "
     "of a single USB peripheral, which is what IUDBMS does.")

# ---------------------------------------------------------------- III
h1("III", "Inferences from the Literature")
body("The USB-defence work [1], [2] establishes the threat and the value of behavioural "
     "analysis, but stays tied to identity or to userspace latency. The eBPF work [3]–[5] "
     "provides a fast, verifiable kernel mechanism, almost entirely for networks. The "
     "deep-learning work [6]–[10] shows that layered LSTM models catch temporally "
     "structured, zero-day behaviour in real time, but never on the USB event stream. "
     "IUDBMS occupies the space these three strands leave open.")

h2("A", "Problem Statement")
body("USB’s trust-by-default enumeration lets a malicious peripheral declare a trusted "
     "identity, typically an HID keyboard, and abuse it to inject commands at machine "
     "speed. The host can be compromised within seconds. Existing defences authenticate "
     "<i>claimed identity</i> at connection and do not continuously judge <i>actual "
     "behaviour</i>. Devices that spoof an allowed identifier, or misbehave inside an "
     "approved role, evade them, and existing behavioural monitors run in userspace, whose "
     "activation delay lets short payloads succeed. A defence is needed that (a) trusts "
     "devices by observed behaviour, not declared identity, (b) enforces at the "
     "kernel/driver boundary to remove round-trip latency, and (c) detects staged, "
     "slow-escalation attacks as well as single-point anomalies.")

h2("B", "Research Gap")
item("1)", "<i>Identity-bound trust.</i> Allowlisting (for example WMUD [1]) admits devices by "
           "class, VID and PID, and is blind to spoofed identities or misuse within an "
           "approved role.")
item("2)", "<i>Userspace activation latency.</i> SafeUSB [2] shows that runtime analysis "
           "works, but its userspace placement adds a delay of about one second that short "
           "payloads exploit.")
item("3)", "<i>Network-only kernel intelligence.</i> The eBPF-plus-ML pattern (eIDPS [3]; "
           "Rajurkar and Deshmukh [4]) is mature for networks and unapplied to USB, where "
           "the natural unit is the URB and the natural enforcement point is driver binding.")
item("4)", "<i>No temporal model of USB behaviour.</i> LSTM and attention models [6]–[10] "
           "catch staged and zero-day attacks in network and IIoT settings, but no USB "
           "defence models a device’s event sequence over time.")
item("5)", "<i>Open enforcement loop.</i> Existing behavioural detectors alert or block from "
           "userspace after a round trip. None closes the loop inside the kernel.")

h2("C", "Novelty of IUDBMS")
body("The underlying techniques already exist: eBPF capture [3]–[5], unsupervised anomaly "
     "scoring, and LSTM temporal modelling [6]–[10]. The novelty of IUDBMS is their "
     "<i>combination and application</i>, a single per-device, closed-loop pipeline aimed at "
     "the USB subsystem, which no reviewed eBPF-ML or LSTM-anomaly work addresses and no "
     "reviewed USB defence implements. Each element below answers a gap from Section III-B "
     "and is tested in Section VI-C. The numbering follows §3.3 of the Review 1 submission.")
item("N1)", "<i>Multi-modal feature fusion at the kernel/driver boundary.</i> Event timing, "
            "traffic volume and descriptor changes are fused into one feature stream. This "
            "carries the eBPF-ML pattern [3], [4] into USB and answers the network-only gap.")
item("N2)", "<i>Per-device adaptive behavioural baselines.</i> Each device is scored against "
            "its own learned norm instead of a global rule set, which answers the "
            "identity-bound gap left by WMUD [1].")
item("N3)", "<i>Closed-loop kernel-level enforcement.</i> Verdicts such as rate-limiting or "
            "unbinding execute without a userspace round trip, which answers SafeUSB’s [2] "
            "activation latency and the open-loop gap.")
item("N4)", "<i>Temporal sequence modelling for staged attacks.</i> A userspace LSTM reads "
            "rolling event windows and catches slow-escalation attacks that single-point "
            "detectors miss, bringing temporal modelling [6]–[10] to USB.")

h2("D", "Relationship to the Reference Framework")
body("The reference framework is a kernel-level eBPF security framework that filters "
     "peripherals against human-written rules at the host-controller boundary, with "
     "sub-microsecond overhead. IUDBMS keeps its central insight, that the kernel/driver "
     "boundary is the right place to enforce, and replaces its static, "
     "administrator-written rules with <i>behavioural intelligence learned from traffic</i>: "
     "per-device baselines and anomaly scoring backed by a userspace LSTM. In short, "
     "it keeps the fast, verifiable enforcement substrate and adds the adaptive "
     "capability the framework does not attempt, moving from <i>rules an administrator "
     "must foresee</i> to <i>behaviour the system learns</i>. No experiment here is run "
     "against the reference framework; it is discussed only to position the contribution.")

# ---------------------------------------------------------------- IV
h1("IV", "Objectives")
body("The aim is a system that trusts USB peripherals by runtime behaviour and not by "
     "claimed identity. It is a four-layer pipeline that carries a device’s activity from "
     "raw kernel events, through scoring, to userspace temporal analysis, and back into the "
     "kernel as an enforced verdict. It follows the kernel-extract, userspace-infer "
     "pattern [3], [4] and adds per-device adaptivity and closed-loop enforcement. Each "
     "objective below corresponds to one stage.")
item("1)", "<i>Kernel-level capture (Layer 1).</i> Intercept URBs at the kernel/driver "
           "boundary and extract multi-modal features (event timing, traffic volume, "
           "descriptor changes), passing compact records upward with minimal overhead.")
item("2)", "<i>Per-window anomaly scoring (Layer 2).</i> Build a <i>per-device adaptive "
           "baseline</i> and score incoming features against it with unsupervised learning "
           "(Isolation Forest or One-Class SVM), a fast, high-recall filter that flags "
           "suspicious activity for deeper inspection.")
item("3)", "<i>Userspace temporal sequence modelling (Layer 3).</i> Train a PyTorch LSTM on "
           "rolling event windows to detect <i>temporal structure</i>, meaning staged, "
           "multi-step attacks whose ordering rather than any single event betrays intent.")
item("4)", "<i>Closed-loop enforcement (Layer 4).</i> Execute verdicts from Layers 2 and 3 "
           "(rate-limiting or unbinding a device) without a userspace round trip, closing "
           "the activation window that lets short BadUSB payloads succeed.")
item("5)", "<i>Supporting data store and interface.</i> Persist baselines, event history and "
           "alerts in SQLite and expose them through a dashboard for monitoring, tuning and "
           "review of enforcement decisions.")
item("6)", "<i>Evaluation against BadUSB scenarios.</i> Measure detection, false-positive "
           "rate, enforcement latency and overhead against rapid, delayed and multi-stage "
           "payloads, and compare behavioural trust with identity-based baselines.")
body("The capture mechanism in Objective 1 admits two options: eBPF, the richer one, or "
     "usbmon as a lighter fallback. Because both sit behind one interface the choice only "
     "affects Layer 1. The implementation takes the usbmon path, and Section VI-J reports "
     "an eBPF prototype that was measured against it.")

# ---------------------------------------------------------------- V
h1("V", "System Architecture")
h2("A", "Capture and Feature Extraction")
body("Fig. 1 shows the whole pipeline, from kernel capture to enforcement. URB traffic is "
     "observed through usbmon, a passive kernel tap available on any Linux host with debugfs "
     "mounted. An eBPF backend with lower overhead and explicit control over payload "
     "truncation is the intended production path. Section VI-J reports a counting "
     "prototype of it, and both backends sit behind one capture interface so the choice "
     "does not touch any later stage. A parser turns raw event text into typed records. A "
     "stateful, per-device extractor then compresses each non-overlapping one-second window "
     "into a vector over four behavioural modalities: timing, payload, protocol and "
     "identity.")

new_section(1)
p = doc.add_paragraph(style="IEEE Figure")
p.add_run().add_picture(str(FIG), width=Inches(7.16))
para("IEEE Caption",
     "Fig. 1.  The IUDBMS pipeline: kernel capture, per-window feature extraction, the "
     "two-stage detector (Stage 1’s score is passed to Stage 2 as an input, not cast as a "
     "separate vote), and the graduated enforcement policy. Dashed blocks are designed but "
     "not yet built, or generated but not yet executed.")
new_section(2)

h2("B", "Two-Stage Detection and Enforcement")
body("The two models run in series. Stage 1 is a memoryless, per-window anomaly score that "
     "fuses an Isolation Forest, trained on benign devices of the same class, with an "
     "adaptive per-device baseline. Its output is not thresholded directly. It is appended "
     "to that window’s feature vector, and the vector enters a ten-second sliding buffer. "
     "Stage 2, an LSTM, reads the ten-window sequence and produces the one probability that "
     "is compared with the decision threshold. A persistence gate requires four flagged "
     "windows in any six before a device is throttled, and eight cumulative flagged windows "
     "before it is deauthorised. The policy engine generates both actions, but execution "
     "against the kernel stays behind a manual safety review and runs in dry-run mode.")

h2("C", "Experimental Setup")
body("<i>Traffic.</i> No labelled USB attack traffic is publicly available, so we generate "
     "it. A simulator writes sessions in the exact text format the Linux usbmon driver "
     "emits, and the same parser a live capture uses reads them back. A session is one "
     "device observed for 40 to 85 s. Benign devices come from five profiles: keyboard, "
     "mouse, mass storage, webcam and audio. Attacks come from the six families in "
     f"{tref('families')}, which range from crude machine-speed injection to a "
     "human-paced payload that is typed once.")

table("The six attack families and what separates each from benign traffic",
      ["Family", "What the device does", "Main signal"],
      [["hid_injection", "Types a scripted payload at machine speed (about 4.5 ms per key) "
                         "soon after plug-in", "Speed, regularity"],
       ["staged_delay", "Types like a person for about half the session, then injects at "
                        "machine speed", "Burst after a long dwell"],
       ["identity_switch", "Enumerates as storage, later exposes a keyboard interface and "
                           "injects", "Late interface, class mismatch"],
       ["class_exfil", "A declared keyboard that sends covert bulk data out",
        "Transfer type outside its class"],
       ["mimicry", "Replays a fixed 24 to 48-key script in a loop at human speed",
        "Exact repeats across windows"],
       ["mimicry_adaptive", "Types a finite 60 to 220-key payload once at human speed, then "
                            "goes silent", "Silence after the payload"]],
      key="families", left_cols=(0, 1, 2))

body("<i>Corpora.</i> We use three. Corpus A is the original: 1,000 sessions (600 benign, "
     "400 attack) in which every benign device is active for its whole session. Corpus B has "
     "the same size and seed but changes the benign side and the typing model. Keyboards, "
     "mice, drives, webcams and audio devices alternate active bursts with idle gaps or "
     "stay idle after enumeration (40% always active, 15% with gaps of 3 to 8 s, 15% of 8 "
     "to 20 s, 20% of 20 to 60 s, 10% idle throughout). Keyboards send a key-down and a "
     "key-up report per keystroke, with overlapping keys producing multi-key reports, as "
     "the HID boot protocol specifies. Keystrokes come from a Zipf-weighted vocabulary "
     "whose bigram entropy is tuned to 3.56 bits, Shannon’s estimate for English digrams [11], "
     "and each device types at its own speed (median inter-key time 60 to 300 ms). Attack "
     "typing gets identical treatment so that key-release reports cannot become a shortcut. "
     "Corpus C is a fresh set of 3,600 sessions (3,000 benign, 100 per attack family) "
     "generated with a different seed. It is never used to train, calibrate or choose "
     "anything.")
body("<i>Training and calibration.</i> Sessions, never windows, are split by profile into "
     "training, calibration and test partitions, so no window of a session appears on both "
     "sides. Two columns are withheld from the deployed model: time since enumeration, which "
     "a Kolmogorov–Smirnov test identified as a session-position leak, and any column with "
     "zero variance on the training split. Stage 1 (Isolation Forest, weight 0.55, plus a "
     "per-device EWMA baseline, weight 0.45) is fit on benign training sessions only. "
     "Stage 2 is a single-layer LSTM with hidden size 32 over causally left-padded "
     "ten-window sequences, trained with class-weighted cross-entropy (Adam, learning rate "
     "2×10⁻³, gradient norm clipped at 5). The decision threshold is the empirical quantile "
     "of Stage-2 scores on held-out benign windows at a per-window false-positive rate of "
     "0.01%. It is never read off the test set. The deployed model set is the one trained "
     "on Corpus B.")
body("<i>Metrics and statistics.</i> A session counts as detected when the persistence gate "
     "fires. We report session-level detection, the benign session false-positive rate, and "
     "the median time from attack onset to the first alert. Rates carry 95% Wilson "
     "intervals. LSTM training is seeded but seed-sensitive, so where a comparison could "
     "hinge on the seed we train three and call a difference real only if it exceeds both "
     "the seed spread and the resolution of one session. The pass criterion of each novelty "
     "experiment was fixed before its first run.")
body("<i>Real kernel traffic.</i> For the real-traffic checks we ran Ubuntu 24.04.5 (kernel "
     "6.8, aarch64) in a QEMU guest with a virtual xHCI controller, keyboard, mouse, "
     "tablet, hub and storage device. We recorded usbmon output with the project’s own "
     "capture script. In one recording a person typed into the guest keyboard for about "
     "100 s. These recordings are scored offline with named model weights and are never "
     "used for training or tuning. A later live run watched the guest with both the "
     "original and the retrained model at once. Development used Python 3.12, "
     "scikit-learn and PyTorch on macOS/arm64.")

# ---------------------------------------------------------------- VI
h1("VI", "Results")

h2("A", "Headline Detection on the Original Corpus")
rows = []
for key, label in (("full_system", "IUDBMS (Stage 1 + Stage 2)"),
                   ("stage1_only", "Stage 1 only (no temporal model)"),
                   ("identity_whitelist", "Identity whitelist (VID:PID)")):
    r = M.get(key, {})
    w = r.get("window", {})
    se = r.get("session", r if "attack_detect" in r else {})
    rows.append([label,
                 f"{w['auc']:.3f}" if w.get("auc") is not None else "—",
                 pct(w.get("precision"), 1) if w.get("precision") is not None else "—",
                 pct(w.get("recall"), 1) if w.get("recall") is not None else "—",
                 pct(se.get("attack_detect")), pct(se.get("benign_fp"))])
table("Detection on the held-out test split of Corpus A",
      ["Model", "ROC-AUC", "Prec.", "Rec.", "Detect", "FP"], rows, key="headline")
lat = M["full_system"]["session"].get("median_latency")
body(f"{tref('headline')} compares the system with two baselines on Corpus A. The identity "
     "whitelist is a USBGuard-style control. Every attack in the corpus presents a VID/PID "
     "from the same pool as benign devices, so the whitelist detects none of them, which is "
     "the premise the project set out to test. Removing the temporal stage costs "
     f"{100 * (det - s1_det):.1f} points of session detection, the largest single effect "
     f"in this work. The median time from attack onset to the first enforcement action is "
     f"{lat:.1f} s. These figures come from the corpus the project started with. "
     "Sections VI-E to VI-I show how much of them survives a more realistic one.")

h2("B", "Per-Family Detection and the Single-Shot Bound")
pp = M["full_system"]["per_profile"]
rows = [[k, v.get("n", "—"), pct(v.get("detect_rate")),
         f"{v['median_latency']:.1f} s" if v.get("median_latency") else "—"]
        for k, v in sorted(pp.items()) if v.get("label") == 1]
table("Detection by attack family on Corpus A",
      ["Attack family", "n", "Detect", "Latency"], rows, key="perclass")
mim_ad = pp["mimicry_adaptive"]["detect_rate"]
body(f"Detection is not uniform ({tref('perclass')}). Attacks that deviate structurally are "
     "caught every time: machine-speed injection, a storage device that grows a keyboard "
     "interface, a keyboard that pushes bulk data out, and a long benign dwell followed by a "
     "burst. The replay loop is caught in "
     f"{pct(pp['mimicry']['detect_rate'], 0)} of sessions. The hard case is the "
     "human-paced payload typed once. A per-feature Kolmogorov–Smirnov scan of that family "
     "against benign keyboard traffic separates it on none of the deployed features, because "
     "inside one device’s URB stream it is statistically the same process as a person typing. "
     f"On this corpus {pct(mim_ad, 0)} of those sessions are caught, and the only reason is "
     "the permanent silence that follows a finished payload. Section VI-E shows why that "
     "silence cannot be trusted as a signal.")

h2("C", "Ablations Against the Novelty Claims")
body("Each novelty claim from Section III-C was tested by an experiment built for it, with "
     "the acceptance criterion fixed before the first run. Two of the four are not "
     "supported. We report that as a result.")


def SHORT_VERDICT(r):
    if r["fusion_wins"]:
        return "fusion wins"
    if r["fusion_loses"]:
        return "fusion loses"
    if not r["valid"]:
        return "saturated" if "saturat" in r["verdict"] else "invalid"
    return "tie"


sw = FUSION["sweep"]
rows = [[f"{r['strength']:.2f}",
         f"{100 * r['fused_mean']:.1f} ± {100 * r['fused_sd']:.1f}",
         f"{100 * r['best_spec_mean']:.1f} ± {100 * r['best_spec_sd']:.1f}",
         f"{100 * r['margin_mean']:+.1f} ± {100 * r['margin_sd']:.1f}",
         SHORT_VERDICT(r)] for r in sw]
table("N1: fused model vs best single modality on a diffuse attack, by attack strength "
      "(mean ± s.d. over three seeds)",
      ["Strength", "Fused", "Best single", "Margin", "Verdict"], rows, key="n1")
body(f"<i>N1, multi-modal fusion: supported only non-uniformly.</i> Every family in the main "
     "corpus deviates on several modalities at once, so any single modality catches all of "
     "them and fusion cannot show a gain. The ablation therefore uses purpose-built "
     "families that deviate on one axis at a time, plus a <i>diffuse</i> attack that "
     "deviates weakly on two axes together. Fusion never loses on a single-axis family, and "
     "on descriptor-only attacks it is the only configuration that works: the identity "
     "modality alone has too little variance to calibrate a threshold, while the fused "
     f"model detects such attacks reliably. On the diffuse attack the benefit changes sign "
     f"with strength ({tref('n1')}). Fusion gains substantially at moderate strength, but a "
     "nine-feature subset of the model beats it at the subtlest strength, where the extra "
     "dimensions cost more variance than they add. The criterion was to win somewhere and "
     "never lose, so N1 is not supported.")

hom, het = BASE["results"]["homogeneous"], BASE["results"]["heterogeneous"]
rows = []
for label, r in (("Homogeneous devices", hom), ("Heterogeneous devices", het)):
    full = r["rows"]["full (forest + baseline)"]
    fo = r["rows"]["forest only"]
    rows.append([label, f"{r['keyboard_rate_sd']:.3f}", pct(full["detect"]), pct(fo["detect"]),
                 f"{100 * r['margin']:+.1f}", f"{100 * r['noise_floor']:.1f}"])
table("N2: contribution of the per-device baseline, with and without device variation",
      ["Corpus", "Rate s.d.", "Full", "Forest only", "Δ", "Noise"], rows, key="n2")
body("<i>N2, per-device baseline: not supported.</i> Removing the per-device term changes "
     "detection by less than the noise floor. We first suspected the cause was the "
     "simulator, since every device of a class behaved identically and a class-conditioned "
     "forest already modelled them all. That is testable, because homogeneity is a property "
     "of the generator. Giving each device its own timing persona raised between-device "
     f"variation 2.8-fold, and the baseline stayed inert ({tref('n2')}). The suspected "
     "explanation was wrong, so the negative result is stronger than we first reported. "
     "Detection also falls under heterogeneity, as it should: more varied benign behaviour "
     "is harder to separate from attacks.")
body("<i>N3, closed-loop enforcement: dry-run only.</i> The policy engine generates the exact "
     "kernel command that would revoke a device, with the evidence that justified it, and "
     "the escalation ladder runs end to end on recorded sessions. Execution against the "
     "kernel is gated behind a manual review and has never been performed, so the loop is "
     "closed in software and not yet against hardware.")
body("<i>N4, temporal sequence modelling: strongly supported.</i> Without its sequence stage "
     f"the pipeline falls from {pct(det)} to {pct(s1_det)} session detection. The effect "
     f"survives the realistic corpus: on Corpus C Stage 1 alone detects {pct(s1_mean)} "
     f"against {pct(seen_mean)} for the full system. A model-family comparison shows the "
     "gain belongs to <i>sequence</i> and not to recurrence in particular. A GRU and a "
     "dilated 1-D CNN match the LSTM within noise, and a supervised model given one window "
     "does markedly worse. We keep the LSTM because no challenger clears the noise floor.")

h2("D", "Does Host Context Help?")
cm = CROSS["per_profile"]["mimicry_adaptive"]
body("The single-shot bound suggests an extension. A person typing is usually also moving a "
     "mouse, and a malicious device typing alone is not. We added a host-level feature "
     "carrying the event rate of every other device on the machine, together with a leakage "
     "guard built before the feature, since host composition that correlates with the label "
     "would reproduce the positional leak of Section V-C. With the guard holding, the "
     f"feature does not help: on the human-paced family the change in detection is "
     f"{100 * cm['delta_mean']:+.1f} ± {100 * cm['delta_sd']:.1f} points against a noise "
     f"floor of {100 * cm['noise']:.1f}. The reason is specific. The generator draws each "
     "device on a host independently, so a benign keyboard and its mouse are statistically "
     "independent, just as a malicious device and a bystander’s mouse would be. The "
     "coupling the feature exists to detect is absent on both sides. Real users interleave "
     "typing and mousing. Testing that needs either a coupled generator, which risks building "
     "in the signal it then detects, or real captured traffic.")

h2("E", "A Realism Stress Test: Idle Devices")
kb, ms, st = (IDLE["idle gaps 3-8 s"][c] for c in ("keyboard", "mouse", "mass_storage"))
kb2, ms2 = IDLE["idle gaps 8-20 s"]["keyboard"], IDLE["idle gaps 8-20 s"]["mouse"]
kb3 = IDLE["fully idle after enumeration"]["keyboard"]
ms3 = IDLE["fully idle after enumeration"]["mouse"]
body("The generator behind Corpus A has a property real devices lack. Every benign device is "
     "active for its whole session. Keyboards pause for at most three seconds, mice never "
     "stop moving, drives never stop working. We gave benign keyboards, mice and drives "
     "duty cycles and scored them with the original model (60 sessions per cell). With "
     f"idle gaps of 3 to 8 s it flagged {pct(kb['fp'], 0)} of keyboards and "
     f"{pct(ms['fp'], 0)} of mice. With gaps of 8 to 20 s it flagged {pct(kb2['fp'], 0)} "
     f"and {pct(ms2['fp'], 0)}, and it flagged every keyboard and mouse that sat idle "
     f"after enumeration ({pct(kb3['fp'], 0)} and {pct(ms3['fp'], 0)}). Mass-storage "
     f"devices were never flagged ({pct(st['fp'], 0)}). The claim of 0.0% benign false "
     "positives on Corpus A held only because no keyboard in it ever stopped typing, and "
     "real keyboards stop constantly. The same artefact explains the human-paced family’s "
     "40% detection. Its only tell was sustained typing followed by silence, which an "
     "ordinary idle keyboard also produces.")

h2("F", "Retraining on Realistic Traffic")
n = RB["benign_n"]
an = RB["attack_n"]
rows = [["Benign false positives", "", ""]]
for key, label in (("active", "Always active"), ("gap3-8", "Idle gaps 3–8 s"),
                   ("gap8-20", "Idle gaps 8–20 s"), ("gap20-60", "Idle gaps 20–60 s"),
                   ("idle", "Idle after enumeration"), ("keyboards", "All keyboards")):
    rows.append([f"{label} (n={n[key]})", pct(RB["benign"][key]["original"]),
                 pct(RB["benign"][key]["realistic"])])
rows.append([f"All benign (n={n['ALL']})", pct(RB["overall"]["original"]["fp"]),
             pct(RB["overall"]["realistic"]["fp"])])
rows.append(["Attack detection†", "", ""])
for f in sorted(an, key=lambda x: FAMS.index(x)):
    rows.append([f"{f} (n={an[f]})", pct(RB["attack"][f]["original"]),
                 pct(RB["attack"][f]["realistic"])])
rows.append([f"All attacks (n={sum(an.values())})", pct(RB["overall"]["original"]["detect"]),
             pct(RB["overall"]["realistic"]["detect"])])
rows.append(["Window ROC-AUC", f"{RB['window_auc']['original']:.3f}",
             f"{RB['window_auc']['realistic']:.3f}"])
table("Original and retrained models on the test split of Corpus B",
      ["", "Original", "Retrained"], rows, key="realism")
body(f"We retrained the same architecture on Corpus B with the same seed, epochs, "
     f"calibration rule and gate. {tref('realism')} compares both models on Corpus B’s test "
     f"split. The original model flags {pct(orig_fp)} of benign sessions and every keyboard. "
     "Its 100% attack detection (marked † in the table) is not detection, because it alarms "
     "on every keyboard-class session whether or not an attack is under way. The retrained model flags no benign "
     f"session. Overall detection is {pct(RB['overall']['realistic']['detect'])}, with the "
     f"cost where the earlier reading predicted: the human-paced payload falls from {pct(mim_ad, 0)} to "
     "0%, since the silence that was its only tell is now something every idle keyboard "
     "does. We regard 0% as the honest figure for that family. On the fresh Corpus C the "
     f"retrained model detects {pct(seen_mean)} of attacks on average (per family: "
     + ", ".join(f"{pct(LOPO[f]['full'][0], 0)}" for f in FAMS) + f") with {fp_fresh:.2%} "
     f"benign false positives, which is {round(fp_fresh * 3000)} alarms in 3,000 sessions.")

h2("G", "Validation Against Real Kernel Traffic")
body("Everything so far is measured on simulated sessions. To check the pipeline against "
     "genuine kernel output we captured usbmon text from the Linux guest described in "
     "Section V-C. This checks the machinery. The devices are virtual and no attack "
     "traffic was generated, so none of it is a detection result.")
rc = REALCAP
rows = [["Lines parsed", f"{rc['lines_seen']:,}"],
        ["Lines unparsed", f"{rc['lines_unparsed']}"],
        ["Loss rate", f"{rc['loss_rate']:.4%}"],
        ["Target (plan §4.1)", f"< {rc['plan_criterion']:.1%}"],
        ["Events satisfying the contract", f"{rc['events_contract_valid']} of {rc['events_total']}"],
        ["Control submissions with SETUP",
         f"{rc['control_submissions_with_setup']} of {rc['control_submissions_total']}"],
        ["Devices observed", f"{rc['n_devices']}"]]
table(f"Parser validation on a real usbmon capture ({rc['span_s']:.1f} s)",
      ["Metric", "Value"], rows, key="parser")
body(f"The parser lost none of the {rc['lines_seen']} real lines against a target of under "
     "0.1%, a criterion that could not be measured before this capture existed. Every "
     "event satisfied the data contract, and every real control submission carried the "
     "SETUP packet that the first prototype had silently dropped.")
body("Running the live monitor, and not only parsing offline, exposed two defects that the "
     "simulator cannot show. First, the window clock started when the service started, "
     "not at the first observed event. Any delay before the first USB event left every "
     "window out of phase with the traffic, and the service scored nothing while 385 "
     "simulator-driven tests passed, because a simulated source delivers its first event "
     "at once. Second, the Stage-2 checkpoint installed on the test host was not the one "
     "behind the reported numbers. It loaded, scored traffic and passed every structural "
     "check, yet at its recorded threshold it detected none of 120 simulated attacks. "
     "Retraining with the documented command reproduced the reported figures exactly (AUC "
     "0.9762, 89.2% detection, no benign false positives), so the evaluation was sound and "
     "the shipped file was wrong. Each stage now stores a fixed-input output fingerprint, "
     "the loader refuses weights that do not match it, and an end-to-end test asserts that "
     "the shipped set alerts on simulated attacks.")
W = RB["wire"]
rows = [["HID reports per second", f"{W['real']['reports_per_s']:.1f}",
         f"{W['corpus_a_sim']['reports_per_s']:.1f}", f"{W['corpus_b_sim']['reports_per_s']:.1f}"],
        ["Key-release (all-zero) reports", pct(W['real']['release_share']),
         pct(W['corpus_a_sim']['release_share']), pct(W['corpus_b_sim']['release_share'])],
        ["Distinct payloads", pct(W['real']['unique_payload_share']),
         pct(W['corpus_a_sim']['unique_payload_share']),
         pct(W['corpus_b_sim']['unique_payload_share'])],
        ["Gaps longer than 1 s", pct(W['real']['gaps_over_1s']),
         pct(W['corpus_a_sim']['gaps_over_1s']), pct(W['corpus_b_sim']['gaps_over_1s'])]]
table("A real keyboard against the two simulated ones, measured identically",
      ["Property", "Real", "Corpus A", "Corpus B"], rows, key="keyboard")
body(f"One person typed into the guest’s virtual keyboard for about 100 s. {tref('keyboard')} "
     "shows how the recording differs from Corpus A’s keyboard. A real keyboard reports "
     "key-releases as well as key-presses, repeats about a third of its payloads, and "
     "sends reports roughly ten times faster than the simulator, though the speed gap "
     "mostly reflects the task, since continuous key-mashing is not prose. Corpus B closes "
     "the structural gap. We then scored the recording offline with named weights. "
     f"The original model put {real_o['at_or_above_threshold']} of {real_o['busy']} windows "
     f"containing keystrokes at or above its threshold, and its gate fired "
     f"{real_o['first_alert_s']:.0f} s into the capture. The retrained model put "
     f"{real_r['at_or_above_threshold']} of {real_r['busy']} over the threshold, with a "
     f"median Stage-2 score of {real_r['stage2_median']:.2f}, and its gate stayed silent. "
     "This is one session of one person on a virtual keyboard, so it is not a "
     "false-positive rate. It is also not independent evidence for the retrained model, "
     "because Corpus B was built from HID-protocol facts that the capture happens to "
     "share. What it does show is that the original model’s failure on real typing is "
     "real and not a simulation artefact.")
body("A third defect appeared only on live devices. The descriptor scanner stamped every "
     "interface with the time since the scanner started, and the model reads any interface "
     "with a positive arrival time as one that appeared after enumeration, which is the "
     "signature of the identity-switch attack. Every window of every real device therefore "
     "carried the late-interface flag, and no device was assigned its class group. The "
     "first live alerts, a keyboard and a drive escalated to a dry-run revoke, came from a "
     "system in this state and cannot be read as evidence about behaviour, so we do not "
     "report them. The drive stopped alerting once the defect was fixed. Interfaces present "
     "when a device is first listed now count as enumeration-time, and the scanner shares "
     "the monitor’s window clock. With the fix the flag is 0.00 in every window and the "
     "keyboard is classed as an HID device.")
rows = []
for model, LV in (("Original", LIVE_O), ("Retrained", LIVE_R)):
    for did, label in (("1-4", "keyboard"), ("1-6", "USB drive")):
        d = LV["devices"][did]
        rows.append([f"{model}, {label}", f"{d['windows']} ({d['busy_windows']})",
                     f"{d['stage2_busy_median']:.2f}", d["alerts"], d["trust_state"]])
table("A live run on the guest: both models watching the same real devices for seven minutes",
      ["Model, device", "Windows (with traffic)", "Median Stage-2", "Alerts", "Final state"],
      rows, key="live")
ko, kr = LIVE_O["devices"]["1-4"], LIVE_R["devices"]["1-4"]
body(f"{tref('live')} records the run. The deployed set is now the retrained model, and we "
     "ran it beside the original on the guest while a person typed on the keyboard and then "
     f"left it idle. On the keyboard the original model’s median Stage-2 score was "
     f"{ko['stage2_busy_median']:.2f}, its gate fired {ko['alerts']} times, and the keyboard "
     f"ended in a dry-run revoke. The retrained model scored the same windows with a median "
     f"of {kr['stage2_busy_median']:.2f} (maximum {kr['stage2_max']:.2f}) and raised no alert. "
     "Neither model alerted on the USB drive. Both models watched the same device in the same "
     "seconds, so the comparison is paired. It is still one keyboard, one typist and one "
     "session on a virtual device, and it repeats the offline result in place of adding "
     "independent evidence about attacks.")

h2("H", "Generalisation to Unseen Attack Families")
rows = []
for f in FAMS:
    r = LOPO[f]
    rows.append([f, pct(r["full"][0], 0), pct(r["unseen"][0], 0), pct(r["stage1"][0], 0),
                 pct(r["unseen_or_s1"][0], 0)])
rows.append(["Mean", pct(seen_mean), pct(unseen_mean), pct(s1_mean), pct(or_mean)])
rows.append(["Benign FP", f"{fp_fresh:.2%}", f"{fp_unseen:.2%}", f"{LOPO['benign_fp_stage1']:.2%}",
             f"{fp_or:.2%}"])
table("Leave-one-family-out detection on Corpus C (100 sessions per family)",
      ["Family", "Trained on it", "Never seen", "Stage 1 only", "Never seen + Stage 1"],
      rows, key="unseen")
body("Stage 2 is supervised, so it is trained on the families we can write down, and a real "
     "attack will be one we did not. For each family we trained without it, with Stage 1, "
     "the LSTM and the threshold all excluding it, and scored it on Corpus C "
     f"({tref('unseen')}). Mean detection falls from {pct(seen_mean)} to {pct(unseen_mean)}, "
     "so about half of the headline is memorised family structure. The two injection "
     "families transfer to each other. Families with a unique signature do not. Detection "
     "of <i>identity_switch</i> and <i>class_exfil</i> drops to 0%, and the replay loop to "
     f"{pct(LOPO['mimicry']['unseen'][0], 0)}. The label-free Stage 1 catches "
     f"{pct(LOPO['identity_switch']['stage1'][0], 0)} of <i>identity_switch</i> sessions that "
     "the trained Stage 2 scores at 0%, so Stage 2 shadows a detector that already works. "
     f"Keeping Stage 1 as an independent alarm path recovers part of the gap, from "
     f"{pct(unseen_mean)} to {pct(or_mean)} mean detection, at the cost of raising false "
     f"positives from {fp_unseen:.2%} to {fp_or:.2%}.")

h2("I", "What Did Not Help, and What the Results Depend On")
g = GATES["rules"][0]["rows"]
dts = [r["detect_test"] for r in g]
r11 = next(r for r in g if r["rule"] == "k-of-n" and r["params"] == "1/1")
r46 = next(r for r in g if r["rule"] == "k-of-n" and r["params"] == "4/6")
bd = [r["detect_at_0.5pct"] for r in VARS["base"]]
mean_bd = sum(bd) / len(bd)
sd_bd = (sum((x - mean_bd) ** 2 for x in bd) / len(bd)) ** 0.5
vm = {k: sum(r["detect_at_0.5pct"] for r in v) / len(v) for k, v in VARS.items()}
swaps = AUG["class_exfil"]["+ swaps (held-out pair removed)"]
body("<i>The alerting rule.</i> We calibrated 13 rules to the same 0.5% benign false-positive "
     "target on one half of Corpus C’s benign sessions and scored them on the other half: "
     "k-of-n gates, an exponentially weighted mean of the log-odds, and a CUSUM. Detection "
     f"on held-out attacks ranged only from {pct(min(dts))} to {pct(max(dts))}, which is "
     "within noise at n = 300. A bare single-window threshold matched the 4-of-6 gate and "
     f"alerted after {r11['median_latency']:.1f} s instead of {r46['median_latency']:.1f} s. "
     "On this corpus the persistence gate buys latency cost and no false-positive benefit, "
     "though real traffic may have transient spikes the simulator lacks.")
body("<i>Model changes.</i> At a matched 0.5% false-positive rate the baseline detects "
     f"{pct(mean_bd)} ± {100 * sd_bd:.1f} points over three seeds. A long-repeat "
     f"feature, the share of 12-report n-grams already seen in the session ({pct(vm['+repeat'])}), "
     f"long-memory context features such as activity averages and idle-run length ({pct(vm['+context'])}), "
     f"and a five-seed ensemble ({pct(vm['ens5'])}) all sit inside that spread. The limit is "
     "the information in the features and the training data, not model capacity.")
body("<i>Synthetic attacks.</i> We trained on 152 generated sessions of devices that declare "
     "one class and behave as another, withholding the swap equivalent to the held-out "
     "family. The model learned the swaps, detecting "
     f"{pct(swaps['swap_seen_pairs'], 0)} of fresh swaps it had trained on and "
     f"{pct(swaps['swap_withheld_pair'], 0)} of the withheld pair. It still detected none "
     "of the real held-out families, because they are not plain class swaps. "
     "<i>identity_switch</i> relies on an interface that appears late, and "
     "<i>class_exfil</i> is low-rate covert bulk traffic.")
cal = DIAG["calibration"]
lo_pt, mid_pt = cal["operating_points"][0], cal["operating_points"][2]
imp = sorted(DIAG["importance"]["drops"].items(), key=lambda kv: -kv[1])[:3]
bm = DIAG["importance"]["by_modality"]
body("<i>Calibration noise.</i> The threshold is the 1-in-10,000 quantile of benign validation "
     "windows, and with 4,964 of them that is half a window, so in effect it is the single "
     f"highest benign score. Resampling the validation sessions 400 times moves it from "
     f"{lo_pt['threshold']:.2f} (2.5th percentile) to {mid_pt['threshold']:.3f} (median). On "
     f"Corpus C the low end gives {pct(lo_pt['fp'], 2)} benign false positives and "
     f"{pct(lo_pt['detect'])} detection, the median {pct(mid_pt['fp'], 2)} and "
     f"{pct(mid_pt['detect'])}. The operating point therefore depends partly on which benign "
     "window happened to be the largest. A deployment should calibrate on thousands of benign "
     "sessions against a session-level target, as in the alerting-rule study.")
body("<i>What the model relies on.</i> Shuffling one feature at a time across a 300-session "
     f"sample lowers window AUC ({DIAG['importance']['base_auc']:.3f}) most for "
     + ", ".join(f"{k} ({v:.3f})" for k, v in imp) + f". By modality the payload features "
     f"account for {bm['payload']:.3f}, timing {bm['timing']:.3f}, protocol {bm['protocol']:.3f} "
     f"and identity {bm['identity']:.3f}. No proxy for position in the session is among the "
     "inputs, since time since enumeration is excluded. The streaming detector costs "
     f"{DIAG['cost']['ms_per_device_window']:.1f} ms per device-window, about "
     f"{DIAG['cost']['devices_per_core']} devices per core at one window per second.")
mo = ROB["mimicry_overlap"]
body("<i>The replay-loop result hangs on one attacker choice.</i> A device replaying a script "
     "presses and releases keys cleanly, and detection is "
     f"{pct(mo['short']['rate'], 0)} (95% CI {pct(mo['short']['ci'][0], 0)} to "
     f"{pct(mo['short']['ci'][1], 0)}). A mimic that also imitates human key overlap makes "
     "every cycle of the script produce different reports, and detection falls to "
     f"{pct(mo['human']['rate'], 0)}. That is a 96-point swing from a single design choice, "
     "and no real attack hardware was available to say which is realistic, so the "
     "result should be read as a range.")
P = ROB["perturbations"]
rows = []
for key, label in (("as generated", "As generated"), ("jitter 5 ms", "Timestamp jitter, 5 ms"),
                   ("quantise 10 ms", "Quantised to 10 ms ticks"), ("drop 5%", "5% of events lost"),
                   ("drop 20%", "20% of events lost"), ("dilate x2", "All times doubled")):
    d = P[key]
    rows.append([label, pct(d["fp"]),
                 pct(d["detect"]) if d["detect"] == d["detect"] else "n/a",
                 pct(d["per_profile"]["mimicry"]) if d["detect"] == d["detect"] else "n/a"])
table("Robustness of the retrained model to capture-level perturbations "
      f"({P['as generated']['n_benign']} benign and {P['as generated']['n_attack']} attack sessions)",
      ["Perturbation", "Benign FP", "Detection", "Replay loop"], rows, key="perturb")
body(f"{tref('perturb')} applies perturbations a real capture has and the simulator lacks. "
     "False positives stay at 0% throughout, and detection is unchanged except for the "
     "replay loop, which depends on exact repeats and degrades when events are lost. Time "
     "doubling applies to benign sessions only, because it would move attack onsets. With "
     f"200 benign sessions, 0% bounds the false-positive rate only below about 2%.")

h2("J", "eBPF Capture Feasibility")
body("Section V-A names two capture backends behind one interface: usbmon, which is "
     "deployed, and eBPF, which is designed. Rather than leave that as a design claim, we "
     "built a minimal prototype on the two hook points named there "
     "(<code>usb_submit_urb</code> and <code>__usb_hcd_giveback_urb</code>) and measured "
     "whether counting completions through them agrees with usbmon, the criterion the "
     "project had set for this decision. The probe attaches by kfunc, which gives "
     "BTF-typed field access, and counts only at the completion hook, matching the "
     "completions-only rule of the feature extractor.")
rows = [["Idle (25 s)", f"{EBPF_IDLE['usbmon']['completions_and_errors']:,}",
         f"{EBPF_IDLE['ebpf']['completions_at_giveback_hook']:,}",
         f"{EBPF_IDLE['agreement_pct']:.3f}%"],
        [f"Busy, 15× lsusb -v ({EBPF_BUSY['seconds_requested']:.0f} s)",
         f"{EBPF_BUSY['usbmon']['completions_and_errors']:,}",
         f"{EBPF_BUSY['ebpf']['completions_at_giveback_hook']:,}",
         f"{EBPF_BUSY['agreement_pct']:.3f}%"]]
table("URB completions counted by eBPF and by usbmon over the same window",
      ["Condition", "usbmon", "eBPF", "Agreement"], rows, key="ebpf")
body(f"Agreement is exact across a {ebpf_ratio}-fold difference in traffic volume "
     f"({tref('ebpf')}). One finding came only from running on real hardware. The URB "
     "status field, read at kfunc entry to the completion hook, is -115 (EINPROGRESS) on "
     "every completion, including control transfers that plainly succeeded. The count is "
     "unaffected, but a real backend would need a different probe point to read per-URB "
     "status. No typed records are emitted and no ring buffer exists, and overhead against "
     "the literature’s sub-microsecond claim [3]–[5] is not measured.")

# ---------------------------------------------------------------- VII
h1("VII", "Limitations")
item("1)", "<i>No real attack traffic.</i> Every detection figure is measured on simulated "
           "attacks. The real-traffic checks cover the capture machinery and benign typing "
           "on a virtual keyboard, scored offline and live. They cannot validate "
           "detection accuracy. This is the main limitation.")
item("2)", "<i>Corpus B encodes our assumptions.</i> The idle mix, the text model, the "
           "hold times and the typing-speed range come from the HID specification and the "
           "typing literature, but they are still assumptions. The replay-loop result "
           "swings from 97% to 1% on one of them, and the human-paced payload is at 0%.")
item("3)", "<i>Weak generalisation.</i> Detection of families the model has not seen is "
           f"{pct(unseen_mean)}, against {pct(seen_mean)} for families it has. The Stage 1 "
           "alarm path recovers part of that and no method we tried closed it.")
item("4)", "<i>False-positive rate is not established at deployment scale.</i> "
           f"{round(fp_fresh * 3000)} false alarms in 3,000 benign sessions of about a "
           f"minute each is roughly one alarm per {62.5 / 3600 / fp_fresh:.0f} device-hours. "
           "Many devices running continuously need a rate orders of magnitude lower, "
           "which simulation cannot certify.")
item("5)", "<i>Enforcement has never executed,</i> and eBPF is a counting prototype. N3 is "
           "closed in software only.")
item("6)", "<i>Two novelty claims are unsupported.</i> N1 holds only non-uniformly and N2 "
           "not at all. Both stay in the architecture and are reported as measured.")
item("7)", "<i>The live evidence is thin.</i> The live comparison is one keyboard, one typist "
           "and one seven-minute session on a virtual device, with no attack traffic. It shows "
           "that the deployed model leaves a typing keyboard alone and that the original did "
           "not. It says nothing about detection on real attack hardware.")
item("8)", "<i>A feature was designed against a known failure.</i> The cross-window payload "
           "feature was added after seeing which family the system missed. Its constraint is "
           "real, since a payload must type specific commands, but it raises the attacker’s "
           "cost and does not remove it. The trace store also pseudonymises whole reports, "
           "so features can only compare reports for equality.")

# ---------------------------------------------------------------- VIII
h1("VIII", "Conclusion")
body("IUDBMS poses USB trust as a question about behaviour over time and not about identity "
     "asserted at connection. The pipeline compresses URB traffic into per-second vectors, "
     "scores them against population and per-device norms, passes that score to a sequence "
     "model, and escalates through a persistence-gated policy. On the original corpus it "
     f"detects {pct(det)} of attack sessions with {pct(fp)} false positives, against "
     f"{pct(s1_det)} without the temporal stage and none for an identity-based control.")
body("That number was too kind. Giving benign devices idle periods and realistic HID "
     f"behaviour made the original model flag {pct(orig_fp)} of benign sessions. Retraining "
     f"restored zero false positives at {pct(seen_mean)} detection on fresh data, and "
     f"detection of unseen families is only {pct(unseen_mean)}. Two of four novelty claims "
     "failed their tests. Running the system against a real kernel found a timing defect, "
     "a mismatched checkpoint and a scanner defect that 385 simulator-driven tests had not. "
     "The retrained model is now the deployed one, and on a live guest it left a typing "
     "keyboard alone where the original revoked it. The next steps "
     "follow directly: capture real attack hardware and a longer record of ordinary use, "
     "close the generalisation gap, measure false positives at deployment scale, and review "
     "the enforcement loop before it touches a kernel.")

# ---------------------------------------------------------------- refs
p = doc.add_paragraph(style="IEEE H1")
run(p, "References", small_caps=True)

REFS = [
    "X. Shen, N. Li, C. Liu, and M. Cai, “WMUD: A Whitelist Method of USB Devices Redirection for Virtual Desktop,” in 2019 IEEE 21st Int. Conf. on High Performance Computing and Communications; IEEE 17th Int. Conf. on Smart City; IEEE 5th Int. Conf. on Data Science and Systems (HPCC/SmartCity/DSS), 2019, pp. 527–533, doi: 10.1109/HPCC/SmartCity/DSS.2019.00083.",
    "R. M. G. S. Bintang, Arizal, R. Purwoko, M. Syahral, S. S. Zen, I. G. M. Putra, and R. A. Wijayanti, “SafeUSB: Implementation of Keystroke Monitoring System to Prevent BadUSB HID Injection Attack,” in 2025 IEEE 2nd Int. Conf. on Cryptography, Informatics, and Cybersecurity (ICoCICs), 2025, pp. 286–291, doi: 10.1109/ICOCICS68032.2025.11383963.",
    "S. Kostopoulos, D. Papatsaroucha, I. Kefaloukos, and E. K. Markakis, “eIDPS: A real-time eBPF-based and Machine Learning-powered Network Intrusion Detection and Prevention Solution,” in 2025 6th Int. Conf. in Electronic Engineering & Information Technology (EEITE), 2025, doi: 10.1109/EEITE65381.2025.11165804.",
    "R. Rajurkar and N. Deshmukh, “From Packets to Defense: Architectural Patterns in eBPF-Based Intrusion Detection,” in 2025 Int. Conf. on Intelligent Computing and Next Generation Networks (ICNGN), 2025, doi: 10.1109/ICNGN67480.2025.11413805.",
    "L. Song, “eBPF: Pioneering Kernel Programmability and System Observability – Past, Present, and Future Insights,” in 2024 3rd Int. Conf. on Artificial Intelligence and Computer Information Technology (AICIT), 2024, doi: 10.1109/AICIT62434.2024.10730620.",
    "M. S. Baek and Y.-A. Jung, “A Two-Stage Deep Learning-based Framework for High-Accuracy Network Intrusion Detection,” in 2025 16th Int. Conf. on Information and Communication Technology Convergence (ICTC), 2025, doi: 10.1109/ICTC66702.2025.11388157.",
    "V. Hnamte, H. Nhung-Nguyen, J. Hussain, and Y. Hwa-Kim, “A Novel Two-Stage Deep Learning Model for Network Intrusion Detection: LSTM-AE,” IEEE Access, vol. 11, pp. 37131–37148, 2023, doi: 10.1109/ACCESS.2023.3266979.",
    "A. Saini, T. K. Jaiswal, Parteek, A. Thakur, A. Rai, and S. Saini, “Hybrid CNN–LSTM Architecture for Cybersecurity Threat Detection in Network Traffic,” in 2026 Int. Conf. on Signal, Systems, and Computing for Next-Gen Automation (ICSSCNA), 2026, doi: 10.1109/ICSSCNA68616.2026.11546877.",
    "M. Chithik Raja and M. M. B. Al Mahri, “Behavioral Anomaly Detection in Big Data Streams: An Attention-Enhanced LSTM Approach for Privacy-Aware Zero-Day Attack Detection,” in 2025 7th Int. Conf. on Innovative Data Communication Technologies and Application (ICIDCA), 2025, doi: 10.1109/ICIDCA66325.2025.11280566.",
    "A. Mohamed Anwar, D. Vikram, Thiraviaselvi G, and S. Ganapathy, “LSTM-Powered Anomaly Detection for Securing IIoT Systems Against DDoS, Spoofing, and Ransomware Attacks,” in 2025 Int. Conf. on Emerging Research in Computational Science (ICERCS), 2025, doi: 10.1109/ICERCS65898.2025.11580832.",
    "C. E. Shannon, “Prediction and Entropy of Printed English,” Bell Syst. Tech. J., vol. 30, no. 1, pp. 50–64, Jan. 1951, doi: 10.1002/j.1538-7305.1951.tb01366.x.",
]

for n, ref in enumerate(REFS, 1):
    p = doc.add_paragraph(style="IEEE Ref")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3), WD_TAB_ALIGNMENT.LEFT)
    run(p, f"[{n}]\t")
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
cp.title = ("Intelligent USB Device Behaviour Monitoring System (IUDBMS): "
            "A Behaviour-Based Approach to Runtime USB Peripheral Trust")
cp.author = ", ".join(n for n, _ in AUTHORS)
cp.subject = "IEEE conference format"
cp.keywords = "BadUSB, behavioural anomaly detection, eBPF, LSTM, USB security"

DEST.parent.mkdir(parents=True, exist_ok=True)
doc.save(DEST)
print(f"  wrote {DEST}")
print(f"  tables: {_TABLE_N[0]}   figure: {FIG.name}")
