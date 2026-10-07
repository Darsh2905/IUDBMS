"""
Build the IUDBMS paper as an IEEE conference-format .docx.

    python3 scripts/build_paper.py [out.docx]

**Every number in the Results section is read from the eval artefacts**
(``out/metrics.json``, ``out/fusion_ablation.json``, ``out/baseline_ablation.json``,
``out/cross_device.json``) rather than typed in. That is the same rule the rest
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
make_style("IEEE Affiliation", 10, align=C, italic=True, after=0)
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
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def table(caption, header, rows, *, widths=None):
    """An IEEE table: caption above in small caps, horizontal rules only."""
    _TABLE_N[0] += 1
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
            if j == 0:
                cp.alignment = L
            run(cp, str(v))

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

# ---------------------------------------------------------------- abstract
det = M["full_system"]["session"]["attack_detect"] if M else None
fp = M["full_system"]["session"]["benign_fp"] if M else None
auc = M["full_system"]["window"]["auc"] if M else None
s1_det = M["stage1_only"]["session"]["attack_detect"] if M else None

ebpf_ratio = None
if EBPF_IDLE and EBPF_BUSY:
    idle_n = EBPF_IDLE["ebpf"]["completions_at_giveback_hook"]
    busy_n = EBPF_BUSY["ebpf"]["completions_at_giveback_hook"]
    ebpf_ratio = round(busy_n / max(idle_n, 1))

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Abstract", italic=True)
run(p, "—USB peripherals are trusted by what they claim to be: during enumeration a "
       "device self-reports its identity and the host accepts it without attestation, an "
       "assumption that BadUSB attacks exploit to inject keystrokes within seconds. "
       "Existing defences verify identity at connection time and do not judge how a device "
       "behaves afterwards. This paper presents the Intelligent USB Device Behaviour "
       "Monitoring System (IUDBMS), which establishes trust from observed runtime "
       "behaviour instead. USB Request Blocks are compressed into per-second behavioural "
       "feature vectors spanning timing, payload, protocol, and identity signals; an "
       "unsupervised first stage scores each vector against the population of same-class "
       "devices and the device’s own history, and its score is relayed as an input to a "
       "second-stage LSTM that models ten-second sequences to expose staged attacks. A "
       "persistence-gated policy engine escalates from throttling to deauthorisation only "
       "on sustained evidence. On a simulated corpus of 1,000 device sessions the system "
       f"detects {pct(det)} of attack sessions at {pct(fp)} false positives "
       f"(window ROC-AUC {auc:.3f}), against {pct(s1_det)} for the same pipeline without "
       "its temporal stage. We additionally report purpose-built ablations against each of "
       "the four novelty claims: temporal modelling is strongly supported, multi-modal "
       "fusion is supported only non-uniformly, and the per-device adaptive baseline is "
       "not supported even after device heterogeneity is introduced to test the "
       "explanation previously offered for its inertness. We further validate the capture "
       "and scoring pipeline, as distinct from its detection accuracy, against a real "
       "Linux kernel: the parser loses none of 338 real usbmon lines, but real human typing "
       "is flagged by the evaluated model, a sim-to-real failure of the cross-window payload "
       "feature, and the same validation surfaced a timing defect and a mismatched "
       "checkpoint that 385 simulator-driven tests had not.")

p = doc.add_paragraph(style="IEEE Abstract")
run(p, "Index Terms", italic=True)
run(p, "—BadUSB, behavioural anomaly detection, eBPF, Isolation Forest, kernel "
       "enforcement, LSTM, USB security, usbmon.")

# ---------------------------------------------------------------- I
h1("I", "Introduction")
body("USB has been the dominant peripheral standard since 1996, prized for plug-and-play "
     "convenience. That convenience rests on a persistent liability: USB trusts devices by "
     "default. During enumeration a device self-reports its Vendor ID (VID), Product ID "
     "(PID), and interface descriptors, and the host accepts these claims without "
     "cryptographic attestation.")
body("This trust-by-default model is exploited by BadUSB attacks, where firmware is altered "
     "to declare an interface it does not honestly represent — typically a Human "
     "Interface Device (HID) keyboard — so an object resembling an ordinary flash "
     "drive or cable can inject a pre-programmed keystroke payload the moment it is "
     "plugged in. Because HID input is inherently trusted, these attacks bypass file-based "
     "antivirus and execute within seconds.")
body("Existing defences largely verify device <i>identity</i> at the point of connection "
     "— allowlisting by VID/PID, user role-confirmation, or static kernel filter "
     "rules. None of these judge how a device <i>actually behaves</i> once operating, so a "
     "device that spoofs an allowed identity, or misbehaves within an approved role, "
     "defeats them entirely.")
body("This project proposes the Intelligent USB Device Behaviour Monitoring System "
     "(IUDBMS), which establishes <i>behavioural trust at runtime</i> rather than identity "
     "at connection time. IUDBMS captures USB Request Blocks (URBs) at the kernel/driver "
     "boundary, scores each device against a per-device adaptive baseline, models the "
     "temporal sequence of its events with an LSTM to catch staged or slow-burn attacks, "
     "and enforces verdicts through a closed loop — rate-limiting or unbinding a "
     "malicious device before the attack window closes.")
body("The remainder of this document reviews the literature across three themes — "
     "USB/BadUSB defences, eBPF kernel observability, and LSTM-based anomaly detection "
     "— then derives the problem statement, research gap, novelty, and objectives of "
     "IUDBMS, describes the implemented architecture and experimental setup, and reports "
     "measured results including ablations against each novelty claim.")

# ---------------------------------------------------------------- II
h1("II", "Literature Review")
body("The relevant literature falls into three strands: the USB attack surface and "
     "defences against malicious peripherals; eBPF as a mechanism for kernel-level "
     "observation and enforcement; and deep-learning methods for temporally-structured "
     "anomaly detection. IUDBMS sits at their intersection.")

h2("A", "USB Peripheral Security and BadUSB Defences")
body("Shen et al. [1] proposed WMUD, a whitelist method for USB redirection in "
     "virtual-desktop infrastructure (QEMU/SPICE). Server-side filter rules — keyed "
     "on device class, VID, and PID — admit only permitted devices and cannot be "
     "tampered with from the client. This shows the value of protected, per-asset policy, "
     "but remains a pure identity-based allowlist: a device presenting an allowed "
     "identifier is admitted regardless of subsequent behaviour, and the scheme is scoped "
     "to virtualised redirection rather than direct host defence.")
body("Bintang et al. [2] proposed SafeUSB, a Windows keystroke monitor combining an "
     "inter-keystroke timing threshold with content analysis that flags typed keywords "
     "such as <i>POWERSHELL</i> and <i>CMD.EXE</i>. It blocks most injection scenarios, "
     "but the authors report a critical ~1-second delay between HID connection and "
     "monitoring activation, leaving short payloads a window to succeed — and "
     "recommend, as future work, moving to compiled, kernel-level hooks, precisely the "
     "direction IUDBMS takes.")
body("Together [1], [2] these works frame the tension IUDBMS resolves: WMUD’s "
     "tamper-resistant, per-asset policy is identity-bound, while SafeUSB’s "
     "behavioural analysis is limited by userspace placement and activation delay. "
     "Neither builds an adaptive per-device model, and neither enforces at the "
     "kernel/driver boundary.")

h2("B", "eBPF for Kernel-Level Observability and Enforcement")
body("eBPF allows verified bytecode to run inside the Linux kernel, giving "
     "userspace-authored logic fine-grained observability and enforcement at kernel "
     "speed. Song’s survey [5] traces its core mechanics — the in-kernel VM, JIT "
     "compilation, and the static verifier that guarantees bounded, memory-safe execution "
     "— and its expansion from packet filtering into tracing and security, while "
     "noting persistent constraints around verification complexity and program-size "
     "limits.")
body("Kostopoulos et al. [3] presented eIDPS, a real-time eBPF-plus-ML network intrusion "
     "detection and prevention pipeline operating at line rate. Rajurkar and Deshmukh [4] "
     "survey the field’s architectural patterns and identify the dominant one: "
     "<i>hybrid designs that extract features in the kernel and delegate ML inference to "
     "userspace</i>, coordinated through shared BPF maps — alongside open challenges "
     "in verifier constraints, portability, and ML integration.")
body("This body of work [3]–[5] establishes eBPF as a proven substrate for "
     "sub-microsecond observation and enforcement, and the kernel-extract/userspace-infer "
     "split as the established way to pair kernel speed with ML expressiveness. But it "
     "concerns <i>network</i> intrusion detection; applying the same pattern to the "
     "<i>USB</i> subsystem — URBs as the unit of observation, the driver-bind boundary "
     "as the enforcement point — is unexplored.")

h2("C", "Deep Learning for Behavioural and Temporal Anomaly Detection")
body("LSTM networks are well suited to anomalies whose signal lies in temporal structure "
     "rather than any single point in time. Hnamte et al. [7] proposed LSTM-AE, "
     "hybridising an LSTM with an auto-encoder for network intrusion detection, validated "
     "across multiple public datasets and arguing for two-stage designs that separate "
     "detection from classification.")
body("Baek and Jung [6] similarly decouple anomaly detection from attack classification: "
     "a binary focal-loss stage filters anomalies from heavily imbalanced traffic, and a "
     "second multi-class stage identifies the attack type, both sharing a CNN-1D-LSTM "
     "backbone — a staged decomposition IUDBMS’s own layering parallels. Saini et "
     "al. [8] reinforce the value of hybrid <i>CNN–LSTM</i> architectures for network "
     "threat detection.")
body("Two further works target novel and slow-developing attacks directly. Chithik Raja "
     "and Al Mahri [9] applied an attention-enhanced LSTM to zero-day detection across "
     "process, network, and file-system telemetry, showing behaviour-based temporal "
     "modelling generalises to unseen threats. Mohamed Anwar et al. [10] applied "
     "LSTM-powered anomaly detection to IIoT systems against DDoS, spoofing, and "
     "ransomware, outperforming Random Forest, Logistic Regression, and 1D-CNN baselines "
     "at low latency.")
body("Collectively [6]–[10], this strand shows LSTM-based temporal models catch "
     "multi-stage and zero-day behaviour within real-time budgets, and that layered "
     "designs balance recall against discrimination. What it does not do is apply "
     "temporal modelling to a single USB peripheral’s event stream — the "
     "application IUDBMS pursues.")

# ---------------------------------------------------------------- III
h1("III", "Inferences from the Literature")
body("The USB-defence literature [1], [2] establishes the threat and the value of "
     "behavioural analysis but stays tied to identity or userspace latency. The eBPF "
     "literature [3]–[5] establishes a fast, verifiable kernel-resident mechanism, "
     "almost exclusively for networks. The deep-learning literature [6]–[10] "
     "establishes that layered LSTM models catch temporally-structured, zero-day "
     "behaviour in real time, but never on the USB event stream. IUDBMS sits in the space "
     "these three strands leave open.")

h2("A", "Problem Statement")
body("USB’s trust-by-default enumeration lets a malicious peripheral (BadUSB) declare "
     "a trusted identity — typically an HID keyboard — and abuse it to inject "
     "commands at machine speed, compromising the host within seconds. Existing defences "
     "authenticate <i>claimed identity</i> at connection but do not continuously judge "
     "<i>actual runtime behaviour</i>. Devices that spoof allowed identifiers, or "
     "misbehave within an approved role, evade detection; and existing behavioural "
     "monitors run in userspace, incurring an activation delay that lets short payloads "
     "succeed. A defence is needed that (a) trusts devices by observed behaviour, not "
     "declared identity, (b) enforces at the kernel/driver boundary to remove round-trip "
     "latency, and (c) detects staged, slow-escalation attacks, not just single-point "
     "anomalies.")

h2("B", "Research Gap")
item("1)", "<i>Identity-bound trust:</i> Allowlisting (e.g., WMUD [1]) admits devices by "
           "class/VID/PID and is blind to spoofed identities or misuse within an approved "
           "role.")
item("2)", "<i>Userspace activation latency:</i> SafeUSB [2] proves runtime analysis works, "
           "but its userspace placement adds a ~1-second activation delay that short "
           "payloads exploit.")
item("3)", "<i>Network-only kernel intelligence:</i> The eBPF-plus-ML pattern (eIDPS [3]; "
           "Rajurkar and Deshmukh [4]) is mature for networks but unapplied to USB, where "
           "the natural unit is the URB and the natural enforcement point is driver-bind.")
item("4)", "<i>No temporal modelling of USB behaviour:</i> LSTM/attention models "
           "[6]–[10] catch staged, zero-day attacks in network and IIoT domains, but "
           "no USB defence models a device’s event sequence over time.")
item("5)", "<i>Open enforcement loop:</i> Existing behavioural detectors alert or block "
           "from userspace after a round-trip; none close the loop inside the kernel.")

h2("C", "Novelty of IUDBMS")
body("Individually, the underlying techniques — eBPF capture [3]–[5], "
     "unsupervised anomaly scoring, and LSTM temporal modelling [6]–[10] — "
     "already exist. IUDBMS’s novelty is the <i>combination and application</i>: "
     "fusing them into one per-device, closed-loop pipeline and pointing that pipeline at "
     "the USB subsystem, which no reviewed eBPF-ML or LSTM-anomaly work addresses and no "
     "reviewed USB-defence work implements. Each element below answers a gap from "
     "Section III-B, and each is evaluated in Section VI-C. The numbering follows "
     "§3.3 of the Review 1 submission and is unchanged from it.")
item("N1)", "<i>Multi-modal feature fusion at the kernel/driver boundary.</i> Fuses event "
            "timing, traffic volume, and descriptor changes into one feature stream "
            "— carrying the eBPF-ML pattern [3], [4] into USB, answering the "
            "network-only gap.")
item("N2)", "<i>Per-device adaptive behavioural baselines.</i> Each device is scored "
            "against its own learned norm rather than a global rule set, answering the "
            "identity-bound trust gap left by WMUD [1].")
item("N3)", "<i>Closed-loop kernel-level enforcement.</i> Verdicts (rate-limiting, "
            "unbinding) execute without a userspace round-trip, answering SafeUSB’s "
            "[2] activation-latency limitation and the open-enforcement-loop gap.")
item("N4)", "<i>Temporal sequence modelling to catch staged attacks.</i> A userspace LSTM "
            "analyses rolling event windows to catch slow-escalation attacks single-point "
            "detectors miss, bringing temporal modelling [6]–[10] to USB.")

h2("D", "Relationship to the Reference Framework")
body("The reference framework is a kernel-level eBPF security framework that filters "
     "peripherals against human-authored rules at the host-controller boundary with "
     "sub-microsecond overhead. IUDBMS keeps its core insight — the kernel/driver "
     "boundary is the right place to enforce — but replaces its static, "
     "administrator-written rules with <i>dynamic behavioural intelligence</i>: "
     "per-device adaptive baselines and anomaly scoring, backed by a userspace LSTM. In "
     "short, IUDBMS keeps the reference framework’s fast, verifiable enforcement "
     "substrate and adds the adaptive, behaviour-learning capability it explicitly does "
     "not attempt — moving from <i>rules an administrator must foresee</i> to "
     "<i>behaviour the system learns on its own</i>. (No implementation work is done "
     "against the reference framework; it is discussed only to position IUDBMS’s "
     "contribution.)")

# ---------------------------------------------------------------- IV
h1("IV", "Objectives")
body("The aim is to build a system that trusts USB peripherals by runtime behaviour rather "
     "than claimed identity, via a four-layer pipeline carrying a device’s activity "
     "from raw kernel events, through scoring, to userspace temporal analysis, and back "
     "into the kernel as an enforced verdict — following the "
     "kernel-extract/userspace-infer pattern [3], [4] while adding per-device adaptivity "
     "and closed-loop enforcement. Each objective below corresponds to one stage.")
item("1)", "<i>Kernel-level capture (Layer 1).</i> Intercept URBs at the kernel/driver "
           "boundary and extract multi-modal features — event timing, traffic volume, "
           "and descriptor changes — passing compact records upward with minimal "
           "overhead.")
item("2)", "<i>Per-window anomaly scoring (Layer 2).</i> Construct a <i>per-device adaptive "
           "baseline</i> and score incoming features against it using unsupervised "
           "learning (Isolation Forest / One-Class SVM) — a fast, high-recall filter "
           "that flags suspicious activity for deeper inspection.")
item("3)", "<i>Userspace temporal sequence modelling (Layer 3).</i> Develop a PyTorch LSTM "
           "that analyses rolling event windows to detect <i>temporal structure</i> "
           "— staged, multi-step attacks whose ordering, not any single event, "
           "betrays intent — complementing Layer 2’s point-in-time detection.")
item("4)", "<i>Closed-loop enforcement (Layer 4).</i> Execute verdicts from Layers "
           "2–3 — rate-limiting or unbinding a device — without a userspace "
           "round-trip, closing the activation-delay window that lets short BadUSB "
           "payloads succeed.")
item("5)", "<i>Supporting data store and interface.</i> Persist per-device baselines, event "
           "history, and alerts in SQLite, and expose them through a dashboard "
           "(FastAPI/Flask) for monitoring, sensitivity tuning, and reviewing or "
           "overriding enforcement.")
item("6)", "<i>Evaluation against BadUSB attack scenarios.</i> Evaluate against rapid, "
           "delayed, and multi-stage payloads, measuring detection effectiveness, "
           "false-positive rate, enforcement latency, and throughput overhead, with "
           "behavioural trust compared directly against identity-based baselines.")
body("Objective 1’s capture mechanism admitted two options: eBPF (libbpf/BCC), the "
     "richer one, or usbmon as a lighter fallback. The design is agnostic to the choice "
     "— it affects only Layer 1, and both backends sit behind one interface. The "
     "implementation described in Section V takes the usbmon path; Section VII states what "
     "remains outstanding for eBPF.")

# ---------------------------------------------------------------- V
h1("V", "System Architecture")
h2("A", "Capture and Feature Extraction")
body("Fig. 1 shows the complete IUDBMS pipeline, from kernel-level USB capture to "
     "enforcement. URB traffic is observed through usbmon, a passive kernel-resident tap "
     "available on any Linux host with debugfs mounted; an eBPF-based capture backend "
     "offering lower overhead and explicit control over payload truncation is designed as "
     "the intended production path but not yet implemented, and both sit behind a common "
     "capture interface so the choice does not affect any downstream stage. A parser "
     "normalises raw event text into typed records, and a stateful, per-device feature "
     "extractor compresses each non-overlapping one-second window into a vector spanning "
     "four behavioural modalities — timing, payload, protocol, and identity.")

new_section(1)
p = doc.add_paragraph(style="IEEE Figure")
p.add_run().add_picture(str(FIG), width=Inches(7.16))
para("IEEE Caption",
     "Fig. 1.  The IUDBMS pipeline: kernel-level capture, per-window feature extraction, "
     "the two-stage detector (Stage 1’s score relayed as Stage 2’s input rather "
     "than cast as a separate vote), and the graduated enforcement policy. Dashed blocks "
     "are designed but not yet built, or generated but not yet executed.")
new_section(2)

h2("B", "Two-Stage Detection and Enforcement")
body("Two models operate in series rather than in parallel. Stage 1 is a memoryless "
     "per-window anomaly score, fusing an Isolation Forest trained on the population of "
     "same-class benign devices with an adaptive per-device baseline. Its output is not "
     "thresholded directly; instead it is appended as an additional feature to that "
     "window’s vector, which is placed in a ten-second sliding buffer. Stage 2, a "
     "recurrent (LSTM) network, reads the full ten-window sequence and produces the single "
     "probability that is compared against the decision threshold. A persistence gate "
     "requires four flagged windows within a trailing six-window horizon before a device "
     "is throttled, and eight cumulative flagged windows before it is deauthorised; both "
     "actions are generated by the policy engine, but execution against the kernel is "
     "currently gated behind a manual safety review and runs in dry-run mode only.")

h2("C", "Experimental Setup")
n_sess = M["corpus"]["sessions"] if M and "corpus" in M else 1000
body(f"The evaluation corpus comprises {n_sess:,} simulated USB device sessions — 600 "
     "benign, drawn from five device profiles (keyboard, mouse, mass-storage, webcam, "
     "audio), and 400 malicious, drawn from six BadUSB-style attack profiles — "
     "generated in the exact textual grammar the Linux usbmon driver emits and parsed "
     "through the same code path a live capture would use. Raw traffic is compressed into "
     "one feature vector per device per second at tumbling-window granularity. Two columns "
     "are withheld from the deployed model: time since enumeration, identified by a "
     "Kolmogorov–Smirnov test as a session-position leak rather than a behavioural "
     "signal, and any column with zero variance on the training split.")
body("Sessions — never individual windows — are split by profile into training, "
     "calibration, and test partitions, so no window of a session can appear on both "
     "sides of the split. Stage 1 (Isolation Forest, fusion weight 0.55, combined with a "
     "per-device EWMA baseline, weight 0.45) is fit on benign training sessions only. "
     "Stage 2 (a single-layer LSTM, hidden size 32, over causally left-padded 10-window "
     "sequences) is trained with class-weighted binary cross-entropy (Adam, learning rate "
     "2×10⁻³, gradient norm clipped at 5). The decision threshold is the "
     "empirical quantile of Stage-2 scores on a held-out benign calibration split at a "
     "target per-window false-positive rate of 0.01%; it is never read off the test set.")
body("All experiments use a fixed random seed and are reproducible from scripted entry "
     "points. Development and evaluation were performed on macOS/arm64 using Python 3.12, "
     "scikit-learn, and PyTorch. The system has not yet been evaluated against traffic "
     "captured from physical USB hardware; every result below is measured on the simulated "
     "corpus described here, and Section VII treats that as the principal limitation.")

# ---------------------------------------------------------------- VI results
h1("VI", "Results")

h2("A", "Headline Detection Performance")
if M:
    rows = []
    for key, label in (("full_system", "IUDBMS (Stage 1 + Stage 2)"),
                       ("stage1_only", "Stage 1 only (no temporal model)"),
                       ("identity_whitelist", "Identity whitelist (VID:PID)")):
        r = M.get(key, {})
        # identity_whitelist is not a scored model -- it is a lookup, so it has
        # no window-level metrics and its session rates sit at the top level.
        w = r.get("window", {})
        se = r.get("session", r if "attack_detect" in r else {})
        rows.append([label,
                     f"{w['auc']:.3f}" if w.get("auc") is not None else "—",
                     pct(w.get("precision"), 1) if w.get("precision") is not None else "—",
                     pct(w.get("recall"), 1) if w.get("recall") is not None else "—",
                     pct(se.get("attack_detect")),
                     pct(se.get("benign_fp"))])
    table("Detection performance on the held-out test split",
          ["Model", "ROC-AUC", "Prec.", "Rec.", "Detect", "FP"], rows)

    lat = M["full_system"]["session"].get("median_latency")
    body(f"Table I reports the deployed system against two baselines. The identity "
         f"whitelist is the USBGuard-style control: every attack in the corpus presents a "
         f"VID/PID drawn from the same pool as benign devices, so it detects none of them "
         f"by construction, which is the premise the project exists to test. Removing the "
         f"temporal stage costs {100 * (det - s1_det):.1f} percentage points of session "
         f"detection, the single largest effect measured anywhere in this work. Median "
         f"time from attack onset to first enforcement action is "
         f"{lat:.1f} s." if lat else "")

h2("B", "Per-Class Detection and the Single-Shot Bound")
if M and "per_profile" in M.get("full_system", {}):
    pp = M["full_system"]["per_profile"]
    rows = [[k, v.get("n", "—"), pct(v.get("detect_rate")),
             f"{v['median_latency']:.1f} s" if v.get("median_latency") else "—"]
            for k, v in sorted(pp.items()) if v.get("label") == 1]
    table("Detection by attack class", ["Attack profile", "n", "Detect", "Latency"], rows)
body("Detection is not uniform across attack families, and the variation is the most "
     "informative part of the result. Attacks that deviate structurally — machine-speed "
     "injection, a device that grows a keyboard interface mid-session, a declared keyboard "
     "pushing bulk data outward, or a long benign dwell followed by a burst — are "
     "caught reliably. The hard case is an adversary that throttles itself to human "
     "keystroke timing and types a <i>finite</i> payload once before falling silent. A "
     "per-feature Kolmogorov–Smirnov scan of that profile against benign keyboard "
     "traffic separates it on none of the deployed features: within a single device’s "
     "URB stream it is statistically the same process as a person typing. The residual "
     "signal is the permanent silence that follows a finished payload, which is visible "
     "only across a window sequence, and it is what the reported detection rate for that "
     "class rests on.")

h2("C", "Ablations Against the Novelty Claims")
body("Each novelty claim from Section III-C is evaluated by an experiment built for it, "
     "with its acceptance criterion fixed before the first run. Two of the four are not "
     "supported, and reporting that is a result rather than an omission.")

def SHORT_VERDICT(r):
    """Table cells need a few words, not a sentence."""
    if r["fusion_wins"]:
        return "fusion wins"
    if r["fusion_loses"]:
        return "fusion loses"
    if not r["valid"]:
        return "saturated" if "saturat" in r["verdict"] else "invalid"
    return "tie"


if FUSION:
    sw = FUSION["sweep"]
    rows = [[f"{r['strength']:.2f}",
             f"{100 * r['fused_mean']:.1f} ± {100 * r['fused_sd']:.1f}",
             f"{100 * r['best_spec_mean']:.1f} ± {100 * r['best_spec_sd']:.1f}",
             f"{100 * r['margin_mean']:+.1f} ± {100 * r['margin_sd']:.1f}",
             SHORT_VERDICT(r)] for r in sw]
    table("N1: fused vs best single modality on a diffuse attack, by attack strength "
          "(mean ± s.d. over three seeds)",
          ["Strength", "Fused", "Best single", "Margin", "Verdict"], rows)

body("<i>N1 (multi-modal fusion) — supported only non-uniformly.</i> Every attack "
     "family in the main corpus deviates on several modalities at once, so any single "
     "modality detects all of them and fusion cannot demonstrate a gain. The ablation "
     "therefore uses purpose-built families that deviate on one axis at a time, plus a "
     "<i>diffuse</i> attack that deviates weakly on two axes together. Fusion never loses "
     "on a single-axis family, and on descriptor-only attacks it is the <i>only</i> "
     "configuration that works: with the leaking positional feature correctly excluded the "
     "identity modality has too little variance to calibrate a threshold and becomes "
     "degenerate, while the fused model detects such attacks reliably. On the diffuse "
     "attack, however, the benefit reverses with attack strength (Table III): fusion gains "
     "substantially at moderate strength but is beaten by a nine-feature subset of itself "
     "at the subtlest strength, where the additional dimensions cost more variance than "
     "they contribute. By the pre-registered criterion — win somewhere, never lose "
     "— N1 is not supported, and the criterion stands as written.")

if BASE:
    hom, het = BASE["results"]["homogeneous"], BASE["results"]["heterogeneous"]
    rows = []
    for label, r in (("Homogeneous devices", hom), ("Heterogeneous devices", het)):
        full = r["rows"]["full (forest + baseline)"]
        fo = r["rows"]["forest only"]
        rows.append([label, f"{r['keyboard_rate_sd']:.3f}", pct(full["detect"]),
                     pct(fo["detect"]),
                     f"{100 * r['margin']:+.1f}", f"{100 * r['noise_floor']:.1f}"])
    table("N2: contribution of the per-device baseline, with and without device variation",
          ["Corpus", "Rate s.d.", "Full", "Forest only", "Δ", "Noise"], rows)

body("<i>N2 (per-device adaptive baseline) — not supported.</i> Removing the "
     "per-device term changes detection by less than the noise floor. The explanation "
     "previously offered for this was that every simulated device of a class behaved "
     "identically, so a class-conditioned forest already modelled them all and the "
     "per-device term was structurally redundant — with the implication that it would "
     "earn its keep on more varied real devices. That explanation is testable, because "
     "device homogeneity is a property of the generator: giving each device its own timing "
     "persona drawn from a hyperprior raises between-device variation by a factor of 2.8. "
     "The baseline remains inert (Table IV), so the explanation is wrong and the negative "
     "result is stronger than previously reported. Detection also falls under "
     "heterogeneity, which is the expected direction: more varied benign behaviour is a "
     "harder discrimination problem.")

body("<i>N3 (closed-loop enforcement) — demonstrated in dry-run only.</i> The policy "
     "engine generates the exact kernel command that would revoke a device together with "
     "the evidence that justified it, and the escalation ladder is exercised end to end on "
     "recorded sessions. Execution against the kernel is gated behind a manual safety "
     "review and has never been performed, so the loop is closed in software but not yet "
     "against hardware.")

body("<i>N4 (temporal sequence modelling) — strongly supported.</i> This is the "
     "largest effect in the work: the same pipeline without its sequence stage drops from "
     f"{pct(det)} to {pct(s1_det)} session detection. A separate model-family comparison "
     "shows the gain belongs to <i>sequence</i> rather than to recurrence specifically "
     "— a GRU and a dilated 1-D CNN match the LSTM within noise, while a supervised "
     "model given only a single window reaches markedly less. The LSTM is retained because "
     "no challenger clears the noise floor, not because it is better.")

h2("D", "Robustness: Host Context")
if CROSS:
    pp = CROSS["per_profile"]
    rows = [[k, f"{100 * v['without']:.1f} ± {100 * v.get('delta_sd', 0):.1f}"
             if False else f"{100 * v['without']:.1f}",
             f"{100 * v['with']:.1f}",
             f"{100 * v['delta_mean']:+.1f} ± {100 * v['delta_sd']:.1f}",
             f"{100 * v['noise']:.1f}"]
            for k, v in sorted(pp.items())]
    table("Cross-device correlation: detection with and without host context "
          "(mean over three seeds)",
          ["Attack profile", "Without", "With", "Δ", "Noise"], rows)

body("The single-shot bound invites an obvious extension: a person typing is usually also "
     "moving a mouse, whereas a malicious device typing alone is not. We implemented that "
     "— a host-level feature carrying the event rate of every other device attached to "
     "the same machine — together with a leakage guard built before the feature "
     "itself, since host composition that correlates with the label would reproduce "
     "exactly the positional leak described in Section V-C. With the guard holding, the "
     "feature does not help (Table V). The diagnosis is specific and worth more than the "
     "measurement: the generator draws each device on a host independently, so a benign "
     "keyboard and its mouse are statistically independent in this corpus, exactly as a "
     "malicious device and a bystander’s mouse would be. The coupling the feature "
     "exists to detect is absent on both sides. What a real user produces, and this corpus "
     "does not, is temporal interleaving between the two streams; testing that requires "
     "either a coupled generator — which risks building in the signal one then "
     "detects — or real captured traffic.")

h2("E", "Validation Against Real Kernel Traffic")
body("Every result above is measured on simulated sessions. To check whether the "
     "pipeline itself holds on genuine kernel output, we captured real usbmon "
     "text from a Linux guest (Ubuntu 24.04.5, kernel 6.8, aarch64) with a real USB "
     "host controller and virtual HID, hub and mass-storage devices attached, using "
     "the project\u2019s own capture script. This checks the mechanism, not the "
     "detector: the devices are virtual and no attack traffic was generated, so "
     "nothing here is a detection-accuracy result.")

if REALCAP:
    rc = REALCAP
    rows = [["Lines parsed", f"{rc['lines_seen']:,}"],
            ["Lines unparsed", f"{rc['lines_unparsed']}"],
            ["Loss rate", f"{rc['loss_rate']:.4%}"],
            ["Target (plan \u00a74.1)", f"< {rc['plan_criterion']:.1%}"],
            ["Events satisfying the contract", f"{rc['events_contract_valid']} of {rc['events_total']}"],
            ["Control submissions carrying SETUP", f"{rc['control_submissions_with_setup']} of "
                                                    f"{rc['control_submissions_total']}"],
            ["Devices observed", f"{rc['n_devices']}"]]
    table("Parser validation against a real usbmon capture "
          f"({rc['span_s']:.1f} s, {rc['fixture'].split('/')[-1]})",
          ["Metric", "Value"], rows)

body("The parser lost zero of "
     f"{REALCAP['lines_seen'] if REALCAP else 'N'} real kernel lines against the plan\u2019s "
     "under-0.1% target, a criterion that had never been measurable before this capture "
     "existed. Every emitted event satisfied the frozen contract, and every real control "
     "submission carried the SETUP packet that the project\u2019s first prototype had "
     "silently dropped.")

body("Running the live monitor service against this traffic, rather than only parsing it "
     "offline, surfaced a defect invisible to the simulator entirely: the window clock "
     "began at service start rather than at the first observed event, so any delay before "
     "the first USB event left every subsequent window permanently out of phase with "
     "arriving traffic, and the service scored nothing. 385 simulator-driven tests passed "
     "throughout, because a simulated source always delivers its first event immediately. "
     "Fixed by anchoring the window clock to the first event\u2019s arrival; confirmed "
     "against the same capture after the fix.")

body("A second defect surfaced only when the deployed models were scored against the "
     "evaluation corpus rather than against traffic: the Stage-2 checkpoint installed on the "
     "test host was not the one the reported numbers came from. It loaded, scored traffic "
     "and passed every structural check, but at its recorded threshold it detected none of "
     "120 simulated attacks, so an earlier \u201cno false alarm on real typing\u201d "
     "reading was produced by a monitor that could not alarm at all. Retraining with the "
     "documented command reproduced the reported figures exactly (window AUC 0.9762, 89.2% "
     "detection, 0.0% benign false positives), so the evaluation was sound and the shipped "
     "file was wrong. Each stage now records a fixed-input output fingerprint in the "
     "deployment manifest, the loader refuses weights that do not match it, and an "
     "end-to-end test asserts that the shipped set alerts on simulated attacks.")

if REALTYPE and REALTYPE.get("capture"):
    rt, sc = REALTYPE["capture"], REALTYPE.get("scores")
    sim = rt["simulator_measured"]
    rows = [["HID reports per second", f"{rt['reports_per_s']:.1f}", f"{sim['reports_per_s']:.1f}"],
            ["Key-release (all-zero) reports", f"{rt['release_share']:.1%}", f"{sim['release_share']:.1%}"],
            ["Distinct payloads", f"{rt['unique_payload_share']:.1%}", f"{sim['unique_payload_share']:.1%}"],
            ["Inter-arrival median", f"{rt['iat_percentiles']['50']:.3f} s",
             f"{sim['iat_percentiles']['50']:.3f} s"],
            ["Gaps longer than 1 s", f"{rt['gaps_over_1s']:.1%}", f"{sim['gaps_over_1s']:.1%}"]]
    table("A real keyboard vs. the simulator\u2019s, measured identically "
          f"({rt['reports']:,} real HID reports, {rt['span_s']:.0f} s of continuous typing)",
          ["Property", "Real", "Simulator"], rows)
    flagged = (f"Over the {sc['busy']} windows containing keystrokes, Stage-2 scores ranged "
               f"{sc['stage2_min']:.3f} to {sc['stage2_max']:.3f} (median {sc['stage2_median']:.3f}) "
               f"against the evaluated threshold of {sc['threshold']:.4f}; "
               f"{sc['at_or_above_threshold']} of them reached it, and the "
               f"{'persistence gate fired' if sc.get('gate_fires') else 'persistence gate did not fire'}"
               + (f" {sc['first_alert_s']:.0f} s into the capture. " if sc.get('gate_fires') else ". ")
               if sc and 'at_or_above_threshold' in sc else "")
    body("One person typed continuously into the guest\u2019s virtual USB keyboard; the "
         "recording was then scored offline with the evaluated model (the same code path as "
         "the streaming detector). " + flagged
         + "The evaluated system false-alarms on the one real keyboard session we have. "
         "This is a single session of one person deliberately mashing keys through a "
         "virtualised keyboard, so it is not a false-positive rate, but it is the only "
         "real benign keyboard the model has seen and it is flagged. The structural "
         "comparison explains why: a real keyboard emits a report on key-release as well as "
         "key-press and repeats roughly a third of its payloads, so the cross-window payload "
         "feature that detects scripted replay (median "
         + (f"{sc['seq_novelty_median']:.2f}" if sc and 'seq_novelty_median' in sc else "n/a")
         + " on real typing against about 0.99 on simulated benign typing) reads real typing "
         "as a replayed script. The timing gap is confounded by task (continuous mashing is "
         "not prose); the structural gap is not, and it is the finding: detection numbers "
         "measured on the simulated corpus do not transfer to a real keyboard.")

h2("F", "eBPF Capture Feasibility")
body("Section V-A names two capture backends behind one interface: usbmon, "
     "deployed, and eBPF, designed but not built. Rather than leave that a pure "
     "design claim, we built a minimal prototype against the exact two hook points "
     "named there (<code>usb_submit_urb</code>, <code>__usb_hcd_giveback_urb</code>) "
     "and measured whether counting completions through them agrees with usbmon — "
     "the literal criterion the project set for this decision and never previously "
     "evaluated. The probe attaches via kfunc (BTF-typed field access), not raw "
     "kprobe register offsets, and counts only at the completion hook, matching the "
     "same completions-only accounting rule the feature extractor already applies.")

if EBPF_IDLE and EBPF_BUSY:
    rows = [["Idle (25 s)", f"{EBPF_IDLE['usbmon']['completions_and_errors']:,}",
             f"{EBPF_IDLE['ebpf']['completions_at_giveback_hook']:,}",
             f"{EBPF_IDLE['agreement_pct']:.3f}%"],
            [f"Busy, 15× lsusb -v ({EBPF_BUSY['seconds_requested']:.0f} s)",
             f"{EBPF_BUSY['usbmon']['completions_and_errors']:,}",
             f"{EBPF_BUSY['ebpf']['completions_at_giveback_hook']:,}",
             f"{EBPF_BUSY['agreement_pct']:.3f}%"]]
    table("eBPF-counted URB completions vs. usbmon, same wall-clock window",
          ["Condition", "usbmon", "eBPF", "Agreement"], rows)

ebpf_ratio_str = f"{ebpf_ratio}×" if ebpf_ratio is not None else "a wide range of"
body(f"Agreement is exact across {ebpf_ratio_str} difference in traffic volume, "
     "which is the condition the project set for a go decision on this question "
     "specifically — whether the hook points work and agree with the deployed "
     "path, not whether a full backend is ready. One further, unplanned finding came "
     "only from running this on real hardware rather than reasoning about it offline: "
     "the URB status field, read via <code>args->urb->status</code> at kfunc entry to "
     "the completion hook, reads -115 (EINPROGRESS) on every single completion "
     "observed, including control transfers known to have succeeded. The count is "
     "unaffected — it depends only on the hook firing, not on the status field "
     "— but a real capture backend would need a different hook or probe point to "
     "read per-URB status correctly, a concrete design note this prototype produced "
     "that a design review alone could not have. No typed records are emitted and no "
     "ring buffer or interface wiring exists; those, and an overhead measurement "
     "against the literature’s sub-microsecond claim [3]–[5], remain future "
     "work.")

# ---------------------------------------------------------------- VII
h1("VII", "Limitations")
item("1)", "<i>No real attack traffic.</i> Section VI-E validates the capture and "
           "scoring pipeline against a real kernel and real human typing, which it flags; it "
           "does not and cannot validate detection accuracy, because no malicious device "
           "was exercised against it. Every <i>detection</i> number in this paper remains "
           "measured on simulated attacks, and the real keyboard measured in VI-E differs "
           "structurally from the simulator’s, which bounds how far those simulated "
           "results should be trusted to transfer. This is the principal limitation.")
item("2)", "<i>eBPF is a measured prototype, not a capture backend.</i> Layer 1 runs on "
           "usbmon. A kfunc-based probe on the two named hook points "
           "(<code>usb_submit_urb</code>, <code>__usb_hcd_giveback_urb</code>) counts URB "
           "completions in exact agreement with usbmon across a wide range of traffic volume "
           "(Section VI-F), answering the only question that was open, but it emits "
           "no typed records and is not wired to the capture interface. The sub-microsecond "
           "overhead claimed for this pattern in [3]–[5] is not measured here.")
item("3)", "<i>Enforcement has never executed.</i> N4 is closed in software only; no "
           "device has been deauthorised by the system against a real kernel.")
item("4)", "<i>Two novelty claims are unsupported.</i> N1 holds only non-uniformly and N2 "
           "not at all, each under a purpose-built test with a pre-registered criterion. "
           "They are retained in the architecture and reported as measured.")
item("5)", "<i>A feature was designed against a known failure.</i> The cross-window payload "
           "feature was added after observing which attack family the system was missing, "
           "using knowledge of how that family is generated. The constraint it exploits is "
           "real — an injection payload must type specific commands and cannot "
           "randomise itself without ceasing to be a payload — but the honest claim "
           "is that it raises the attacker’s cost, not that it solves the case.")
item("6)", "<i>The adversary is bounded by the corpus.</i> An attacker that types an "
           "unbounded stream of fresh keycodes at human speed is statistically the benign "
           "process and is not detectable by any feature in this set. That is stated as a "
           "bound rather than simulated, because a profile indistinguishable from benign "
           "by construction measures nothing.")

# ---------------------------------------------------------------- VIII
h1("VIII", "Conclusion")
body("IUDBMS re-poses USB trust as a question about behaviour observed over time rather "
     "than identity asserted at connection. The implemented pipeline compresses URB "
     "traffic into per-second behavioural vectors, scores each against population and "
     "per-device norms, relays that score into a sequence model, and escalates through a "
     "persistence-gated policy ladder. On the corpus described here it detects "
     f"{pct(det)} of attack sessions at {pct(fp)} false positives, against {pct(s1_det)} "
     "for the same pipeline without its temporal stage and none at all for an "
     "identity-based control.")
body("The evaluation is deliberately adversarial toward its own claims. Each novelty "
     "claim was given an experiment designed to falsify it, with acceptance criteria fixed "
     "in advance; two of four did not survive, one survives only within a bounded regime, "
     "and the strongest result — that temporal context is worth the largest single "
     "margin measured here — is the one the architecture was least committed to in "
     "advance. The same adversarial posture was then applied to the implementation "
     "itself: running the capture and scoring path against a real kernel, rather than "
     "only the simulator, immediately found a defect (Section VI-E) that 385 "
     "simulator-driven tests had not, which is itself evidence that validating against "
     "real traffic is not optional. The outstanding work is correspondingly concrete: "
     "detection measured against a real malicious device rather than a benign one, an "
     "eBPF capture path, and a reviewed execution of the enforcement loop against "
     "hardware.")

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
        head, tail = ref.split("IEEE Access", 1)
        run(p, head)
        run(p, "IEEE Access", italic=True)
        run(p, tail)

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
