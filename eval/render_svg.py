"""
Render the report's charts as inline SVG.

Static rather than scripted: this document is read on screen and printed to PDF
for a submission, and a chart that needs JavaScript is a chart that is blank in
the PDF. Every value is therefore direct-labelled or present in the adjacent
table, so nothing is reachable only by hover.

Colours come from CSS custom properties defined by the page, so the charts
follow the reader's theme instead of carrying a baked-in white background.
Palette slots are the validated categorical 1/2/3.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["render_all"]

S1, S2, S3 = "var(--c1)", "var(--c2)", "var(--c3)"
GRID, AXIS, TXT, MUT = "var(--grid)", "var(--axis)", "var(--tx)", "var(--mut)"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _open(w, h, label):
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{_esc(label)}" xmlns="http://www.w3.org/2000/svg">')


def _txt(x, y, s, size=11, fill=MUT, anchor="middle", weight="400", cls=""):
    c = f' class="{cls}"' if cls else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{c}>{_esc(s)}</text>')


def _legend(items, x, y, gap=150):
    out = []
    for i, (label, colour) in enumerate(items):
        cx = x + i * gap
        out.append(f'<rect x="{cx}" y="{y - 5}" width="15" height="3" rx="1.5" fill="{colour}"/>')
        out.append(_txt(cx + 21, y + 1, label, 11, MUT, "start"))
    return "".join(out)


# ---------------------------------------------------------------- bars
def detection_by_class(d) -> str:
    rows = d["by_class"]
    W, H = 880, 62 * len(rows) + 92
    L, R, T = 176, 78, 54
    pw = W - L - R
    o = [_open(W, H, "Session detection rate by attack class, three defences compared")]
    o.append(_legend([("IUDBMS (S1+S2)", S1), ("Stage 1 only", S2),
                      ("Identity whitelist", S3)], L, 20, 168))
    for gx in (0, 0.25, 0.5, 0.75, 1.0):
        x = L + gx * pw
        o.append(f'<line x1="{x:.1f}" y1="{T - 8}" x2="{x:.1f}" y2="{H - 30}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(x, H - 14, f"{gx * 100:.0f}%", 10.5, MUT, "middle", cls="num"))
    for i, r in enumerate(rows):
        y0 = T + i * 62
        o.append(_txt(L - 12, y0 + 17, r["profile"], 12.5, TXT, "end", "500", "mono"))
        o.append(_txt(L - 12, y0 + 32, f"n={r['n']}", 10.5, MUT, "end", cls="num"))
        for j, (key, colour) in enumerate((("iudbms", S1), ("stage1", S2),
                                           ("whitelist", S3))):
            v = r[key] or 0.0
            by = y0 + j * 13
            w = max(v * pw, 0)
            if w > 0:
                o.append(f'<rect x="{L}" y="{by}" width="{w:.1f}" height="9" rx="3" '
                         f'fill="{colour}"/>')
            o.append(_txt(L + w + 7, by + 8, f"{v * 100:.0f}%", 10.5,
                          TXT if v else MUT, "start", "600" if key == "iudbms" else "400",
                          "num"))
    o.append(_txt(L + pw / 2, H - 1, "session detection rate", 10.5, MUT))
    o.append("</svg>")
    return "".join(o)


def modality_ablation(d) -> str:
    rows = d["modality"]
    fused = d["modality_fused"]
    W, H = 700, 320
    L, R, T, B = 60, 26, 30, 64
    pw, ph = W - L - R, H - T - B
    lo, hi = 0.80, 1.0
    o = [_open(W, H, "Window ROC-AUC for each single modality against all fused")]
    for g in (0.80, 0.85, 0.90, 0.95, 1.0):
        y = T + (1 - (g - lo) / (hi - lo)) * ph
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(L - 9, y + 3.5, f"{g:.2f}", 10.5, MUT, "end", cls="num"))
    bw = pw / len(rows)
    for i, r in enumerate(rows):
        cx = L + bw * (i + 0.5)
        v = max(min(r["auc"], hi), lo)
        y = T + (1 - (v - lo) / (hi - lo)) * ph
        o.append(f'<rect x="{cx - 34}" y="{y:.1f}" width="68" height="{T + ph - y:.1f}" '
                 f'rx="3" fill="{S1}"/>')
        o.append(_txt(cx, y - 7, f"{r['auc']:.4f}", 11, TXT, "middle", "600", "num"))
        o.append(_txt(cx, H - 42, r["name"], 12, TXT, "middle", "500"))
        o.append(_txt(cx, H - 28, f"{r['features']} features", 10.5, MUT))
        o.append(_txt(cx, H - 14, f"{r['detect'] * 100:.1f}% detect", 10.5, MUT, cls="num"))
    fy = T + (1 - (fused["auc"] - lo) / (hi - lo)) * ph
    o.append(f'<line x1="{L}" y1="{fy:.1f}" x2="{W - R}" y2="{fy:.1f}" stroke="{S2}" '
             f'stroke-width="2" stroke-dasharray="7 5"/>')
    o.append(_txt(W - R, fy - 8, f"all fused  {fused['auc']:.4f}", 11, S2, "end", "600"))
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- lines
def roc(d) -> str:
    W, H = 520, 470
    L, R, T, B = 56, 22, 44, 52
    pw, ph = W - L - R, H - T - B
    o = [_open(W, H, "Window-level ROC curve for the full system and Stage 1 alone")]
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = T + (1 - g) * ph
        x = L + g * pw
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + ph}" stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(L - 9, y + 3.5, f"{g:.2f}", 10.5, MUT, "end", cls="num"))
        o.append(_txt(x, T + ph + 17, f"{g:.2f}", 10.5, MUT, cls="num"))
    o.append(f'<line x1="{L}" y1="{T + ph}" x2="{W - R}" y2="{T}" stroke="{MUT}" '
             f'stroke-width="1.5" stroke-dasharray="5 5"/>')
    o.append(_txt(L + pw * 0.62, T + ph * 0.30, "identity-based defence", 10.5, MUT))
    for key, colour, label, auc in (("stage1", S2, "Stage 1 only", d["headline"]["auc_s1"]),
                                    ("iudbms", S1, "IUDBMS", d["headline"]["auc"])):
        c = d["roc"].get(key)
        if not c:
            continue
        pts = " ".join(f"{L + fp * pw:.1f},{T + (1 - tp) * ph:.1f}"
                       for fp, tp in zip(c["fpr"], c["tpr"]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
    o.append(_legend([(f"IUDBMS  AUC {d['headline']['auc']:.4f}", S1)], L, 18, 0))
    o.append(_legend([(f"Stage 1 only  AUC {d['headline']['auc_s1']:.4f}", S2)], L, 34, 0))
    o.append(_txt(L + pw / 2, H - 8, "false positive rate (benign windows)", 10.5, MUT))
    o.append(f'<text x="16" y="{T + ph / 2:.1f}" text-anchor="middle" font-size="10.5" '
             f'fill="{MUT}" transform="rotate(-90 16 {T + ph / 2:.1f})">'
             f'true positive rate</text>')
    o.append("</svg>")
    return "".join(o)


def timeline(tl, threshold) -> str:
    W, H = 880, 260
    L, R, T, B = 52, 20, 34, 50
    pw, ph = W - L - R, H - T - B
    t = tl["t"]
    tmax = max(t) if t else 1
    X = lambda v: L + (v / tmax) * pw
    Y = lambda v: T + (1 - max(0.0, min(1.0, v))) * ph
    o = [_open(W, H, f"Score timeline for a {tl['profile']} device")]

    mal = [tt for tt, lb in zip(t, tl["label"]) if lb == 1]
    if mal:
        o.append(f'<rect x="{X(min(mal)):.1f}" y="{T}" width="{max(X(max(mal)) - X(min(mal)), 1):.1f}" '
                 f'height="{ph}" fill="{MUT}" opacity="0.12"/>')
    for g in (0, 0.5, 1.0):
        y = Y(g)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(L - 9, y + 3.5, f"{g:.1f}", 10.5, MUT, "end", cls="num"))
    step = 20 if tmax > 60 else 10
    for tick in range(0, int(tmax) + 1, step):
        o.append(_txt(X(tick), T + ph + 17, str(tick), 10.5, MUT, cls="num"))

    o.append(f'<line x1="{L}" y1="{Y(threshold):.1f}" x2="{W - R}" y2="{Y(threshold):.1f}" '
             f'stroke="{MUT}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(_txt(W - R, Y(threshold) - 7, f"threshold {threshold:.3f}", 10.5, MUT, "end"))

    for key, colour in (("stage1", S2), ("stage2", S1)):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t, tl[key]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')

    for val, colour, label, dash in ((tl["onset"], S3, "attack onset", "3 3"),
                                     (tl["fired"], S1, "enforcement", "")):
        if val is None:
            continue
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<line x1="{X(val):.1f}" y1="{T - 6}" x2="{X(val):.1f}" y2="{T + ph}" '
                 f'stroke="{colour}" stroke-width="2"{da}/>')
        o.append(_txt(X(val), T - 11, f"{label} {val:.0f}s", 10.5, colour, "middle", "600"))
    o.append(_txt(L + pw / 2, H - 8, "time since enumeration (s)", 10.5, MUT))
    o.append("</svg>")
    return "".join(o)


def novelty(d) -> str:
    W, H = 760, 330
    L, R, T, B = 56, 150, 30, 52
    pw, ph = W - L - R, H - T - B
    o = [_open(W, H, "Payload sequence novelty by window position, three device profiles")]
    xmax = 85
    X = lambda v: L + (v / xmax) * pw
    Y = lambda v: T + (1 - v) * ph
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(L - 9, y + 3.5, f"{g:.2f}", 10.5, MUT, "end", cls="num"))
    for tick in range(0, xmax + 1, 20):
        o.append(_txt(X(tick), T + ph + 17, str(tick), 10.5, MUT, cls="num"))
    labels = {"keyboard": ("benign keyboard", S3),
              "mimicry": ("mimicry (looping script)", S2),
              "mimicry_adaptive": ("mimicry_adaptive (typed once)", S1)}
    for key, (label, colour) in labels.items():
        ser = d["novelty"].get(key)
        if not ser or not ser["x"]:
            continue
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(ser["x"], ser["y"]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
        for a, b in zip(ser["x"], ser["y"]):
            o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="3.2" fill="{colour}"/>')
        ex, ey = ser["x"][-1], ser["y"][-1]
        o.append(f'<circle cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="5" fill="{colour}" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
        o.append(_txt(X(ex) + 11, Y(ey) - 4, label, 11, TXT, "start", "500"))
        o.append(_txt(X(ex) + 11, Y(ey) + 10, f"{ey:.3f}", 10.5, MUT, "start", cls="num"))
    o.append(_txt(L + pw / 2, H - 8, "window position in session (s)", 10.5, MUT))
    o.append(f'<text x="16" y="{T + ph / 2:.1f}" text-anchor="middle" font-size="10.5" '
             f'fill="{MUT}" transform="rotate(-90 16 {T + ph / 2:.1f})">'
             f'payload sequence novelty</text>')
    o.append("</svg>")
    return "".join(o)


def operating(d) -> str:
    rows = d["threshold_sweep"]
    W, H = 700, 330
    L, R, T, B = 56, 130, 30, 56
    pw, ph = W - L - R, H - T - B
    o = [_open(W, H, "Detection and benign false positives against calibration target")]
    n = len(rows)
    X = lambda i: L + (i / max(n - 1, 1)) * pw
    Y = lambda v: T + (1 - v) * ph
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(_txt(L - 9, y + 3.5, f"{g * 100:.0f}%", 10.5, MUT, "end", cls="num"))
    for i, r in enumerate(rows):
        o.append(_txt(X(i), T + ph + 17, f"{r['target_fpr'] * 100:g}%", 10.5, MUT, cls="num"))
    for key, colour, label in (("detect", S1, "attack detection"),
                               ("benign_fp", S2, "benign session FP")):
        pts = " ".join(f"{X(i):.1f},{Y(r[key]):.1f}" for i, r in enumerate(rows))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
        for i, r in enumerate(rows):
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(r[key]):.1f}" r="3.4" fill="{colour}"/>')
        o.append(_txt(X(n - 1) + 11, Y(rows[-1][key]) + 4, label, 11, TXT, "start", "500"))
        o.append(_txt(X(n - 1) + 11, Y(rows[-1][key]) + 18, f"{rows[-1][key] * 100:.1f}%",
                      10.5, MUT, "start", cls="num"))
    o.append(f'<line x1="{X(0):.1f}" y1="{T - 5}" x2="{X(0):.1f}" y2="{T + ph}" '
             f'stroke="{AXIS}" stroke-width="1.5"/>')
    o.append(_txt(X(0), T - 10, "deployed", 10.5, TXT, "middle", "600"))
    o.append(_txt(L + pw / 2, H - 8, "per-window false-positive rate the threshold is calibrated to",
                  10.5, MUT))
    o.append("</svg>")
    return "".join(o)


def render_all(path: Path) -> dict[str, str]:
    d = json.loads(Path(path).read_text())
    thr = d["headline"]["threshold"]
    out = {
        "detection_by_class": detection_by_class(d),
        "modality": modality_ablation(d),
        "roc": roc(d),
        "novelty": novelty(d),
        "operating": operating(d),
    }
    for tl in d["timelines"]:
        out[f"timeline_{tl['profile']}"] = timeline(tl, thr)
    return out
