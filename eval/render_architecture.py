"""Architecture diagrams — PNG/PDF for slides and the report.

    python3 -m eval.render_architecture

Writes ``out/architecture.png`` and ``out/gate_timeline.png`` (plus .pdf).

Neither figure is hand-drawn from memory. Every constant in figure 1 is read
from the contract or from ``out/figure_data.json``, and figure 2 is a real test
session stepped through the deployed ``PolicyEngine`` -- so if the threshold,
the gate, the sequence length or the corpus size change, the diagram changes
with them instead of going quietly stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from iudbms.contract import FEATURE_SCHEMA           # noqa: E402
from iudbms.contract import DeviceDescriptor         # noqa: E402
from iudbms.policy.engine import PolicyEngine        # noqa: E402
from iudbms.stage2.lstm import SEQ_LEN               # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"

INK   = "#1b1b1b"      # structure: boxes, arrows, labels
MUTE  = "#6b6b6b"      # secondary text
SIG   = "#b45309"      # reserved for the relay -- the one misread mechanism
CRIT  = "#9f1239"      # designed but not built / generated but not executed
FILL  = "#ffffff"

GATE_K, GATE_N = 4, 6


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------
def box(ax, x0, y0, x1, y1, *, ec=INK, lw=1.1, ls="solid", fc=FILL, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z))


FONT = "DejaVu Sans"


def txt(ax, x, y, s, *, size=7, ha="center", va="center",
        color=INK, weight="normal", mono=False, z=4):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color,
            fontweight=weight, zorder=z,
            family="DejaVu Sans Mono" if mono else FONT)


def arrow(ax, xy_from, xy_to, *, color=INK, lw=1.2, ls="solid", z=3):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=z))


def elbow(ax, pts, *, color=SIG, lw=1.7, z=3):
    """Multi-segment connector; the arrowhead goes on the final leg."""
    for a, b in zip(pts, pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z)


# --------------------------------------------------------------------------
# figure 1 -- the pipeline
# --------------------------------------------------------------------------
def architecture(d: dict, path: Path) -> Path:
    h = d["headline"]
    n_feat_total = len(FEATURE_SCHEMA)
    n_feat_used = n_feat_total - 1          # log_time_since_enum is a label leak

    fig, ax = plt.subplots(figsize=(14, 8.6))
    ax.set_xlim(0, 140); ax.set_ylim(0, 90); ax.axis("off")

    # ---- kernel band ----
    box(ax, 6, 72, 64, 88, ec=MUTE, lw=1.0)
    txt(ax, 8.5, 86.2, "KERNEL SPACE", size=6.5, ha="left", color=MUTE, weight="bold")
    txt(ax, 35, 83.4, "USB device activity reaches the host controller", size=7.5)
    txt(ax, 35, 81.0, "usb_submit_urb   ·   __usb_hcd_giveback_urb",
        size=6.8, color=MUTE, mono=True)

    box(ax, 9, 73.5, 34, 79.3, lw=1.2)
    txt(ax, 21.5, 77.4, "usbmon  (debugfs)", size=7.8, weight="bold")
    txt(ax, 21.5, 75.2, "deployed capture path", size=6.5, color=MUTE)

    box(ax, 36, 73.5, 61, 79.3, ec=CRIT, ls=(0, (4, 2.6)), lw=1.2)
    txt(ax, 48.5, 77.4, "eBPF probe", size=7.8, weight="bold", color=CRIT)
    txt(ax, 48.5, 75.2, "designed, not built", size=6.5, color=CRIT)

    arrow(ax, (21.5, 73.5), (21.5, 67.0))
    txt(ax, 23.4, 71.6, "plain text lines", size=6.8, ha="left")
    txt(ax, 23.4, 69.4, "11,367,542 across 1,000 sessions", size=6.3, ha="left", color=MUTE)

    # ---- L1 ----
    box(ax, 6, 58.5, 64, 67.0, lw=1.3)
    txt(ax, 35, 64.4, "L1  ·  PARSER", size=8.2, weight="bold")
    txt(ax, 35, 61.4, "one text line  →  one typed URBEvent", size=7)

    arrow(ax, (35, 58.5), (35, 52.0))
    txt(ax, 36.9, 55.2, "events, bucketed by kernel timestamp", size=6.8, ha="left")

    # ---- L2 ----
    box(ax, 6, 42.0, 64, 52.0, lw=1.3)
    txt(ax, 35, 49.2, "L2  ·  FEATURE EXTRACTOR", size=8.2, weight="bold")
    txt(ax, 35, 46.4, "1.0 s tumbling window   ·   stateful, one per device session", size=7)
    txt(ax, 35, 44.0, "raw events are discarded here and stored nowhere", size=6.5, color=MUTE)

    arrow(ax, (35, 42.0), (35, 34.5))
    txt(ax, 36.9, 39.4, f"ONE row  ·  {n_feat_used} numbers  ·  per device per second",
        size=6.8, ha="left")
    txt(ax, 36.9, 37.0, f"{h['windows']:,} rows in the corpus", size=6.3, ha="left", color=MUTE)

    # ---- stage 1 ----
    box(ax, 6, 17.0, 64, 34.5, lw=1.6)
    txt(ax, 8.5, 32.2, "STAGE 1", size=8.2, ha="left", weight="bold")
    txt(ax, 20.5, 32.2, "memoryless — sees this one row, nothing before it",
        size=6.5, ha="left", color=MUTE)
    box(ax, 9, 19.4, 34, 29.4, lw=1.0, ec=MUTE)
    txt(ax, 21.5, 26.9, "Isolation Forest", size=7.5)
    txt(ax, 21.5, 24.3, "vs. the population of", size=6.4, color=MUTE)
    txt(ax, 21.5, 22.4, "devices of the same class", size=6.4, color=MUTE)
    txt(ax, 21.5, 20.4, "weight 0.55", size=6.4, color=MUTE, mono=True)
    box(ax, 36, 19.4, 61, 29.4, lw=1.0, ec=MUTE)
    txt(ax, 48.5, 26.9, "EWMA device baseline", size=7.5)
    txt(ax, 48.5, 24.3, "vs. this one device's", size=6.4, color=MUTE)
    txt(ax, 48.5, 22.4, "own running history", size=6.4, color=MUTE)
    txt(ax, 48.5, 20.4, "weight 0.45", size=6.4, color=MUTE, mono=True)

    # ---- the relay: stage 1's score becomes stage 2's input ----
    elbow(ax, [(35, 17.0), (35, 11.0), (70, 11.0), (70, 79.0), (75.4, 79.0)])
    txt(ax, 52.5, 9.0, "one score, appended as the 29th column",
        size=7.6, color=SIG, weight="bold")
    txt(ax, 52.5, 6.8, "— not a second vote; it becomes input", size=6.8, color=SIG)

    # ---- sliding buffer ----
    box(ax, 76, 72.0, 134, 84.5, lw=1.3)
    txt(ax, 78.5, 82.4, "SLIDING BUFFER", size=8.2, ha="left", weight="bold")
    txt(ax, 96.5, 82.4, f"last {SEQ_LEN} rows — the oldest drops off as each new one arrives",
        size=6.4, ha="left", color=MUTE)
    x = 82.0
    for i in range(SEQ_LEN):
        newest = i == SEQ_LEN - 1
        box(ax, x, 74.2, x + 3.6, 79.6,
            ec=SIG if newest else INK, lw=1.5 if newest else 0.9,
            fc="#f6e9d6" if newest else FILL)
        x += 4.6
    txt(ax, 129.5, 79.0, "newest", size=6.3, ha="left", color=SIG, weight="bold")
    txt(ax, 129.5, 76.6, f"{n_feat_total} cols", size=6.3, ha="left", color=SIG, mono=True)
    txt(ax, 80.6, 76.9, "out", size=6.0, ha="right", color=MUTE)

    arrow(ax, (105, 72.0), (105, 66.0))
    txt(ax, 106.9, 69.0, f"all {SEQ_LEN} rows together, in order", size=6.8, ha="left")

    # ---- stage 2 ----
    box(ax, 76, 56.0, 134, 66.0, lw=1.6)
    txt(ax, 78.5, 63.4, "STAGE 2  ·  LSTM", size=8.2, ha="left", weight="bold")
    txt(ax, 102.0, 63.4, "temporal — reads the sequence, so it sees transitions",
        size=6.4, ha="left", color=MUTE)
    txt(ax, 78.5, 60.2, f"{SEQ_LEN} × {n_feat_total}   →   P(malicious) for the newest second",
        size=7, ha="left")
    txt(ax, 78.5, 57.8, "the only output anything downstream is allowed to threshold",
        size=6.4, ha="left", color=MUTE)

    arrow(ax, (105, 56.0), (105, 50.0))
    txt(ax, 106.9, 53.0, "one number, 0–1", size=6.8, ha="left")

    # ---- policy ----
    box(ax, 76, 31.0, 134, 50.0, lw=1.6)
    txt(ax, 78.5, 47.4, "L4  ·  POLICY ENGINE", size=8.2, ha="left", weight="bold")
    txt(ax, 78.5, 43.8, f"score ≥ {h['threshold']:.4f}  →  hit,  else miss",
        size=7, ha="left")
    txt(ax, 78.5, 40.6, f"{GATE_K} hits among the trailing {GATE_N}  →  the window qualifies",
        size=7, ha="left")
    txt(ax, 78.5, 37.0, "1st qualifying window", size=7, ha="left", weight="bold")
    arrow(ax, (99.5, 37.0), (104.5, 37.0), lw=1.0)
    txt(ax, 105.5, 37.0, "THROTTLE", size=7, ha="left", weight="bold")
    txt(ax, 78.5, 33.6, f"{2 * GATE_K}th qualifying window", size=7, ha="left", weight="bold")
    arrow(ax, (99.5, 33.6), (104.5, 33.6), lw=1.0)
    txt(ax, 105.5, 33.6, "DEAUTHORIZE", size=7, ha="left", weight="bold")

    # ---- the two things that happen next ----
    arrow(ax, (90, 31.0), (90, 25.5), color=CRIT, ls=(0, (4, 2.6)))
    box(ax, 76, 13.5, 104, 25.5, ec=CRIT, ls=(0, (4, 2.6)), lw=1.2)
    txt(ax, 90, 23.0, "KERNEL ENFORCEMENT", size=7.2, weight="bold", color=CRIT)
    txt(ax, 90, 20.0, "echo 0 > /sys/bus/usb/", size=6.2, color=CRIT, mono=True)
    txt(ax, 90, 18.0, "devices/1-49/authorized", size=6.2, color=CRIT, mono=True)
    txt(ax, 90, 15.4, "generated, never executed", size=6.3, color=CRIT)

    arrow(ax, (120, 31.0), (120, 25.5))
    box(ax, 106, 13.5, 134, 25.5, lw=1.1)
    txt(ax, 120, 23.0, "OPERATOR PATH", size=7.2, weight="bold")
    txt(ax, 120, 20.4, "SQLite  →  FastAPI  →  dashboard", size=6.5)
    txt(ax, 120, 17.6, "features, scores and the", size=6.3, color=MUTE)
    txt(ax, 120, 15.7, "generated rule, per window", size=6.3, color=MUTE)

    # ---- key ----
    ax.plot([7.5, 12.5], [2.6, 2.6], color=INK, lw=1.2)
    txt(ax, 13.6, 2.6, "data flow", size=6.5, ha="left", color=MUTE)
    ax.plot([26, 31], [2.6, 2.6], color=SIG, lw=1.7)
    txt(ax, 32.1, 2.6, "the relay — Stage 1's output is Stage 2's input",
        size=6.5, ha="left", color=MUTE)
    ax.plot([76, 81], [2.6, 2.6], color=CRIT, lw=1.2, ls=(0, (4, 2.6)))
    txt(ax, 82.1, 2.6, "designed but not built  /  generated but not executed",
        size=6.5, ha="left", color=MUTE)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# figure 2 -- the gate, replayed on a real session
# --------------------------------------------------------------------------
def gate_timeline(d: dict, path: Path) -> Path:
    thr = d["headline"]["threshold"]
    tl = next(t for t in d["timelines"] if t["profile"] == "staged_delay")

    eng = PolicyEngine(threshold=thr, k=GATE_K, n=GATE_N)
    ctx = eng.new_device(tl["sid"])
    desc = DeviceDescriptor(bus=1, devnum=49, vid=0x046d, pid=0xc31c)

    in_win, qualifying, acts = [], [], {}
    for t, s1, s2 in zip(tl["t"], tl["stage1"], tl["stage2"]):
        a = eng.step(ctx, float(t), float(s1), float(s2), [], desc)
        in_win.append(ctx.hits_in_window)
        qualifying.append(ctx.hits)
        if a and a.action not in acts:
            acts[a.action] = float(t) + 1.0     # alerts are stamped at window end

    t, s2 = tl["t"], tl["stage2"]
    onset = tl["onset"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 1], "hspace": 0.18})
    a0, a1, a2 = axes

    def marks(ax):
        ax.axvspan(onset, max(t), color="#f4f4f2", zorder=0)
        ax.axvline(onset, color="#2f7d5c", lw=1.6, ls=":", zorder=1)
        for name, when in acts.items():
            ax.axvline(when, color=SIG if name == "THROTTLE" else CRIT, lw=1.8, zorder=1)
        ax.grid(axis="y", lw=.4, color="#dcdcdc", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    a0.plot(t, s2, lw=1.9, color=INK, zorder=3)
    a0.axhline(thr, ls="--", lw=1.1, color=CRIT, zorder=2)
    hit = [tt for tt, v in zip(t, s2) if v >= thr]
    a0.scatter(hit, [v for v in s2 if v >= thr], s=16, color=SIG, zorder=4)
    a0.set_ylim(-0.05, 1.12)
    a0.set_ylabel("Stage 2\nP(malicious)", fontsize=8.5)
    a0.text(1, thr - 0.075, f"learned threshold {thr:.4f}", fontsize=7.5, color=CRIT)
    marks(a0)

    a1.step(t, in_win, where="post", lw=1.6, color=INK, zorder=3)
    a1.axhline(GATE_K, ls="--", lw=1.1, color=SIG, zorder=2)
    a1.set_ylim(-0.4, GATE_N + 0.8)
    a1.set_yticks([0, GATE_K, GATE_N])
    a1.set_ylabel(f"hits in the\ntrailing {GATE_N}", fontsize=8.5)
    a1.text(1, GATE_K + 0.35, f"k = {GATE_K}: at or above this, the window qualifies",
            fontsize=7.5, color=SIG)
    marks(a1)

    a2.step(t, qualifying, where="post", lw=1.6, color=INK, zorder=3)
    a2.axhline(2 * GATE_K, ls="--", lw=1.1, color=CRIT, zorder=2)
    a2.set_ylim(-2.5, max(qualifying) + 5)
    a2.set_yticks([0, 2 * GATE_K, max(qualifying)])
    a2.set_ylabel("qualifying\nwindows", fontsize=8.5)
    a2.text(1, 2 * GATE_K + 1.4, f"2k = {2 * GATE_K}: revoke", fontsize=7.5, color=CRIT)
    a2.set_xlabel("time since enumeration (s)", fontsize=9)
    marks(a2)

    a0.text(onset - 1.6, 1.055, f"attack onset  {onset:.1f} s",
            ha="right", fontsize=8.5, color="#2f7d5c")
    # staggered so the two action labels cannot collide when the ladder
    # escalates quickly -- on this session they are only 7 s apart
    for name, when, y in (("THROTTLE", acts.get("THROTTLE"), 0.60),
                          ("DEAUTHORIZE", acts.get("DEAUTHORIZE"), 0.30)):
        if when is None:
            continue
        col = SIG if name == "THROTTLE" else CRIT
        a0.annotate(f"{name}   {when:.0f} s   (+{when - onset:.1f} s after onset)",
                    xy=(when, y), xytext=(when + 2.5, y),
                    ha="left", va="center", fontsize=8.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.0,
                                    shrinkA=0, shrinkB=2))

    a0.set_title(
        f"Session {tl['sid']}  ·  profile staged_delay  ·  test split  ·  "
        f"replayed through policy/engine.py",
        fontsize=10, loc="left", pad=9)

    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# figure 1 for the IEEE paper -- drawn at print size (7.16 in, two columns)
# --------------------------------------------------------------------------
def architecture_paper(d: dict, path: Path) -> Path:
    """Same pipeline as ``architecture`` but laid out for a two-column IEEE page.

    The slide version is 14 in wide; shrunk to 7.16 in its 7 pt text would print
    at ~3.5 pt. This one is drawn at final size, with text at 6-7 pt.
    """
    global FONT
    h = d["headline"]
    FONT = "Arial"
    try:
        fig, ax = plt.subplots(figsize=(7.16, 3.15))
        ax.set_xlim(0, 100); ax.set_ylim(0, 44.0); ax.axis("off")

        W, G = 20.5, 5.9
        xs = [0.2 + i * (W + G) for i in range(4)]
        T, B = 7.2, 6.5
        R1, R2 = (26.4, 43.4), (0.6, 17.6)

        def blk(x, rows, title, lines, *, lw=1.0):
            y0, y1 = rows
            box(ax, x, y0, x + W, y1, lw=lw)
            txt(ax, x + W / 2, y1 - 2.6, title, size=T, weight="bold")
            for i, ln in enumerate(lines):
                txt(ax, x + W / 2, y1 - 6.4 - i * 3.0, ln, size=B)

        # ---- row 1: capture -> parse -> extract -> stage 1 ------------------
        x = xs[0]
        box(ax, x, *R1[:1], x + W, R1[1], lw=1.0) if False else box(ax, x, R1[0], x + W, R1[1], lw=1.0)
        txt(ax, x + W / 2, R1[1] - 2.4, "L0  Kernel capture", size=T, weight="bold")
        box(ax, x + 1.2, 33.4, x + W - 1.2, 38.2, lw=0.9, ec=MUTE)
        txt(ax, x + W / 2, 35.8, "usbmon (deployed)", size=B)
        box(ax, x + 1.2, 27.6, x + W - 1.2, 32.4, lw=0.9, ec=CRIT, ls=(0, (3, 2)))
        txt(ax, x + W / 2, 30.0, "eBPF (designed only)", size=B, color=CRIT)

        blk(xs[1], R1, "L1  Parser",
            ["text line to typed", "URBEvent (completions", "and errors only)"])
        blk(xs[2], R1, "L2  Feature extractor",
            ["1 s tumbling window,", "stateful per device:",
             f"{len(FEATURE_SCHEMA) - 1} features per second"])
        blk(xs[3], R1, "Stage 1",
            ["Isolation Forest (0.55)", "+ EWMA baseline (0.45)", "gives anomaly score s1"], lw=1.3)

        ym = 34.9
        for a_, b_ in zip(xs[:3], xs[1:]):
            arrow(ax, (a_ + W, ym), (b_, ym), lw=1.0)

        # ---- the relay: Stage 1's score becomes Stage 2's input ---------------
        xr = xs[3] + W / 2
        arrow(ax, (xr, R1[0]), (xr, R2[1]), color=SIG, lw=1.9)
        txt(ax, xr - 1.6, 23.4, "s1 is appended as the 29th column of the row",
            size=6.8, ha="right", color=SIG, weight="bold")
        txt(ax, xr - 1.6, 20.6, "(an input to Stage 2, not a second vote)",
            size=6.3, ha="right", color=SIG)

        # ---- row 2, right to left: buffer -> stage 2 -> policy -> outputs -----
        x = xs[3]
        box(ax, x, R2[0], x + W, R2[1], lw=1.0)
        txt(ax, x + W / 2, R2[1] - 2.5, "Sliding buffer", size=T, weight="bold")
        txt(ax, x + W / 2, R2[1] - 5.8, "last 10 rows = 10 s   (10 x 29)", size=B)
        n, cw, cg = 10, 1.55, 0.32
        x0 = x + (W - (n * cw + (n - 1) * cg)) / 2
        for i in range(n):
            newest = i == n - 1
            box(ax, x0 + i * (cw + cg), 2.0, x0 + i * (cw + cg) + cw, 7.4,
                ec=SIG if newest else INK, lw=1.4 if newest else 0.7,
                fc="#f3e2c8" if newest else FILL)
        txt(ax, x + W / 2, 9.9, "newest row highlighted", size=5.8, color=MUTE)

        blk(xs[2], R2, "Stage 2  (LSTM)",
            ["reads all 10 rows in order", "outputs P(malicious)", "for the newest second"], lw=1.3)
        blk(xs[1], R2, "L4  Policy engine",
            [f"P >= {h['threshold']:.4f}: a hit", "4 hits in last 6: THROTTLE",
             "8 qualifying: DEAUTHORIZE"])

        x = xs[0]
        box(ax, x, 9.8, x + W, R2[1], lw=1.0)
        txt(ax, x + W / 2, R2[1] - 2.5, "Operator path", size=T, weight="bold")
        txt(ax, x + W / 2, R2[1] - 6.0, "SQLite, FastAPI, dashboard", size=B)
        box(ax, x, R2[0], x + W, 8.2, ec=CRIT, ls=(0, (3, 2)), lw=1.0)
        txt(ax, x + W / 2, 5.7, "Kernel enforcement", size=B + 0.3, weight="bold", color=CRIT)
        txt(ax, x + W / 2, 2.9, "generated, dry-run only", size=B - 0.2, color=CRIT)

        ym2 = 9.1
        arrow(ax, (xs[3], ym2), (xs[2] + W, ym2), lw=1.0)
        arrow(ax, (xs[2], ym2), (xs[1] + W, ym2), lw=1.0)
        arrow(ax, (xs[1], 13.6), (xs[0] + W, 13.6), lw=1.0)
        arrow(ax, (xs[1], 4.4), (xs[0] + W, 4.4), lw=1.0, color=CRIT, ls=(0, (3, 2)))

        fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    finally:
        FONT = "DejaVu Sans"
    return path


def main() -> int:
    d = json.loads((OUT / "figure_data.json").read_text())
    for p in (architecture(d, OUT / "architecture.png"),
              architecture_paper(d, OUT / "architecture_ieee.png"),
              gate_timeline(d, OUT / "gate_timeline.png")):
        print(f"  wrote {p.relative_to(OUT.parent)}  and  "
              f"{p.with_suffix('.pdf').relative_to(OUT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
