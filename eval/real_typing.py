"""
What does a real keyboard look like on the wire, and how does the simulator compare?

    python3 -m eval.real_typing CAPTURE.txt [--device 1-4]
    python3 -m eval.real_typing --scores DB [--device 1-4] [--last 140]   # on the host

Two independent readings of the same recording:

1. **The capture** (any machine): report rate, key-release share, payload
   uniqueness and the inter-arrival distribution of a real keyboard's HID
   reports, set beside the constants the simulator uses for "a human typing"
   (``HUMAN_IAT_LOGNORM``, one URB per keystroke, no release report).
2. **The monitor's scores** (on the host that ran it): what the deployed model
   said about those same seconds, read from the database the live service wrote.

Neither is a calibration. A single short session of one person mashing keys
through a virtual (QEMU) keyboard is a stress sample, not a model of typing.
What it can show reliably is *structure*: whether the simulator's events have the
same shape as real ones, which no amount of simulation could tell us. The
simulator column is MEASURED from a simulated session, never asserted.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "src"))

from iudbms.capture.simulator import HUMAN_IAT_LOGNORM, human_keys_per_second  # noqa: E402
from iudbms.capture.usbmon import UsbmonSource  # noqa: E402


def _keyboard_stats(ev, device: str) -> dict:
    kb = [e for e in ev if e.device_key == device and e.evt == "C"
          and e.ttype == "intr" and len(e.data) == 8]
    if len(kb) < 20:
        raise SystemExit(f"only {len(kb)} HID reports from {device}: nothing was typed, "
                         f"or this is the wrong device")
    ts = np.array([e.ts for e in kb])
    iat = np.diff(ts)
    span = float(ts[-1] - ts[0])
    zero = np.array([not any(e.data) for e in kb])
    li = np.log(iat[iat > 0.0005])
    return {
        "reports": len(kb), "span_s": span, "reports_per_s": len(kb) / span,
        "release_share": float(zero.mean()),
        "unique_payload_share": len({e.data for e in kb}) / len(kb),
        "iat_percentiles": {q: float(np.percentile(iat, q)) for q in (1, 10, 50, 90, 99)},
        "iat_lognormal": {"mu": float(li.mean()), "sigma": float(li.std()),
                          "median_s": float(np.exp(li.mean()))},
        "gaps_over_1s": float(np.mean(iat > 1.0)),
    }


def simulated_reference(seed: int = 7, duration: float = 80.0) -> dict:
    """The simulator's own keyboard, measured exactly the way the real one is.

    Measured rather than asserted: an earlier version of this script printed
    "[simulator: ~100%]" for payload distinctness from memory. The simulator
    draws from ~110 distinct reports, so the real figure is 53%, not ~100%.
    """
    from iudbms.capture.simulator import SimulatedSource, Simulator
    s = Simulator(seed=seed).session("K", "keyboard", duration=duration)
    ev = list(SimulatedSource(s).events())
    return _keyboard_stats(ev, s.descriptor.device_key)


def capture_report(path: Path, device: str) -> dict:
    src = UsbmonSource(path=path)
    ev = list(src.events())
    r = _keyboard_stats(ev, device)
    mu, sg = HUMAN_IAT_LOGNORM
    r.update({"lines": src.stats.lines_seen, "unparsed": src.stats.lines_unparsed,
              "loss_rate": src.stats.loss_rate, "device": device,
              "simulator_constants": {"mu": mu, "sigma": sg, "keys_per_s": human_keys_per_second()},
              "simulator_measured": simulated_reference()})
    return r


def print_capture(r: dict) -> None:
    sim = r["simulator_measured"]
    print(f"capture: {r['lines']:,} lines, {r['unparsed']} unparsed ({r['loss_rate']:.4%})")
    print(f"device {r['device']}: {r['reports']:,} HID reports over {r['span_s']:.1f} s\n")
    print(f"  {'':34s}{'real':>10s}{'simulator':>12s}")
    rows = [
        ("reports per second", f"{r['reports_per_s']:.1f}", f"{sim['reports_per_s']:.1f}"),
        ("key-release (all-zero) reports", f"{r['release_share']:.1%}", f"{sim['release_share']:.1%}"),
        ("distinct payloads", f"{r['unique_payload_share']:.1%}", f"{sim['unique_payload_share']:.1%}"),
        ("inter-arrival median (s)", f"{r['iat_percentiles'][50]:.3f}", f"{sim['iat_percentiles'][50]:.3f}"),
        ("inter-arrival p10 (s)", f"{r['iat_percentiles'][10]:.3f}", f"{sim['iat_percentiles'][10]:.3f}"),
        ("inter-arrival p90 (s)", f"{r['iat_percentiles'][90]:.3f}", f"{sim['iat_percentiles'][90]:.3f}"),
        ("log-normal sigma", f"{r['iat_lognormal']['sigma']:.2f}", f"{sim['iat_lognormal']['sigma']:.2f}"),
        ("gaps longer than 1 s", f"{r['gaps_over_1s']:.1%}", f"{sim['gaps_over_1s']:.1%}"),
    ]
    for name, a, b in rows:
        print(f"  {name:34s}{a:>10s}{b:>12s}")


def score_report(db: Path, device: str, last: int, threshold: float = 0.9916) -> dict:
    cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = cx.execute("SELECT t_start, stage1, stage2, features FROM windows "
                      "WHERE device_id=? ORDER BY id DESC LIMIT ?", (device, last)).fetchall()[::-1]
    busy = [(s1, s2, json.loads(f)) for _, s1, s2, f in rows if json.loads(f)["n_events"] > 0]
    s2 = [x[1] for x in busy]
    return {"windows": len(rows), "busy": len(busy), "threshold": threshold,
            "stage2_min": min(s2), "stage2_median": statistics.median(s2),
            "stage2_max": max(s2), "at_or_above_threshold": sum(1 for x in s2 if x >= threshold),
            "n_events_median": statistics.median(x[2]["n_events"] for x in busy),
            "alerts": cx.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
            "actions": cx.execute("SELECT COUNT(*) FROM actions").fetchone()[0]}


def rescore_report(capture: Path, descriptors: Path, device: str, models: Path) -> dict:
    """Score the recorded capture offline with an explicit model directory.

    Reading scores back from the service's database reports whatever model the
    service happened to be running. That is how a Stage-2 checkpoint that could
    not fire at all (0 of 120 simulated attacks detected) produced a clean
    "no false alarm on real typing" -- it could not alarm on anything. Rescoring
    the capture with named weights makes the result a property of those weights,
    and the offline path is bit-identical to the streaming one
    (tests/test_live.py::test_streaming_matches_offline_scoring).
    """
    import numpy as np

    from eval.analysis_common import deployed, run_gate, score_with
    from eval.corpus import SessionData
    from iudbms.features.extractor import WindowFeatureExtractor, window_events
    from iudbms.live.cli import _load_descriptor_snapshot

    desc = _load_descriptor_snapshot(descriptors)[device]
    ev = [e for e in UsbmonSource(path=capture).events() if e.device_key == device]
    ex, X, T = WindowFeatureExtractor(), [], []
    dur = max(e.ts for e in ev)
    for t0, b in window_events(ev, duration=dur):
        X.append(ex.extract(b, t0, desc))
        T.append(t0)
    X, T = np.vstack(X), np.array(T)
    cfg, s1, lstm = deployed(models)
    sess = SessionData(sid="real", profile="keyboard", label=0, descriptor=desc, X=X, t=T,
                       y=np.zeros(len(T), int), onset=None, duration=dur)
    _, S2, _ = score_with(s1, lstm, [sess], np.array(cfg["feature_idx"]), cfg["seq_len"])
    sc = S2["real"]
    busy = X[:, 0] > 0                                  # column 0 is n_events
    thr = cfg["threshold"]
    seq = X[busy, 22] if X.shape[1] > 22 else np.array([np.nan])   # payload_seq_novelty
    first = run_gate({"real": sc}, [sess], thr, cfg["k"], cfg["n"])
    return {"windows": len(sc), "busy": int(busy.sum()), "threshold": thr,
            "stage2_min": float(sc[busy].min()), "stage2_median": float(np.median(sc[busy])),
            "stage2_max": float(sc[busy].max()),
            "at_or_above_threshold": int((sc[busy] >= thr).sum()),
            "gate_fires": first["real"] is not None,
            "first_alert_s": first["real"],
            "seq_novelty_median": float(np.median(seq)),
            "n_events_median": float(np.median(X[busy, 0])),
            "alerts": int(first["real"] is not None), "actions": int(first["real"] is not None),
            "source": f"offline rescoring with {models}"}


def print_scores(r: dict) -> None:
    print(f"monitor scores: {r['busy']} windows with traffic (of {r['windows']} read)")
    print(f"  Stage 2  min {r['stage2_min']:.3f}   median {r['stage2_median']:.3f}   "
          f"max {r['stage2_max']:.3f}   threshold {r['threshold']}")
    print(f"  windows at/above threshold: {r['at_or_above_threshold']} of {r['busy']}   "
          f"(median {r['n_events_median']:.0f} events per window)")
    print(f"  alerts {r['alerts']}   actions {r['actions']}")
    if "gate_fires" in r:
        print(f"  4-of-6 gate fires: {r['gate_fires']}"
              + (f" (first alert at {r['first_alert_s']:.0f} s)" if r["gate_fires"] else "")
              + f"   payload_seq_novelty median {r['seq_novelty_median']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", type=Path)
    ap.add_argument("--scores", type=Path, metavar="DB",
                    help="read scores from a service database (reflects whichever model "
                         "the service ran -- prefer --rescore)")
    ap.add_argument("--models", type=Path, default=ROOT / "out" / "models",
                    help="model directory for --rescore (default: the deployment's)")
    ap.add_argument("--rescore", type=Path, metavar="DESCRIPTORS",
                    help="score the capture offline with out/models; needs the "
                         "descriptors JSON recorded beside it")
    ap.add_argument("--device", default="1-4")
    ap.add_argument("--last", type=int, default=140)
    ap.add_argument("--json", action="store_true",
                    help="also write out/real_typing.json for scripts/build_paper.py")
    a = ap.parse_args()
    if a.capture is None and a.scores is None:
        ap.error("give a capture file, --scores DB, or both")
    if a.rescore is not None and a.capture is None:
        ap.error("--rescore needs the capture file")

    result = {"capture_path": str(a.capture) if a.capture else None,
             "scores_db": str(a.scores) if a.scores else None, "device": a.device}
    if a.capture is not None:
        cap = capture_report(a.capture, a.device)
        print_capture(cap)
        result["capture"] = cap
    if a.rescore is not None:
        print()
        sc = rescore_report(a.capture, a.rescore, a.device, a.models)
        print_scores(sc)
        result["scores"] = sc
    elif a.scores is not None:
        if a.capture is not None:
            print()
        sc = score_report(a.scores, a.device, a.last)
        print_scores(sc)
        result["scores"] = sc

    if a.json:
        OUT.mkdir(exist_ok=True)
        (OUT / "real_typing.json").write_text(json.dumps(result, indent=2, default=float))
        print(f"\n  wrote out/real_typing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
