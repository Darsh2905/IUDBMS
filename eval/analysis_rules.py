"""
Experiment 10 -- what does a handful of hand-written rules achieve, with no learning at all?

    python3 -m eval.analysis_rules

The paper's only baseline is an identity whitelist, which scores 0% by construction, and
its own ablations. Neither answers the question a reviewer asks first: *could a few
lines of code do most of this?* This implements three rules, each justified from the
protocol and fixed BEFORE looking at detection results (no threshold is tuned on
attacks), and scores them exactly as the model is scored: same gate (4 flagged windows
in any 6), same sessions (Corpus C), same false-positive count.

    speed     >= 20 events in the window and mean inter-arrival < 5 ms, on a device that
              declares an HID interface (a first version without the HID condition flagged
              27% of benign sessions and is reported alongside). Below the fastest
              human cadence on a real keyboard: the recorded typing session has a median
              window mean of 45 ms and no window under 5 ms.
    class     >= 25% of the window's events use a transfer type the declared interface
              classes do not use (a keyboard doing bulk transfers).
    late      the device has grown an interface after enumeration AND is sending.

A window is flagged if any rule fires. The rules are reported singly and together, and
OR-ed with the learned model.
"""

from __future__ import annotations

import json

import numpy as np

from iudbms.contract import FEATURE_NAMES

from .analysis_common import OUT, deployed, run_gate, score_with, wilson
from .analysis_powered import load_or_build

F = {n: i for i, n in enumerate(FEATURE_NAMES)}
FAMS = ["hid_injection", "identity_switch", "staged_delay", "class_exfil", "mimicry",
        "mimicry_adaptive"]


def speed_any(X):
    """The rule as first written: any device. Webcams, audio and drives legitimately emit
    events faster than 5 ms, so it flags 27% of benign sessions. Kept in the report as the
    uncorrected version, not hidden."""
    return (X[:, F["n_events"]] >= 20) & (X[:, F["mean_iat"]] < 0.005)


def speed(X):
    """A keystroke-speed rule only makes sense for a device that declares itself an HID."""
    return speed_any(X) & (X[:, F["has_hid"]] == 1)


def klass(X):
    return (X[:, F["n_events"]] >= 5) & (X[:, F["class_behaviour_mismatch"]] >= 0.25)


def late(X):
    return (X[:, F["late_iface_flag"]] == 1) & (X[:, F["n_events"]] >= 5)


def replay(X):
    """EXPLORATORY fourth rule, added after seeing that the three above miss the replay
    loop. The threshold (0.25) was fixed once, not swept. payload_seq_novelty is the
    fraction of this window's report transitions not seen earlier in the session; a
    replayed script drives it toward zero."""
    return (X[:, F["has_hid"]] == 1) & (X[:, F["n_events"]] >= 5) & (X[:, F["payload_seq_novelty"]] < 0.25)


RULES = {"speed": speed, "class": klass, "late": late, "speed_any": speed_any, "replay": replay}


def flags(sessions, which):
    return {s.sid: np.logical_or.reduce([RULES[r](s.X) for r in which]).astype(float)
            for s in sessions}


def rates(first, sessions):
    out = {}
    for fam in FAMS:
        g = [s for s in sessions if s.profile == fam]
        k = sum(first[s.sid] is not None for s in g)
        out[fam] = (k / len(g), wilson(k, len(g)))
    ben = [s for s in sessions if s.label == 0]
    k = sum(first[s.sid] is not None for s in ben)
    out["benign_fp"] = (k / len(ben), wilson(k, len(ben)), k, len(ben))
    out["mean"] = float(np.mean([out[f][0] for f in FAMS]))
    return out


def main() -> int:
    sessions, _ = load_or_build()
    cfg, s1, lstm = deployed(OUT / "analysis_models" / "realistic_v2")
    S2 = score_with(s1, lstm, sessions, np.array(cfg["feature_idx"]), cfg["seq_len"])[1]
    model = run_gate(S2, sessions, cfg["threshold"], cfg["k"], cfg["n"])

    cols = {}
    for label, which in (("speed (any dev.)", ["speed_any"]), ("speed (HID only)", ["speed"]),
                         ("class", ["class"]), ("late", ["late"]),
                         ("all three rules", ["speed", "class", "late"]),
                         ("replay (explor.)", ["replay"]),
                         ("all four rules", ["speed", "class", "late", "replay"])):
        cols[label] = rates(run_gate(flags(sessions, which), sessions, 0.5, 4, 6), sessions)
    cols["learned model"] = rates(model, sessions)
    rule_first = run_gate(flags(sessions, ["speed", "class", "late"]), sessions, 0.5, 4, 6)
    either = {sid: min([t for t in (rule_first[sid], model[sid]) if t is not None], default=None)
              for sid in model}
    cols["rules OR model"] = rates(either, sessions)

    names = list(cols)
    print(f"Corpus C, n=100 per family, 3,000 benign; gate 4-of-6 for every column\n")
    print(f"  {'family':18s}" + "".join(f"{n[:16]:>17s}" for n in names))
    for fam in FAMS:
        print(f"  {fam:18s}" + "".join(f"{cols[n][fam][0]:17.0%}" for n in names))
    print(f"  {'MEAN':18s}" + "".join(f"{cols[n]['mean']:17.1%}" for n in names))
    print(f"  {'benign FP':18s}" + "".join(f"{cols[n]['benign_fp'][0]:17.2%}" for n in names))

    # --- the real typing capture: do the rules false-alarm on it? -------------------
    import glob
    from pathlib import Path

    from iudbms.capture.usbmon import UsbmonSource
    from iudbms.features.extractor import WindowFeatureExtractor, window_events
    from iudbms.live.cli import _load_descriptor_snapshot
    cap_dir = Path.home() / "Downloads" / "iudbms-real-captures"
    cap, desc = sorted(glob.glob(str(cap_dir / "capture-*.txt"))), sorted(glob.glob(str(cap_dir / "descriptors-*.json")))
    real = None
    if cap and desc:
        d = _load_descriptor_snapshot(Path(desc[-1]))["1-4"]
        ev = [e for e in UsbmonSource(path=Path(cap[-1])).events() if e.device_key == "1-4"]
        ex = WindowFeatureExtractor()
        X = np.vstack([ex.extract(b, t0, d) for t0, b in window_events(ev, duration=max(e.ts for e in ev))])
        busy = X[:, F["n_events"]] > 0
        real = {"windows_with_traffic": int(busy.sum()),
                **{r: int(RULES[r](X).sum()) for r in RULES},
                "min_mean_iat_ms": float(1000 * X[busy & (X[:, F['n_events']] >= 20), F["mean_iat"]].min())}
        print(f"\nreal typing capture ({real['windows_with_traffic']} windows with traffic): "
              f"windows flagged by speed {real['speed']}, class {real['class']}, late {real['late']}, replay {real['replay']}; "
              f"fastest window mean inter-arrival {real['min_mean_iat_ms']:.1f} ms")

    out = {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for n, c in cols.items()}
    out["real_capture"] = real
    (OUT / "analysis_rules.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote out/analysis_rules.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
