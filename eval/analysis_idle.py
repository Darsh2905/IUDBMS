"""
Experiment 1 -- what does the evaluated model do with a device that goes quiet?

    python3 -m eval.analysis_idle

Every benign generator in the simulator is *always active*: a keyboard types for
the whole session with think-pauses of at most 3 s, a mouse always moves, a drive
always does I/O. Real peripherals are idle most of the time. The only attack
profile whose signal is "sustained activity, then permanent silence"
(``mimicry_adaptive``, detected at 40%) therefore has a tell that no benign
device in the corpus ever shows -- and a real keyboard that simply stops being
used shows exactly that tell.

This experiment gives benign devices realistic duty cycles (active bursts
separated by idle gaps of controlled length) and scores them with the evaluated
model. It trains nothing and reads the shipped checkpoint.
"""

from __future__ import annotations

import json

import numpy as np

from .analysis_common import (OUT, deployed, featurize, run_gate, score_with, wilson)
from iudbms.capture.simulator import Simulator


class DutyCycleSim(Simulator):
    """Benign generators that alternate active bursts with idle gaps."""

    def __init__(self, *a, idle=(3.0, 8.0), active=(5.0, 25.0), fully_idle=False, **k):
        super().__init__(*a, **k)
        self._idle, self._active, self._fully_idle = idle, active, fully_idle

    def _cycled(self, parent, t, dur, *args, **kw):
        end = t + dur
        if self._fully_idle:
            return t
        while t < end:
            seg = min(self.rng.uniform(*self._active), end - t)
            t = parent(t, seg, *args, **kw)
            t += self.rng.uniform(*self._idle)
        return t

    def _human_typing(self, t, dur, bus, dev, lines, ep=1):
        return self._cycled(super()._human_typing, t, dur, bus, dev, lines, ep)

    def _mouse_motion(self, t, dur, bus, dev, lines, ep=1):
        return self._cycled(super()._mouse_motion, t, dur, bus, dev, lines, ep)

    def _bulk_io(self, t, dur, bus, dev, lines, ep_in=1, ep_out=2, rate=(10, 60)):
        return self._cycled(super()._bulk_io, t, dur, bus, dev, lines, ep_in, ep_out, rate)

    def _iso_stream(self, t, dur, bus, dev, lines, ep=3, period=0.008, size=3072):
        return self._cycled(super()._iso_stream, t, dur, bus, dev, lines, ep, period, size)


CONDITIONS = [
    ("always active (as in the corpus)", dict(idle=(0.0, 0.0), active=(1e9, 1e9))),
    ("idle gaps 3-8 s",                  dict(idle=(3.0, 8.0))),
    ("idle gaps 8-20 s",                 dict(idle=(8.0, 20.0))),
    ("idle gaps 20-60 s",                dict(idle=(20.0, 60.0))),
    ("fully idle after enumeration",     dict(fully_idle=True)),
]
N = 60


def main() -> int:
    cfg, s1, lstm = deployed()
    idx = np.array(cfg["feature_idx"])
    thr = cfg["threshold"]
    print(f"evaluated model, threshold {thr:.4f}, {cfg['k']}-of-{cfg['n']} gate; {N} sessions per cell\n")
    print(f"{'condition':36s} " + "  ".join(f"{p:>22s}" for p in ("keyboard", "mouse", "mass_storage")))
    print(f"{'':36s} " + "  ".join(f"{'session FP  (95% CI)':>22s}" for _ in range(3)))
    out = {}
    for label, kw in CONDITIONS:
        row = []
        for prof in ("keyboard", "mouse", "mass_storage"):
            sim = DutyCycleSim(seed=4242, **kw)
            sessions = [featurize(sim.session(f"i{prof[:2]}{i}", prof), f"i{prof[:2]}{i}", prof)
                        for i in range(N)]
            _, S2, _ = score_with(s1, lstm, sessions, idx, cfg["seq_len"])
            first = run_gate(S2, sessions, thr, cfg["k"], cfg["n"])
            k = sum(first[s.sid] is not None for s in sessions)
            lo, hi = wilson(k, N)
            frac_windows = float(np.mean(np.concatenate([S2[s.sid] for s in sessions]) >= thr))
            row.append(f"{k / N:6.1%} ({lo:4.0%}-{hi:4.0%})")
            out.setdefault(label, {})[prof] = {"fp": k / N, "k": k, "n": N, "ci": [lo, hi],
                                               "windows_over_threshold": frac_windows}
        print(f"{label:36s} " + "  ".join(f"{c:>22s}" for c in row))
    OUT.mkdir(exist_ok=True)
    (OUT / "analysis_idle.json").write_text(json.dumps(out, indent=2))
    print("\nwrote out/analysis_idle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
