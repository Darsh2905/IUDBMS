"""
How human is the simulator's typing? Measured against two real populations.

    python3 -m eval.analysis_human_timing      # needs `python3 -m testbed.human_input fetch`

Reviewer point 7 asks for the typing speed, key-release and idle-period choices to be
explained. They were set from literature figures and the HID specification
(eval.analysis_realistic); this script checks them against real people:

  * Aalto 136M Keystrokes -- 1,972 participants, free transcription typing, press
    and release times (Dhakal et al., CHI 2018);
  * CMU keystroke benchmark -- 51 subjects, one password typed 400 times
    (Killourhy & Maxion, DSN 2009);
  * the simulator's keyboard (Corpus B generator, always-active duty so pauses are
    the generator's own), measured from its press/release report stream.

Quantities: inter-key interval (press to next press) median/p10/p90; key hold
median; rollover (next key pressed before this one is released); share of
inter-key gaps over 1 s; and HID reports per second -- peak and sustained (at that
rate in 4 of any 6 consecutive seconds, the alarm gate's own definition).

Writes out/analysis_human_timing.json. The real data stay under data/human/.
"""

from __future__ import annotations

import json

import numpy as np

from .analysis_common import OUT
from .analysis_realistic import RealisticSim


def _rates(times: np.ndarray) -> tuple[int, int]:
    if len(times) == 0:
        return 0, 0
    b = np.bincount((np.asarray(times) // 1.0).astype(int))
    peak = int(b.max())
    sus = max((int(np.sort(b[i:i + 6])[-4]) for i in range(max(1, len(b) - 5))), default=0)
    return peak, sus


def stats_from_strokes(sessions) -> dict:
    iki, hold, over, gaps, n = [], [], 0, 0, 0
    peaks, sus = [], []
    for ks in sessions:
        ks = sorted(ks, key=lambda k: k[0])
        for a, b in zip(ks, ks[1:]):
            d = b[0] - a[0]
            if d <= 0:
                continue
            iki.append(d)
            over += b[0] < a[1]
            gaps += d > 1.0
            n += 1
        hold += [u - d for d, u in ks]
        p, s = _rates(np.array([t for k in ks for t in k]))
        peaks.append(p)
        sus.append(s)
    iki, hold = np.array(iki), np.array(hold)
    return {"sessions": len(sessions), "keystrokes": int(sum(len(k) for k in sessions)),
            "iki_median_ms": 1e3 * float(np.median(iki)),
            "iki_p10_ms": 1e3 * float(np.percentile(iki, 10)),
            "iki_p90_ms": 1e3 * float(np.percentile(iki, 90)),
            "hold_median_ms": 1e3 * float(np.median(hold)),
            "rollover_share": over / max(n, 1), "gaps_over_1s_share": gaps / max(n, 1),
            "reports_per_s_peak_max": int(max(peaks)), "reports_per_s_sustained_max": int(max(sus)),
            "reports_per_s_sustained_p99": float(np.percentile(sus, 99))}


def simulator_strokes(n: int = 200, duration: float = 120.0, seed: int = 5) -> list:
    """Press/release pairs of the simulator's keyboard, read back from its report stream."""
    from iudbms.capture.simulator import SimulatedSource
    sim = RealisticSim(seed=seed)
    out = []
    for i in range(n):
        sim._idle, sim._active, sim._fully_idle = (0.0, 0.0), (1e9, 1e9), False
        s = sim.session(f"k{i}", "keyboard", duration=duration)
        ev = [e for e in SimulatedSource(s).events()
              if e.evt == "C" and e.ttype == "intr" and len(e.data) == 8]
        down: dict[int, float] = {}
        strokes = []
        prev: set[int] = set()
        for e in ev:
            cur = {k for k in e.data[2:] if k}
            for k in cur - prev:
                down[k] = e.ts
            for k in prev - cur:
                if k in down:
                    strokes.append((down.pop(k), e.ts))
            prev = cur
        out.append(strokes)
    return out


def main() -> int:
    from testbed import human_input as hi
    aalto = [[(k.down, k.up) for k in s.strokes] for s in hi.aalto_sessions()]
    cmu = [[(k.down, k.up) for k in s.strokes] for s in hi.cmu_sessions()]
    sim = simulator_strokes()
    res = {"aalto": stats_from_strokes(aalto), "cmu": stats_from_strokes(cmu),
           "simulator": stats_from_strokes(sim)}
    keys = [("iki_median_ms", "inter-key median (ms)", "{:.0f}"),
            ("iki_p10_ms", "inter-key p10 (ms)", "{:.0f}"),
            ("iki_p90_ms", "inter-key p90 (ms)", "{:.0f}"),
            ("hold_median_ms", "key hold median (ms)", "{:.0f}"),
            ("rollover_share", "rollover share", "{:.1%}"),
            ("gaps_over_1s_share", "gaps over 1 s", "{:.1%}"),
            ("reports_per_s_peak_max", "peak reports/s (max)", "{:d}"),
            ("reports_per_s_sustained_max", "sustained reports/s (max)", "{:d}")]
    print(f"  {'':28s}{'Aalto':>10s}{'CMU':>10s}{'simulator':>11s}")
    # Aalto: one session per participant; CMU: 8 sessions for each of 51 subjects
    print(f"  {'sessions (people)':28s}{res['aalto']['sessions']:>4d} ({res['aalto']['sessions']:d})"
          f"{res['cmu']['sessions']:>5d} (51){res['simulator']['sessions']:>6d} (sim)")
    for k, label, f in keys:
        print(f"  {label:28s}" + "".join(f"{f.format(res[s][k]):>{w}s}"
                                         for s, w in (("aalto", 10), ("cmu", 10), ("simulator", 11))))
    (OUT / "analysis_human_timing.json").write_text(json.dumps(res, indent=1))
    print("wrote out/analysis_human_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
