"""
From a window threshold to a session false-alarm rate: why it is measured, not derived.

    python3 -m eval.analysis_session_fp          # ~4 min

The alarm rule has two layers: a window is "over" if its Stage-2 score reaches the
threshold, and a device alarms when 4 of its last 6 windows are over. If windows
were independent with per-window rate p, the chance that a given 6-window span fires
would be about C(6,4) p^4, vanishingly small at p = 1e-4. They are not independent:
consecutive Stage-2 inputs share 9 of their 10 rows, so an over-threshold window is
usually followed by another. This script measures, on long benign sessions:

  p          per-window over-threshold rate
  q          P(window t+1 over | window t over) -- the clustering
  naive      alarms per device-hour predicted by the independence model
  observed   alarms per device-hour the real gate produces

at the deployed threshold and at the per-window threshold it replaced, so the
reader can see exactly how far the independence assumption is off and why the
operating point is calibrated directly on gate firings (eval.analysis_calibration).

Writes out/analysis_session_fp.json.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .analysis_calibration import critical_threshold, long_benign
from .analysis_common import MODELS, OUT, deployed, score_with

SEED = 6007          # its own seed: not the calibration set, not an evaluation set
K, N = 4, 6


def stats(S2: dict, sessions, thr: float) -> dict:
    over_all, pairs, both = [], 0, 0
    alarms = 0
    for s in sessions:
        o = S2[s.sid] >= thr
        over_all.append(o)
        pairs += int(o[:-1].sum())
        both += int((o[:-1] & o[1:]).sum())
        alarms += critical_threshold(S2[s.sid], K, N) >= thr
    o = np.concatenate(over_all)
    hours = sum(s.duration for s in sessions) / 3600.0
    p = float(o.mean())
    # independence: a span of N windows fires with prob ~ sum_{j>=K} C(N,j) p^j (1-p)^(N-j);
    # one new span per window, 3600 windows per hour
    span = sum(math.comb(N, j) * p ** j * (1 - p) ** (N - j) for j in range(K, N + 1))
    return {"threshold": thr, "windows": int(len(o)), "hours": hours,
            "p_window_over": p, "q_next_over_given_over": (both / pairs) if pairs else None,
            "naive_alarms_per_hour": 3600.0 * span, "observed_alarms_per_hour": alarms / hours,
            "observed_alarms": int(alarms)}


def main() -> int:
    cfg, s1, lstm = deployed(MODELS)
    thr = cfg["threshold"]
    prev = [h["threshold"] for h in cfg.get("threshold_history", [])]
    sess = long_benign(SEED, 40, 0)                       # 40 keyboard-hours
    _, S2, _ = score_with(s1, lstm, sess, np.array(cfg["feature_idx"]), cfg["seq_len"])
    out = {"seed": SEED, "rows": [stats(S2, sess, t) for t in prev + [thr]]}
    print(f"40 benign keyboard-hours (seed {SEED})")
    print(f"  {'threshold':>10s} {'p (window)':>11s} {'P(next|over)':>13s} {'naive alarms/h':>15s} "
          f"{'observed /h':>12s}")
    for r in out["rows"]:
        q = r["q_next_over_given_over"]
        print(f"  {r['threshold']:10.6f} {r['p_window_over']:11.2e} "
              f"{(f'{q:.2f}' if q is not None else '-'):>13s} {r['naive_alarms_per_hour']:15.2e} "
              f"{r['observed_alarms_per_hour']:12.3f}")
    (OUT / "analysis_session_fp.json").write_text(json.dumps(out, indent=1))
    print("wrote out/analysis_session_fp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
