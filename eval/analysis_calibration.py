"""
Calibrate the alarm threshold in false alarms per device-hour, gate included.

    python3 -m eval.analysis_calibration [--models out/models] [--adopt]

**Why.** The deployed threshold is the (1 - 1e-4) quantile of benign Stage-2 window
scores from 40-85 s validation sessions. That is a per-WINDOW false-positive target.
A monitored keyboard produces 3,600 windows an hour, so any model at that per-window
rate expects ~0.36 over-threshold windows per device-hour; they cluster (consecutive
Stage-2 inputs share 9 of 10 rows) and the 4-of-6 gate turns them into the 0.15-0.36
false alarms per keyboard-hour measured on long sessions (eval.analysis_long_sessions,
eval.analysis_robust_features). Retraining with sample-size-robust features did not
change that (analysis_robust_features: not adopted). The operating point was set in
the wrong unit, so set it in the right one.

**Method.** On a CALIBRATION set of long benign sessions (seed 4049; never used for
evaluation) score every window once, then for each candidate threshold run the real
4-of-6 gate and count firings per device-hour. Pick the smallest threshold (highest
detection) whose calibration rate is at or below TARGET. Because the gate fires at most
a handful of times in tens of hours, the rate is a Poisson count, so the choice also
requires the one-sided 95% upper bound to be reported alongside.

**Pre-registered (fixed before the first run).**
  * TARGET = 0.01 false alarms per device-hour (about one per keyboard per 100 hours).
  * The operating point is chosen on the calibration set alone.
  * It is then evaluated ONCE on a fresh long-session set (seed 2029, never used before)
    and on Corpus C (fresh, seed 2025) for detection; the detection cost is reported as
    measured, whatever it is, together with the full threshold curve.
  * ``--adopt`` writes the chosen threshold into the deployed deploy.json (weights
    untouched), recording the previous threshold and how this one was chosen.

Writes out/analysis_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .analysis_common import MODELS, OUT, deployed, featurize, run_gate, score_with, wilson
from .analysis_long_sessions import LATE_ATTACKS, _duty, late_session
from .analysis_powered import load_or_build
from .analysis_realistic import RealisticSim

TARGET = 0.01
CAL_SEED, EVAL_SEED = 4049, 2029
KBD_N, KBD_DUR = 60, 3600.0          # calibration: 60 keyboard-hours
OTHER_N, OTHER_DUR = 10, 1800.0      # + 5 hours of each other benign class
EVAL_KBD_N = 80                      # evaluation: 80 keyboard-hours
LATE_N, LATE_ONSET = 20, 1200.0


def long_benign(seed: int, kbd_n: int, other_n: int) -> list:
    sim, rng = RealisticSim(seed=seed), np.random.default_rng(seed)
    out = []
    for i in range(kbd_n):
        _duty(sim, rng)
        sid = f"K{seed}-{i:03d}"
        out.append(featurize(sim.session(sid, "keyboard", duration=KBD_DUR), sid, "keyboard"))
    for prof in ("mouse", "mass_storage", "webcam", "audio"):
        for i in range(other_n):
            _duty(sim, rng)
            sid = f"O{seed}-{prof}-{i:03d}"
            out.append(featurize(sim.session(sid, prof, duration=OTHER_DUR), sid, prof))
    return out


def poisson_upper(k: int, hours: float, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on a Poisson rate (exact, via chi-square)."""
    # chi2.ppf(conf, 2k+2)/2 without scipy: invert the Poisson CDF by bisection
    lo, hi = 0.0, max(10.0, 10.0 * (k + 1))
    for _ in range(200):
        mu = (lo + hi) / 2
        cdf = sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k + 1))
        lo, hi = (mu, hi) if cdf > 1 - conf else (lo, mu)
    return hi / hours


def critical_threshold(scores: np.ndarray, k: int, n: int) -> float:
    """The largest threshold at which the k-of-n gate fires on this score sequence.

    The gate fires at threshold t iff some span of the last n windows (shorter at the
    start of a session) holds k scores >= t, i.e. iff t <= the span's k-th largest
    score. So the session alarms at t iff t <= max over spans of that order statistic,
    and the whole alarms-vs-threshold curve is a sort, not a gate simulation per point.
    tests/test_calibration.py checks this against PolicyEngine.
    """
    s = np.asarray(scores, dtype=np.float64)
    if len(s) < k:
        return -np.inf
    best = -np.inf
    for m in range(k, min(n, len(s)) + 1):            # spans at the start, length k..n-1
        if m < n:
            best = max(best, float(np.sort(s[:m])[-k]))
    if len(s) >= n:
        win = np.lib.stride_tricks.sliding_window_view(s, n)
        best = max(best, float(np.sort(win, axis=1)[:, -k].max()))
    return best


def alarms_per_hour(S2, sessions, thr, k, n) -> tuple[int, float]:
    first = run_gate(S2, sessions, thr, k, n)
    hours = sum(s.duration for s in sessions) / 3600.0
    return sum(first[s.sid] is not None for s in sessions), hours


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--adopt", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT / "analysis_calibration.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg, s1, lstm = deployed(a.models)
    idx, K, N, thr0 = np.array(cfg["feature_idx"]), cfg["k"], cfg["n"], cfg["threshold"]

    print(f"calibration set (seed {CAL_SEED}) ...", flush=True)
    cal = long_benign(CAL_SEED, KBD_N, OTHER_N)
    _, S2c, _ = score_with(s1, lstm, cal, idx, cfg["seq_len"])
    # Each session alarms at threshold t iff t <= its critical threshold, so the smallest
    # threshold meeting TARGET leaves at most floor(TARGET * hours) sessions above it.
    tau = np.sort([critical_threshold(S2c[s.sid], K, N) for s in cal])[::-1]
    hours = sum(s.duration for s in cal) / 3600.0
    allowed = int(math.floor(TARGET * hours))
    chosen = float(np.nextafter(tau[allowed], 1.0)) if allowed < len(tau) else thr0
    chosen = max(chosen, thr0)        # never LOWER the threshold below the window-level rule
    curve = [{"threshold": float(t), "cal_alarms": int(i + 1), "cal_rate": (i + 1) / hours}
             for i, t in enumerate(tau[:50]) if np.isfinite(t)]
    k_c, h_c = alarms_per_hour(S2c, cal, chosen, K, N)
    k_0, _ = alarms_per_hour(S2c, cal, thr0, K, N)
    print(f"  {h_c:.0f} benign device-hours; current threshold {thr0:.6f}: {k_0} alarms "
          f"({k_0 / h_c:.3f}/h);  chosen {chosen:.6f}: {k_c} alarms ({k_c / h_c:.3f}/h, "
          f"95% upper {poisson_upper(k_c, h_c):.3f}/h)  [{time.time() - t0:.0f} s]", flush=True)

    print(f"fresh evaluation set (seed {EVAL_SEED}) ...", flush=True)
    ev = long_benign(EVAL_SEED, EVAL_KBD_N, OTHER_N)
    sim2, rng2 = RealisticSim(seed=EVAL_SEED + 1), np.random.default_rng(EVAL_SEED + 1)
    late = []
    for prof in LATE_ATTACKS:
        for i in range(LATE_N):
            _duty(sim2, rng2)
            sid = f"L{EVAL_SEED}-{prof}-{i:03d}"
            late.append(featurize(late_session(sim2, sid, prof, LATE_ONSET), sid, prof))
    _, S2e, _ = score_with(s1, lstm, ev, idx, cfg["seq_len"])
    _, S2l, _ = score_with(s1, lstm, late, idx, cfg["seq_len"])
    C, _ = load_or_build()
    _, S2C, _ = score_with(s1, lstm, C, idx, cfg["seq_len"])

    def evaluate(thr: float) -> dict:
        kb = [s for s in ev if s.profile == "keyboard"]
        ot = [s for s in ev if s.profile != "keyboard"]
        kk, hk = alarms_per_hour(S2e, kb, thr, K, N)
        ko, ho = alarms_per_hour(S2e, ot, thr, K, N)
        fl = run_gate(S2l, late, thr, K, N)
        lt = {p: sum(fl[s.sid] is not None and fl[s.sid] > s.onset for s in late if s.profile == p)
              for p in LATE_ATTACKS}
        pre = sum(fl[s.sid] is not None and fl[s.sid] <= s.onset for s in late)
        fC = run_gate(S2C, C, thr, K, N)
        atk = [s for s in C if s.label == 1]
        ben = [s for s in C if s.label == 0]
        det = sum(fC[s.sid] is not None for s in atk)
        fp = sum(fC[s.sid] is not None for s in ben)
        per = {p: sum(fC[s.sid] is not None for s in atk if s.profile == p) /
               sum(1 for s in atk if s.profile == p) for p in sorted({s.profile for s in atk})}
        lat = [fC[s.sid] - s.onset for s in atk if fC[s.sid] is not None]
        return {"threshold": thr,
                "keyboard": {"alarms": kk, "hours": hk, "rate": kk / hk,
                             "upper95": poisson_upper(kk, hk)},
                "other_classes": {"alarms": ko, "hours": ho, "rate": ko / ho},
                "late_attacks_detected": lt, "late_n_each": LATE_N, "late_alarm_before_onset": pre,
                "corpus_c": {"detect": det / len(atk), "detect_k": det, "n_attack": len(atk),
                             "detect_ci": wilson(det, len(atk)), "fp": fp / len(ben), "fp_k": fp,
                             "n_benign": len(ben), "per_profile": per,
                             "median_latency": float(np.median(lat)) if lat else None}}

    before, after = evaluate(thr0), evaluate(chosen)
    # the whole trade-off on the fresh sets, at a grid between the two operating points
    grid = np.unique(np.concatenate([np.linspace(thr0, min(0.99999, chosen + (chosen - thr0)), 9),
                                     [thr0, chosen]]))
    trade = [evaluate(float(t)) for t in grid]
    print("\nFRESH EVALUATION (seed 2029 long sessions; Corpus C seed 2025)")
    print(f"  {'':22s} {'kbd alarms/h':>14s} {'other/h':>9s} {'C detect':>10s} {'C FP':>9s} "
          f"{'latency':>8s}  late (of {LATE_N} each)")
    for name, r in (("current threshold", before), ("calibrated (0.01/h)", after)):
        print(f"  {name:22s} {r['keyboard']['rate']:8.3f} ({r['keyboard']['alarms']:2d}) "
              f"{r['other_classes']['rate']:9.3f} {r['corpus_c']['detect']:10.1%} "
              f"{r['corpus_c']['fp_k']:4d}/{r['corpus_c']['n_benign']} "
              f"{r['corpus_c']['median_latency'] or float('nan'):7.1f}s  "
              + " ".join(f"{p[:6]}:{v}" for p, v in r["late_attacks_detected"].items()))
    print("\n  threshold curve (fresh):  thr  ->  kbd alarms/h,  Corpus-C detection")
    for r in trade:
        print(f"    {r['threshold']:.6f} -> {r['keyboard']['rate']:.3f},  {r['corpus_c']['detect']:.1%}")

    out = {"pre_registered_target_per_device_hour": TARGET, "calibration_seed": CAL_SEED,
           "evaluation_seed": EVAL_SEED, "models": str(a.models), "previous_threshold": thr0,
           "chosen_threshold": chosen,
           "calibration": {"hours": h_c, "alarms_at_previous": k_0, "alarms_at_chosen": k_c,
                           "upper95_at_chosen": poisson_upper(k_c, h_c), "curve": curve},
           "fresh_before": before, "fresh_after": after, "fresh_tradeoff": trade,
           "runtime_s": time.time() - t0}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {a.out}  ({time.time() - t0:.0f} s)")

    if a.adopt:
        p = a.models / "deploy.json"
        d = json.loads(p.read_text())
        d.setdefault("threshold_history", []).append(
            {"threshold": d["threshold"], "rule": f"per-window FPR {d.get('target_fpr')} on "
                                                  f"validation benign windows"})
        d["threshold"] = chosen
        d["threshold_rule"] = (f"smallest threshold with <= {TARGET} gate firings per benign "
                               f"device-hour on {h_c:.0f} h of long simulated sessions "
                               f"(eval.analysis_calibration, seed {CAL_SEED})")
        p.write_text(json.dumps(d, indent=2))
        print(f"adopted: {p} threshold {thr0:.6f} -> {chosen:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
