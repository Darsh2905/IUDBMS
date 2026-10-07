"""
Experiment 4 -- is the k-of-n gate the best alerting rule, and is its threshold well set?

    python3 -m eval.analysis_gates

Every rule here has the same shape: a running statistic over a device's Stage-2 scores,
and an alarm the first time it reaches a level. That lets them all be calibrated the
*same way* -- the level is the smallest one at which the false-positive rate on a
benign **calibration** half is at most a stated target -- and then compared like for
like on halves the rule never saw:

    benign   1,500 calibration | 1,500 held-out        (fresh seed 2025)
    attack     300 for choosing rule parameters | 300 held-out (stratified by profile)

Rules:

* ``k-of-n``    the deployed gate (k windows over the threshold within n), with the
                threshold re-derived at session level rather than from the 1-in-10,000
                per-window quantile of ~8,000 validation windows, which is an order
                statistic of about one sample.
* ``EWMA``      exponentially-weighted mean of the log-odds of the Stage-2 score.
* ``CUSUM``     one-sided cumulative sum of (log-odds - drift): uses *how far* above
                normal the scores sit, not just whether they cross a line.

Model: out/analysis_models/realistic_v2 (trained on seed 7; this set never touched it).
"""

from __future__ import annotations

import json

import numpy as np

from .analysis_common import OUT, deployed, score_with, wilson
from .analysis_powered import load_or_build

ALPHAS = (0.005, 0.02)        # benign session false-positive targets
LOGIT_CLIP = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


# --- statistics: each returns the running statistic for one score series -------------

def kth_of_n(scores: np.ndarray, k: int, n: int) -> np.ndarray:
    out = np.full(len(scores), -np.inf)
    for t in range(k - 1, len(scores)):
        w = scores[max(0, t - n + 1):t + 1]
        if len(w) >= k:
            out[t] = np.partition(w, len(w) - k)[len(w) - k]      # k-th largest
    return out


def ewma(scores: np.ndarray, lam: float) -> np.ndarray:
    z = logit(scores)
    out = np.empty(len(z))
    acc = z[0]
    for t, v in enumerate(z):
        acc = lam * acc + (1 - lam) * v
        out[t] = acc
    return out


def cusum(scores: np.ndarray, drift: float) -> np.ndarray:
    z = logit(scores)
    out = np.empty(len(z))
    acc = 0.0
    for t, v in enumerate(z):
        acc = max(0.0, acc + v - drift)
        out[t] = acc
    return out


RULES = (
    [("k-of-n", f"{k}/{n}", (lambda s, k=k, n=n: kth_of_n(s, k, n)))
     for k, n in ((1, 1), (2, 3), (3, 6), (4, 6), (5, 6), (8, 10))]
    + [("EWMA", f"lam={lam}", (lambda s, lam=lam: ewma(s, lam))) for lam in (0.5, 0.7, 0.85)]
    + [("CUSUM", f"drift={d}", (lambda s, d=d: cusum(s, d))) for d in (0.0, 2.0, 4.0, 6.0)]
)


def level_for(benign_series_max: np.ndarray, alpha: float) -> float:
    """Smallest level whose benign false-positive rate is <= alpha (alarm iff max >= level)."""
    v = np.sort(benign_series_max)[::-1]
    allowed = int(np.floor(alpha * len(v)))
    return float(np.nextafter(v[allowed], np.inf)) if allowed < len(v) else float("-inf")


def first_cross(stat: np.ndarray, level: float) -> int | None:
    idx = np.flatnonzero(stat >= level)
    return int(idx[0]) if len(idx) else None


def main() -> int:
    sessions, cond = load_or_build()
    cfg, s1, lstm = deployed(OUT / "analysis_models" / "realistic_v2")
    idx = np.array(cfg["feature_idx"])
    S2 = score_with(s1, lstm, sessions, idx, cfg["seq_len"])[1]

    rng = np.random.default_rng(11)
    ben = [s for s in sessions if s.label == 0]
    atk = [s for s in sessions if s.label == 1]
    perm = rng.permutation(len(ben))
    cal, hold = [ben[i] for i in perm[:len(ben) // 2]], [ben[i] for i in perm[len(ben) // 2:]]
    sel, test = [], []
    for p in sorted({s.profile for s in atk}):
        g = [s for s in atk if s.profile == p]
        pi = rng.permutation(len(g))
        sel += [g[i] for i in pi[:len(g) // 2]]
        test += [g[i] for i in pi[len(g) // 2:]]
    print(f"benign {len(cal)} calibration / {len(hold)} held-out;  attack {len(sel)} selection / "
          f"{len(test)} held-out;  model {cfg['threshold']:.4f} (per-window 1e-4)")

    # --- the rule actually deployed, untouched ----------------------------------------
    stat = {s.sid: kth_of_n(S2[s.sid], cfg["k"], cfg["n"]) for s in sessions}
    lvl = cfg["threshold"]
    fp = sum(first_cross(stat[s.sid], lvl) is not None for s in hold)
    det = sum(first_cross(stat[s.sid], lvl) is not None for s in test)
    print(f"\nDEPLOYED rule as shipped (4/6 at {lvl:.4f}):  benign FP {fp}/{len(hold)} = "
          f"{fp / len(hold):.2%} (CI {wilson(fp, len(hold))[0]:.2%}-{wilson(fp, len(hold))[1]:.2%}),  "
          f"detection {det}/{len(test)} = {det / len(test):.1%}")

    results = {"deployed": {"fp": fp / len(hold), "detect": det / len(test)}, "rules": []}
    for alpha in ALPHAS:
        print(f"\n=== benign session FP target {alpha:.1%} ===")
        print(f"  {'rule':8s} {'params':10s} {'level':>8s} | {'FP held-out':>11s} | {'detect(sel)':>11s} "
              f"{'detect(test)':>12s} {'med lat':>8s} | {'mimicry':>8s} {'adaptive':>8s}")
        rows = []
        for name, par, fn in RULES:
            st = {s.sid: fn(S2[s.sid]) for s in sessions}
            level = level_for(np.array([st[s.sid].max() for s in cal]), alpha)
            fp = sum(first_cross(st[s.sid], level) is not None for s in hold)
            d_sel = np.mean([first_cross(st[s.sid], level) is not None for s in sel])
            hits = {s.sid: first_cross(st[s.sid], level) for s in test}
            d_test = np.mean([h is not None for h in hits.values()])
            lat = [hits[s.sid] + 1 - s.onset for s in test if hits[s.sid] is not None]
            by = {p: np.mean([hits[s.sid] is not None for s in test if s.profile == p])
                  for p in ("mimicry", "mimicry_adaptive")}
            print(f"  {name:8s} {par:10s} {level:8.3f} | {fp / len(hold):10.2%} | {d_sel:11.1%} "
                  f"{d_test:12.1%} {np.median(lat) if lat else float('nan'):7.1f}s | "
                  f"{by['mimicry']:8.0%} {by['mimicry_adaptive']:8.0%}")
            rows.append({"rule": name, "params": par, "level": level, "fp": fp / len(hold),
                         "detect_sel": float(d_sel), "detect_test": float(d_test),
                         "median_latency": float(np.median(lat)) if lat else None,
                         "mimicry": float(by["mimicry"]), "adaptive": float(by["mimicry_adaptive"])})
        results["rules"].append({"alpha": alpha, "rows": rows})

    (OUT / "analysis_gates.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nwrote out/analysis_gates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
