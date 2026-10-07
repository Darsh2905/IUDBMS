"""
Experiment 0 -- reproduce the deployed headline, put intervals on it, and look
inside every miss.

    python3 -m eval.analysis_baseline

Two things the reported tables do not show:

1. **How wide the headline really is.** 89.2% is 107 of 120 attack sessions, and
   the per-class cells are 15-24 sessions each. A rate over n sessions cannot
   resolve less than 1/n, so every per-class number is quoted here with a Wilson
   interval, and every later experiment is held to the same standard.
2. **Why the misses miss.** For each undetected attack session: how high did
   Stage 2 ever get after onset, and how many windows cleared the threshold? A
   session that peaks at 0.99 but never holds four of six is a *gate* problem; one
   that never leaves 0.4 is a *model* problem. The remedy differs.

Reads the deployed checkpoint; trains nothing; writes out/analysis_baseline.json.
"""

from __future__ import annotations

import json

import numpy as np

from .analysis_common import (OUT, deployed, load_splits, per_profile, run_gate,
                              score_with, session_rates)


def main() -> int:
    _, _, _, test = load_splits()
    cfg, s1, lstm = deployed()
    idx = np.array(cfg["feature_idx"])
    thr = cfg["threshold"]

    S1, S2, _ = score_with(s1, lstm, test, idx, seq_len=cfg["seq_len"])
    first = run_gate(S2, test, thr, cfg["k"], cfg["n"])
    head = session_rates(first, test)
    cls = per_profile(first, test)

    print(f"deployed: threshold {thr:.4f}, gate {cfg['k']}-of-{cfg['n']}, "
          f"{len(idx)} features, seq_len {cfg['seq_len']}")
    lo, hi = head["detect_ci"]
    print(f"\nHEADLINE  detection {head['detect']:.1%}  "
          f"({head['detect_k']}/{head['n_attack']}, 95% CI {lo:.1%}-{hi:.1%})")
    lo, hi = head["fp_ci"]
    print(f"          benign FP {head['fp']:.1%}  "
          f"({head['fp_k']}/{head['n_benign']}, 95% CI {lo:.1%}-{hi:.1%})")
    print(f"          latency median {head['median_latency']:.1f}s  p90 {head['p90_latency']:.1f}s")

    print("\nPER CLASS")
    for p, r in cls.items():
        tag = "attack" if r["label"] else "benign"
        print(f"  {p:18s} {tag}  {r['k']:3d}/{r['n']:<3d} = {r['rate']:6.1%}   "
              f"95% CI {r['ci'][0]:5.1%}-{r['ci'][1]:5.1%}")

    # ---- forensics on every miss -----------------------------------------
    print("\nMISSES: why did the gate not fire?   (post-onset Stage-2, threshold "
          f"{thr:.3f})")
    forensics = []
    for s in test:
        if s.label != 1 or first[s.sid] is not None:
            continue
        post = s.y == 1
        sc = S2[s.sid][post]
        over = int((sc >= thr).sum())
        # best k-of-n run: the most windows over threshold inside any n-window span
        flags = (S2[s.sid] >= thr).astype(int)
        best = max((int(flags[i:i + cfg["n"]].sum())
                    for i in range(max(1, len(flags) - cfg["n"] + 1))), default=0)
        rec = {"sid": s.sid, "profile": s.profile, "post_onset_windows": int(post.sum()),
               "max": float(sc.max()), "p90": float(np.percentile(sc, 90)),
               "windows_over": over, "best_k_in_n": best}
        forensics.append(rec)
    by = {}
    for r in forensics:
        by.setdefault(r["profile"], []).append(r)
    for p, rs in by.items():
        print(f"\n  {p}: {len(rs)} missed")
        print(f"    {'sid':>10s} {'post-win':>8s} {'max':>6s} {'p90':>6s} {'over':>5s} {'best k/n':>9s}")
        for r in rs:
            print(f"    {r['sid']:>10s} {r['post_onset_windows']:8d} {r['max']:6.3f} "
                  f"{r['p90']:6.3f} {r['windows_over']:5d} {r['best_k_in_n']:6d}/{cfg['n']}")
        mx = np.array([r["max"] for r in rs])
        near = int((np.array([r["best_k_in_n"] for r in rs]) >= cfg["k"] - 1).sum())
        print(f"    -> max-score median {np.median(mx):.3f};  "
              f"{near}/{len(rs)} reached {cfg['k'] - 1}-of-{cfg['n']} (one window short)")

    OUT.mkdir(exist_ok=True)
    (OUT / "analysis_baseline.json").write_text(json.dumps(
        {"headline": head, "per_class": cls, "misses": forensics,
         "threshold": thr, "gate": [cfg["k"], cfg["n"]]}, indent=2, default=float))
    print("\nwrote out/analysis_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
