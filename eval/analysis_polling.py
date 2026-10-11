"""
Train on injection whose speed is bounded by the device's polling interval.

    python3 -m eval.analysis_polling          # ~25 min from cold (two corpora + train + calibrate)

**Why.** Corpus B injects one report every ~4.5 ms (~390 reports/s), which assumes the
attacking device polls every 1-4 ms. On real kernels (testbed, seed 1) an emulated
keyboard polled every 10 ms injects at ~93 reports/s; the deployed model's Stage-2
score peaked at 0.9997 but held there too briefly, and at the calibrated threshold
it detected 0 of 6 injection-speed episodes on Debian and Fedora (finding 40).

**Change.** Corpus B v3 = Corpus B with ``poll_limited=True``: every injecting device
draws a polling interval from {1, 2, 4, 8, 10} ms (common declared values; the USB
full-speed range is 1-255) and types no faster than one keystroke per two polls.
Nothing else changes: same seed, sessions, benign traffic, features, architecture,
epochs, and the same per-device-hour calibration.

**Pre-registered adoption (fixed before the first run).** Adopt the v3 model as the
deployed set only if ALL hold:
  (a) on the seed-2 testbed attack round (4 distributions, captured before this change
      was designed), the v3 model at its calibrated threshold detects MORE
      injection-speed episodes (hid_injection, staged_delay, identity_switch) than the
      deployed model at its calibrated threshold -- model alone, without the rule;
  (b) real-kernel benign devices alarmed (all testbed runs, deployed path) do not rise;
  (c) on Corpus C (v2 generator, the existing fresh set) detection does not fall by more
      than one session per family (1%) and benign FP does not rise above 1/3000;
  (d) fresh long-session keyboard false alarms stay <= 0.02 per hour.
(a), (b) are checked by testbed.score with --models; this script prints (c), (d).

Writes out/analysis_polling.json and the candidate set to out/analysis_models/polling_v3/.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .analysis_common import MODELS, OUT, deployed, run_gate, save_set, score_with, session_rates
from .analysis_powered import CACHE as C2_CACHE, build as build_c
from .analysis_realistic import build as build_b
from .corpus import Corpus, load_corpus, save_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

B3 = OUT / "analysis_corpus_realistic_v3.npz"
C3 = OUT / "analysis_corpus_powered_v3.npz"
DEST = OUT / "analysis_models" / "polling_v3"
SEED, EPOCHS, TARGET_FPR = 7, 14, 0.0001


def _cached(path, fn):
    meta = path.with_suffix(".cond.json")
    if path.exists() and meta.exists():
        return load_corpus(path).sessions
    t0 = time.time()
    sessions, cond = fn()
    save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), path)
    meta.write_text(json.dumps(cond))
    print(f"  built {path.name}: {len(sessions)} sessions in {time.time() - t0:.0f} s", flush=True)
    return sessions


def per_family(S2, sessions, thr):
    first = run_gate(S2, sessions, thr)
    r = session_rates(first, sessions)
    fam = {}
    for p in sorted({s.profile for s in sessions if s.label == 1}):
        g = [s for s in sessions if s.profile == p]
        fam[p] = sum(first[s.sid] is not None for s in g) / len(g)
    return {"detect": r["detect"], "detect_k": r["detect_k"], "n_attack": r["n_attack"],
            "fp_k": r["fp_k"], "n_benign": r["n_benign"], "per_family": fam}


def main() -> int:
    t0 = time.time()
    print("corpora:", flush=True)
    B = _cached(B3, lambda: build_b(poll_limited=True))
    C_new = _cached(C3, lambda: build_c(poll_limited=True))
    C_old = load_corpus(C2_CACHE).sessions
    fit, val, test = stratified_split(B, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    r = run_configuration(fit, val, test, name="polling_v3", feature_idx=idx, seed=SEED,
                          epochs=EPOCHS, seq_len=10, hidden=32, target_fpr=TARGET_FPR,
                          keep_scores=True, verbose=False)
    save_set(DEST, r, idx, target_fpr=TARGET_FPR, support_from=fit)
    print(f"trained -> {DEST}  per-window threshold {r.threshold:.6f}  [{time.time() - t0:.0f} s]",
          flush=True)

    # same per-device-hour calibration as the deployed set (adopts into DEST only)
    from . import analysis_calibration
    cal_out = OUT / "analysis_calibration_polling_v3.json"
    analysis_calibration.main(["--models", str(DEST), "--adopt", "--out", str(cal_out)])
    cal = json.loads(cal_out.read_text())

    res = {}
    for name, mdir in (("deployed", MODELS), ("polling_v3", DEST)):
        cfg, s1, lstm = deployed(mdir)
        ix = np.array(cfg["feature_idx"])
        res[name] = {"threshold": cfg["threshold"]}
        for cname, C in (("corpus_c_v2", C_old), ("corpus_c_v3", C_new)):
            _, S2, _ = score_with(s1, lstm, C, ix, cfg["seq_len"])
            res[name][cname] = per_family(S2, C, cfg["threshold"])
    print("\n  model        corpus C (v2 generator)            corpus C (poll-limited injection)")
    for name in res:
        a, b = res[name]["corpus_c_v2"], res[name]["corpus_c_v3"]
        print(f"  {name:12s} {a['detect']:6.1%} FP {a['fp_k']}/{a['n_benign']}  "
              f"{b['detect']:6.1%} FP {b['fp_k']}/{b['n_benign']}")
        for p in a["per_family"]:
            print(f"      {p:18s} {a['per_family'][p]:6.1%}   {b['per_family'][p]:6.1%}")
    kbd = cal["fresh_after"]["keyboard"]
    res["polling_v3"]["fresh_long_keyboard"] = kbd
    d_old = res["deployed"]["corpus_c_v2"]
    d_new = res["polling_v3"]["corpus_c_v2"]
    res["criteria_c_d"] = {
        "c_detection": d_new["detect"] >= d_old["detect"] - 0.01,
        "c_fp": d_new["fp_k"] <= 1,
        "d_long_keyboard": kbd["rate"] <= 0.02}
    print(f"\n  (c) Corpus C v2 detection {d_old['detect']:.1%} -> {d_new['detect']:.1%}, FP "
          f"{d_new['fp_k']}/3000: {'PASS' if res['criteria_c_d']['c_detection'] and res['criteria_c_d']['c_fp'] else 'FAIL'}")
    print(f"  (d) fresh long-session keyboard alarms {kbd['rate']:.3f}/h: "
          f"{'PASS' if res['criteria_c_d']['d_long_keyboard'] else 'FAIL'}")
    print("  (a), (b): python3 -m testbed.score --models out/models --models out/analysis_models/polling_v3")
    (OUT / "analysis_polling.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote out/analysis_polling.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
