"""
Experiment 7 -- how brittle is the model to differences between simulation and a real capture?

    python3 -m eval.analysis_robustness

Part A -- the key-overlap assumption behind the replay-loop (``mimicry``) result.
  A device replaying a fixed script presses and releases keys cleanly. A mimic that also
  imitates human *rollover* (overlapping keys) makes every cycle of the script produce
  different reports, which defeats exact-repeat detection. Detection is measured under
  both, because no real attack hardware exists here to say which is realistic.

Part B -- event-level perturbations a real capture has and the simulator lacks. Each is
  applied to the same fresh benign and attack sessions (generated once), and the shipped
  v2 model + 4-of-6 gate is scored on every variant:

    jitter       Gaussian noise on every timestamp (scheduler / VM jitter), sigma in ms
    quantise     deliver events on a polling tick (HID devices are polled at 1-10 ms)
    drop         lose a fraction of events (usbmon ring buffer overflow under load)
    dilate x2    all times doubled (a slower typist/device; benign only -- onset moves)

A model whose false-positive rate or detection moves a lot under a small, plausible
perturbation will not survive contact with a real capture, and this says which ones.
"""

from __future__ import annotations

import dataclasses
import json
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES, BENIGN_PROFILES

from .analysis_common import OUT, deployed, featurize, featurize_multi, run_gate, score_with, wilson
from .analysis_idle_retrain import IDLE_MIX, QUIET_CLASSES
from .analysis_ngram import NGramExtractor
from .analysis_realistic import RealisticSim

MODEL = OUT / "analysis_models" / "realistic_v2"
SEED = 3030
N_BENIGN_PER, N_ATTACK_PER = 40, 30


def _resort(ev):
    return sorted(ev, key=lambda e: e.ts)


def jitter(sigma_ms, seed=0):
    def f(events):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma_ms / 1000.0, len(events))
        return _resort([dataclasses.replace(e, ts=max(0.0, e.ts + dz)) for e, dz in zip(events, noise)])
    return f


def quantise(tick_ms):
    q = tick_ms / 1000.0
    return lambda events: _resort([dataclasses.replace(e, ts=float(np.ceil(e.ts / q) * q)) for e in events])


def drop(p, seed=0):
    def f(events):
        keep = np.random.default_rng(seed).random(len(events)) >= p
        return [e for e, k in zip(events, keep) if k]
    return f


def dilate(factor):
    return lambda events: [dataclasses.replace(e, ts=e.ts * factor) for e in events]


TRANSFORMS = {"as generated": None,
              "jitter 0.5 ms": jitter(0.5), "jitter 2 ms": jitter(2.0), "jitter 5 ms": jitter(5.0),
              "quantise 1 ms": quantise(1.0), "quantise 8 ms": quantise(8.0), "quantise 10 ms": quantise(10.0),
              "drop 1%": drop(0.01), "drop 5%": drop(0.05), "drop 20%": drop(0.20),
              "dilate x2": dilate(2.0)}


def mimicry_sensitivity(cfg, s1, lstm, idx, n=100):
    print("PART A -- replay-loop detection vs the key-overlap assumption "
          f"({n} sessions each, deployed-style 4-of-6 gate)")
    res = {}
    for hold, label in (("short", "clean press/release (a script)"),
                        ("human", "human-like overlapping keys (rollover imitated)")):
        sim = RealisticSim(seed=SEED + 1, mimicry_hold=hold)
        sim._idle, sim._active, sim._fully_idle = (0.0, 0.0), (1e9, 1e9), False
        ss = [featurize(sim.session(f"m{i}", "mimicry"), f"m{i}", "mimicry", NGramExtractor) for i in range(n)]
        S2 = score_with(s1, lstm, ss, idx, cfg["seq_len"])[1]
        first = run_gate(S2, ss, cfg["threshold"], cfg["k"], cfg["n"])
        k = sum(first[s.sid] is not None for s in ss)
        lo, hi = wilson(k, n)
        print(f"  {label:50s} {k:3d}/{n} = {k / n:5.0%}   95% CI {lo:.0%}-{hi:.0%}")
        res[hold] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}
    return res


def main() -> int:
    t0 = time.time()
    cfg, s1, lstm = deployed(MODEL)
    idx = np.array(cfg["feature_idx"])
    print(f"model {MODEL.name}: threshold {cfg['threshold']:.4f}, gate {cfg['k']}-of-{cfg['n']}\n")
    out = {"mimicry_overlap": mimicry_sensitivity(cfg, s1, lstm, idx)}

    # ---- Part B: generate once, featurise under every transform ----------------
    sim = RealisticSim(seed=SEED)
    rng = np.random.default_rng(SEED + 7)
    probs = np.array([m[1] for m in IDLE_MIX])
    plan = sim.plan_balanced(N_ATTACK_PER, benign_profiles=BENIGN_PROFILES,
                             attack_profiles=ATTACK_PROFILES,
                             benign_multiple=1) 
    plan = [(sid, p) for sid, p in plan]
    benign_extra = [(f"X{p[:3]}{i:03d}", p) for p in BENIGN_PROFILES
                    for i in range(N_BENIGN_PER - N_ATTACK_PER)]
    by_t: dict[str, list] = {k: [] for k in TRANSFORMS}
    for sid, profile in plan + benign_extra:
        if profile in QUIET_CLASSES:
            _, _, kw = IDLE_MIX[int(rng.choice(len(IDLE_MIX), p=probs))]
        else:
            kw = IDLE_MIX[0][2]
        sim._idle, sim._active, sim._fully_idle = kw["idle"], kw["active"], kw["fully_idle"]
        s = sim.session(sid, profile)
        for name, sd in featurize_multi(s, sid, profile, {
                k: v for k, v in TRANSFORMS.items() if not (k.startswith("dilate") and s.label == 1)},
                NGramExtractor).items():
            by_t[name].append(sd)
        s.lines = []
    print(f"\nPART B -- {sum(1 for x in by_t['as generated'] if x.label == 0)} benign + "
          f"{sum(1 for x in by_t['as generated'] if x.label == 1)} attack sessions, "
          f"each featurised under {len(TRANSFORMS)} perturbations ({time.time() - t0:.0f} s)\n")
    print(f"  {'perturbation':16s} {'benign FP':>10s} {'FP (keyboards)':>15s} {'detect':>8s}   "
          + " ".join(f"{p[:6]:>7s}" for p in ATTACK_PROFILES))
    out["perturbations"] = {}
    for name, sess in by_t.items():
        S2 = score_with(s1, lstm, sess, idx, cfg["seq_len"])[1]
        first = run_gate(S2, sess, cfg["threshold"], cfg["k"], cfg["n"])
        ben = [s for s in sess if s.label == 0]; kb = [s for s in ben if s.profile == "keyboard"]
        atk = [s for s in sess if s.label == 1]
        fp = np.mean([first[s.sid] is not None for s in ben])
        fpk = np.mean([first[s.sid] is not None for s in kb])
        det = np.mean([first[s.sid] is not None for s in atk]) if atk else float("nan")
        per = {p: (np.mean([first[s.sid] is not None for s in atk if s.profile == p]) if atk else float("nan"))
               for p in ATTACK_PROFILES}
        print(f"  {name:16s} {fp:10.1%} {fpk:15.1%} {det:8.1%}   "
              + " ".join(f"{per[p]:7.0%}" for p in ATTACK_PROFILES))
        out["perturbations"][name] = {"fp": float(fp), "fp_keyboard": float(fpk), "detect": float(det),
                                      "per_profile": {p: float(v) for p, v in per.items()},
                                      "n_benign": len(ben), "n_attack": len(atk)}
    (OUT / "analysis_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote out/analysis_robustness.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
