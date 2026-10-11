"""
A human-capability bound beside the learned model: does it carry unseen families?

    python3 -m eval.analysis_hybrid          # ~10 min (retrains the six held-out models)

The learned detector generalises poorly to an attack family it never saw (82.7% ->
34.8%, eval.analysis_lopo; the transfer matrix in eval.analysis_transfer says why:
supervised Stage 2 learns family signatures, and families share few). Some attacks,
though, violate a PHYSICAL bound that no training family is needed to state:

    no human sustains HUMAN_MAX_RPS HID reports per second on a keyboard.

The bound is measured, not chosen: over 1,972 real people typing 354,669 active
seconds (Aalto 136M Keystrokes, testbed.human_input), the fastest sustained rate --
at or above x in 4 of any 6 consecutive seconds, the gate's own definition -- is 26
reports/s (p99.9: 25); the fastest single second is 34. HUMAN_MAX_RPS = 30 sits above
every sustained human rate in that population and was fixed BEFORE any attack was
scored with it.

**The rule.** A window is flagged if the device declares a boot-protocol keyboard
interface (class 3, protocol 1) and carries at least HUMAN_MAX_RPS completed URBs.
Flags feed the same 4-of-6 gate as the model's. Report-protocol-only keyboards are
NOT covered by the rule (the learned model still is) -- stated, not hidden.

**Pre-registered adoption criteria (fixed before the first run).** Adopt the hybrid
(model OR rule) as the deployed decision rule only if:
  (i)  the rule alone raises ZERO alarms on every benign set: Corpus C (3,000
       sessions), 80 keyboard-hours of fresh long sessions (seed 2031), and every
       benign keyboard in the real-kernel testbed (scored separately, testbed.score);
  (ii) mean unseen-family detection under leave-one-family-out rises.
Detection on families the rule cannot see (the human-speed mimicry families) is
reported unchanged, as a bound of the rule rather than hidden.

Writes out/analysis_hybrid.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES

from .analysis_common import MODELS, OUT, deployed, featurize, run_gate, score_with, wilson
from .analysis_long_sessions import _duty
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE, RealisticSim
from .corpus import load_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

HUMAN_MAX_RPS = 30
N_EVENTS = 0                     # contract column 0
SEED, EPOCHS = 7, 14
LONG_SEED, LONG_N, LONG_DUR = 2031, 80, 3600.0


def boot_keyboard(descriptor) -> bool:
    """policy.support.is_keyboard: the report descriptor's Keyboard page when known,
    else the boot-keyboard protocol (every simulated keyboard)."""
    from iudbms.policy import is_keyboard
    return is_keyboard(descriptor)


def rule_scores(sessions) -> dict:
    """1.0 where the capability rule flags a window, else 0.0 (fed to the same gate)."""
    return {s.sid: ((s.X[:, N_EVENTS] >= HUMAN_MAX_RPS) & boot_keyboard(s.descriptor)).astype(float)
            for s in sessions}


def combine(model_first: dict, rule_first: dict) -> dict:
    """Alarm time of model OR rule (each through its own gate): whichever fires first."""
    return {sid: min([t for t in (model_first[sid], rule_first[sid]) if t is not None], default=None)
            for sid in model_first}


def rates(first, sessions) -> dict:
    atk = [s for s in sessions if s.label == 1]
    ben = [s for s in sessions if s.label == 0]
    per = {}
    for p in sorted({s.profile for s in atk}):
        g = [s for s in atk if s.profile == p]
        k = sum(first[s.sid] is not None for s in g)
        per[p] = {"rate": k / len(g), "k": k, "n": len(g), "ci": wilson(k, len(g))}
    det = sum(first[s.sid] is not None for s in atk)
    fp = sum(first[s.sid] is not None for s in ben)
    return {"detect": det / len(atk), "detect_k": det, "n_attack": len(atk),
            "fp": fp / len(ben), "fp_k": fp, "n_benign": len(ben), "per_profile": per}


def main() -> int:
    t0 = time.time()
    cfg, s1, lstm = deployed(MODELS)
    idx, thr = np.array(cfg["feature_idx"]), cfg["threshold"]
    C, _ = load_or_build()
    R = rule_scores(C)
    rule_first = run_gate(R, C, 0.5, 4, 6)
    _, S2, _ = score_with(s1, lstm, C, idx, cfg["seq_len"])
    model_first = run_gate(S2, C, thr, 4, 6)
    res = {"human_max_rps": HUMAN_MAX_RPS, "threshold": thr,
           "corpus_c": {"model": rates(model_first, C), "rule": rates(rule_first, C),
                        "hybrid": rates(combine(model_first, rule_first), C)}}
    print(f"Corpus C (fresh, seed 2025), deployed model at threshold {thr:.6f}:")
    for k in ("model", "rule", "hybrid"):
        r = res["corpus_c"][k]
        print(f"  {k:7s} detect {r['detect']:6.1%} ({r['detect_k']}/{r['n_attack']})   benign FP "
              f"{r['fp_k']}/{r['n_benign']}   " + "  ".join(
                  f"{p[:10]} {v['rate']:.0%}" for p, v in r["per_profile"].items()))

    # (i) the rule on long benign keyboards
    sim, rng = RealisticSim(seed=LONG_SEED), np.random.default_rng(LONG_SEED)
    longs = []
    for i in range(LONG_N):
        _duty(sim, rng)
        sid = f"H{LONG_SEED}-{i:03d}"
        longs.append(featurize(sim.session(sid, "keyboard", duration=LONG_DUR), sid, "keyboard"))
    lf = run_gate(rule_scores(longs), longs, 0.5, 4, 6)
    kb_alarms = sum(lf[s.sid] is not None for s in longs)
    peak = max(float(s.X[:, N_EVENTS].max()) for s in longs)
    res["long_keyboards"] = {"hours": LONG_N * LONG_DUR / 3600, "rule_alarms": kb_alarms,
                             "max_reports_in_a_window": peak}
    print(f"\nlong benign keyboards (seed {LONG_SEED}, {LONG_N} h): rule alarms {kb_alarms}; "
          f"busiest window {peak:.0f} reports")

    # (ii) leave-one-family-out, with and without the rule
    B = load_corpus(TRAIN_CACHE).sessions
    fit, val, test = stratified_split(B, seed=SEED)
    fidx = deployed_features(fit, verbose=False)
    lopo = {}
    for held in ATTACK_PROFILES:
        r = run_configuration([s for s in fit if s.profile != held],
                              [s for s in val if s.profile != held], test, name=f"w/o {held}",
                              feature_idx=fidx, seed=SEED, epochs=EPOCHS, seq_len=10, hidden=32,
                              target_fpr=0.0001, verbose=False)
        S = score_with(r.models[0], r.models[1], C, fidx, 10)[1]
        mf = run_gate(S, C, r.threshold, 4, 6)
        g = [s for s in C if s.profile == held]
        ben = [s for s in C if s.label == 0]
        hyb = combine(mf, rule_first)
        lopo[held] = {
            "unseen_model": sum(mf[s.sid] is not None for s in g) / len(g),
            "unseen_hybrid": sum(hyb[s.sid] is not None for s in g) / len(g),
            "rule_alone": sum(rule_first[s.sid] is not None for s in g) / len(g),
            "fp_model": sum(mf[s.sid] is not None for s in ben),
            "fp_hybrid": sum(hyb[s.sid] is not None for s in ben), "n_benign": len(ben)}
        print(f"  without {held:17s}: unseen model {lopo[held]['unseen_model']:5.0%}  "
              f"hybrid {lopo[held]['unseen_hybrid']:5.0%}  (rule alone "
              f"{lopo[held]['rule_alone']:4.0%})  FP {lopo[held]['fp_model']}->"
              f"{lopo[held]['fp_hybrid']}/{len(ben)}   [{time.time() - t0:.0f} s]", flush=True)
    m_model = float(np.mean([v["unseen_model"] for v in lopo.values()]))
    m_hyb = float(np.mean([v["unseen_hybrid"] for v in lopo.values()]))
    res["lopo"] = lopo
    res["lopo_mean"] = {"model": m_model, "hybrid": m_hyb}

    ok_i = (res["corpus_c"]["rule"]["fp_k"] == 0) and kb_alarms == 0
    ok_ii = m_hyb > m_model
    res["verdict"] = {"i_rule_silent_on_benign_simulated": ok_i, "ii_unseen_rises": ok_ii,
                      "testbed_check": "pending: testbed.score --rule",
                      "adopt_if_testbed_also_silent": ok_i and ok_ii}
    print(f"\nLOFO mean unseen detection: model {m_model:.1%} -> hybrid {m_hyb:.1%}")
    print(f"PRE-REGISTERED: (i) rule silent on simulated benign: {'PASS' if ok_i else 'FAIL'};  "
          f"(ii) unseen detection rises: {'PASS' if ok_ii else 'FAIL'};  "
          f"(i) on the real-kernel testbed is checked by testbed.score")
    (OUT / "analysis_hybrid.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote out/analysis_hybrid.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
