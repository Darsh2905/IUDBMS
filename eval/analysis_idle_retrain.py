"""
Experiment 2 -- retrain with benign devices that go idle, and see what survives.

    python3 -m eval.analysis_idle_retrain

``eval.analysis_idle`` showed the evaluated model alarms on 90-100% of benign
keyboards and mice once they have idle gaps. This builds a corpus in which benign
devices have realistic duty cycles, trains the *same* architecture and procedure on
it (seed, epochs, calibration, gate all unchanged), and reports, on a test split
that contains both always-active and idle-gapped benign devices:

* the false-positive rate on each benign condition,
* detection per attack profile -- including ``mimicry_adaptive``, whose only tell
  was "typing, then permanent silence", which a real idle keyboard now shares,
* the same numbers for the original (always-active) model on the same test split,
  so the comparison is like for like.

Attack generation is deliberately unchanged, so any change in detection is due to
the benign side getting harder, which is the point.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import Simulator
from .analysis_common import (OUT, deployed, featurize, per_profile, run_gate, save_set,
                              score_with, session_rates)
from .analysis_idle import DutyCycleSim
from .corpus import Corpus, SessionData, load_corpus, save_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

CACHE = OUT / "analysis_corpus_idle.npz"
SEED, EPOCHS = 7, 14

#: (name, probability, DutyCycleSim settings). Drawn per benign session of the
#: classes that really do go quiet. Probabilities are fixed here, a priori, not
#: tuned against any outcome.
IDLE_MIX = [
    ("active",  0.40, dict(idle=(0.0, 0.0), active=(1e9, 1e9), fully_idle=False)),
    ("gap3-8",  0.15, dict(idle=(3.0, 8.0), active=(5.0, 25.0), fully_idle=False)),
    ("gap8-20", 0.15, dict(idle=(8.0, 20.0), active=(5.0, 25.0), fully_idle=False)),
    ("gap20-60", 0.20, dict(idle=(20.0, 60.0), active=(5.0, 25.0), fully_idle=False)),
    ("idle",    0.10, dict(idle=(0.0, 0.0), active=(0.0, 0.0), fully_idle=True)),
]
QUIET_CLASSES = ("keyboard", "mouse", "mass_storage", "webcam", "audio")


def build(n_benign: int = 600, n_attack: int = 400, seed: int = SEED):
    sim = DutyCycleSim(seed=seed)
    rng = np.random.default_rng(seed + 99)
    probs = np.array([m[1] for m in IDLE_MIX])
    sessions, cond = [], {}
    for sid, profile in sim.plan(n_benign=n_benign, n_attack=n_attack):
        if profile in QUIET_CLASSES:
            name, _, kw = IDLE_MIX[int(rng.choice(len(IDLE_MIX), p=probs))]
        else:
            name, kw = "attack" if profile != "x" else "x", IDLE_MIX[0][2]
        sim._idle, sim._active, sim._fully_idle = kw["idle"], kw["active"], kw["fully_idle"]
        sessions.append(featurize(sim.session(sid, profile), sid, profile))
        cond[sid] = name
    return sessions, cond


def main() -> int:
    t0 = time.time()
    meta_path = CACHE.with_suffix(".cond.json")
    if CACHE.exists() and meta_path.exists():
        corpus = load_corpus(CACHE)
        sessions, cond = corpus.sessions, json.loads(meta_path.read_text())
        print(f"loaded cached idle corpus: {len(sessions)} sessions")
    else:
        sessions, cond = build()
        save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), CACHE)
        meta_path.write_text(json.dumps(cond))
        print(f"built idle corpus: {len(sessions)} sessions in {time.time() - t0:.0f} s")

    fit, val, test = stratified_split(sessions, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    r = run_configuration(fit, val, test, name="retrained, idle-aware", feature_idx=idx,
                          seed=SEED, epochs=EPOCHS, seq_len=10, hidden=32,
                          target_fpr=0.0001, keep_scores=True, verbose=False)
    s1n, lstmn = r.models
    save_set(OUT / "analysis_models" / "idle_aware", r, idx, target_fpr=0.0001)

    # the original model, scored on the same test split
    cfg, s1o, lstmo = deployed()
    S2o = score_with(s1o, lstmo, test, np.array(cfg["feature_idx"]), cfg["seq_len"])[1]
    first_o = run_gate(S2o, test, cfg["threshold"], cfg["k"], cfg["n"])
    first_n = run_gate(r.stage2, test, r.threshold, 4, 6)

    print(f"\nnew model: threshold {r.threshold:.4f}, window AUC {r.window['auc']:.4f}   "
          f"(original threshold {cfg['threshold']:.4f})")
    ben = [s for s in test if s.label == 0]
    print("\nBENIGN false-positive rate by condition (session level, test split)")
    print(f"  {'condition':10s} {'n':>4s} {'original':>10s} {'idle-aware':>11s}")
    out = {"benign": {}, "attack": {}}
    for name, *_ in IDLE_MIX:
        g = [s for s in ben if cond[s.sid] == name]
        if not g:
            continue
        fo = sum(first_o[s.sid] is not None for s in g) / len(g)
        fn = sum(first_n[s.sid] is not None for s in g) / len(g)
        print(f"  {name:10s} {len(g):4d} {fo:10.1%} {fn:11.1%}")
        out["benign"][name] = {"n": len(g), "original": fo, "idle_aware": fn}
    ho, hn = session_rates(first_o, test), session_rates(first_n, test)
    print(f"  {'ALL':10s} {len(ben):4d} {ho['fp']:10.1%} {hn['fp']:11.1%}")

    print("\nATTACK detection by profile (test split)")
    po, pn = per_profile(first_o, test), per_profile(first_n, test)
    print(f"  {'profile':18s} {'n':>3s} {'original':>9s} {'idle-aware':>11s}   95% CI (idle-aware)")
    for p in sorted(pn):
        if pn[p]["label"] != 1:
            continue
        lo, hi = pn[p]["ci"]
        print(f"  {p:18s} {pn[p]['n']:3d} {po[p]['rate']:9.1%} {pn[p]['rate']:11.1%}   {lo:.0%}-{hi:.0%}")
        out["attack"][p] = {"n": pn[p]["n"], "original": po[p]["rate"], "idle_aware": pn[p]["rate"]}
    print(f"  {'ALL':18s} {ho['n_attack']:3d} {ho['detect']:9.1%} {hn['detect']:11.1%}")
    print(f"  median latency: original {ho['median_latency']:.1f}s   idle-aware {hn['median_latency']:.1f}s")
    out["overall"] = {"original": {k: ho[k] for k in ("detect", "fp")},
                      "idle_aware": {k: hn[k] for k in ("detect", "fp")},
                      "window_auc": r.window["auc"], "threshold": r.threshold}
    (OUT / "analysis_idle_retrain.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote out/analysis_idle_retrain.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
