"""
Experiment 5 -- does Stage 2 catch an attack family it was never trained on?

    python3 -m eval.analysis_lopo

Stage 2 is *supervised*: it is trained on the six attack families the simulator can
generate. A real attack will be a seventh. Every reported detection rate is
therefore an in-distribution number, and nothing in the project measures the thing
that matters for deployment -- generalisation to an unseen family.

Leave-one-profile-out: for each attack family P, train on benign traffic plus the
*other five* families (Stage 1, the LSTM, and the threshold calibration all exclude
P), then score P on the fresh seed-2025 set. Three columns per family:

    full        trained WITH P (the usual in-distribution number)
    unseen      trained WITHOUT P (the generalisation number)
    stage 1     the label-free model, which never sees any attack

The gap between ``full`` and ``unseen`` is how much of the headline is memorised
family structure rather than behaviour that transfers.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES

from .analysis_common import OUT, deployed, run_gate, score_with, wilson
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE
from .corpus import load_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

SEED, EPOCHS = 7, 14


def rate(first, sessions):
    k = sum(first[s.sid] is not None for s in sessions)
    return k / len(sessions), wilson(k, len(sessions))


def either(a: dict, b: dict) -> dict:
    """Alarm time of an OR of two detectors: whichever fires first."""
    out = {}
    for sid in a:
        ts = [t for t in (a[sid], b[sid]) if t is not None]
        out[sid] = min(ts) if ts else None
    return out


def main() -> int:
    t0 = time.time()
    train_sessions = load_corpus(TRAIN_CACHE).sessions
    fit, val, test = stratified_split(train_sessions, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    powered, _ = load_or_build()
    ben = [s for s in powered if s.label == 0]
    print(f"train: {len(fit)} fit / {len(val)} val sessions (realistic corpus, seed 7);  "
          f"evaluate on {len(powered)} fresh sessions ({len(ben)} benign), seed 2025\n")

    def evaluate(models_cfg):
        cfg, s1, lstm = models_cfg
        S2 = score_with(s1, lstm, powered, np.array(cfg["feature_idx"]), cfg["seq_len"])[1]
        return run_gate(S2, powered, cfg["threshold"], cfg["k"], cfg["n"])

    full = evaluate(deployed(OUT / "analysis_models" / "realistic_v2"))

    s1only = run_configuration(fit, val, test, name="stage1-only", feature_idx=idx, seed=SEED,
                               use_stage2=False, target_fpr=0.0001, verbose=False)
    cfg1 = {"feature_idx": idx, "threshold": s1only.threshold, "seq_len": 10, "k": 4, "n": 6}
    S1 = score_with(s1only.models[0], None, powered, idx, 10)[1]
    first_s1 = run_gate(S1, powered, s1only.threshold, 4, 6)

    rows = {}
    for held in ATTACK_PROFILES:
        r = run_configuration([s for s in fit if s.profile != held],
                              [s for s in val if s.profile != held], test,
                              name=f"without {held}", feature_idx=idx, seed=SEED, epochs=EPOCHS,
                              seq_len=10, hidden=32, target_fpr=0.0001, verbose=False)
        S2 = score_with(r.models[0], r.models[1], powered, idx, 10)[1]
        first = run_gate(S2, powered, r.threshold, 4, 6)
        tgt = [s for s in powered if s.profile == held]
        or_unseen = either(first, first_s1)
        rows[held] = {
            "full": rate(full, tgt), "unseen": rate(first, tgt), "stage1": rate(first_s1, tgt),
            "unseen_or_s1": rate(or_unseen, tgt),
            "fp_unseen": rate(first, ben)[0], "fp_unseen_or_s1": rate(or_unseen, ben)[0]}
        print(f"  trained without {held:17s} done ({time.time() - t0:.0f} s)", flush=True)

    fp_full, fp_s1 = rate(full, ben)[0], rate(first_s1, ben)[0]
    print(f"\ndetection of each family on the fresh set (n=100 per family; 95% CI in brackets)")
    print(f"  {'family':18s} {'full (seen)':>16s} {'UNSEEN':>16s} {'stage 1 only':>16s} {'UNSEEN OR S1':>16s}")
    for p, r in rows.items():
        def f(x): return f"{x[0]:5.0%} [{x[1][0]:3.0%}-{x[1][1]:3.0%}]"
        print(f"  {p:18s} {f(r['full']):>16s} {f(r['unseen']):>16s} {f(r['stage1']):>16s} "
              f"{f(r['unseen_or_s1']):>16s}")
    fp_or = float(np.mean([rows[p]['fp_unseen_or_s1'] for p in rows]))
    print(f"  {'benign FP':18s} {fp_full:>16.2%} {'(per-model)':>16s} {fp_s1:>16.2%} {fp_or:>16.2%}")
    mean = lambda k: float(np.mean([rows[p][k][0] for p in rows]))
    print(f"  {'MEAN over families':18s} {mean('full'):>16.1%} {mean('unseen'):>16.1%} "
          f"{mean('stage1'):>16.1%} {mean('unseen_or_s1'):>16.1%}")

    (OUT / "analysis_lopo.json").write_text(json.dumps(
        {p: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
         for p, r in rows.items()} | {"benign_fp_full": fp_full, "benign_fp_stage1": fp_s1},
        indent=2, default=float))
    print(f"\nwrote out/analysis_lopo.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
