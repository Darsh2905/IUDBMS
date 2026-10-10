"""
Retrain the evaluated system exactly as ``eval.run_eval`` does, and compare it with
(a) the numbers in out/metrics.json and (b) the checkpoint in out/models/.

    python3 -m eval.analysis_retrain            # trains, saves to out/analysis_models/

Why this exists: scoring the *shipped* checkpoint on the test split gave window
AUC 0.9598 and **0 of 120** attack sessions detected, against a reported 0.9762 and
89.2%. Stage 1 reproduced exactly (max |diff| 0.0), so the corpus and the feature
plumbing are right; the Stage-2 weights are the thing that does not belong to the
reported run. This script settles whether the *reported* numbers are reproducible
(retraining with the same seed and parameters) independently of which file happens
to sit in out/models/.

Writes out/analysis_models/{stage1.pkl,stage2.pt,deploy.json} -- a separate
directory, never out/models/, which is the deployment's.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .analysis_common import ORIGINAL, MODELS, OUT, deployed, load_splits, run_gate, score_with, session_rates
from .pipeline import deployed_features, run_configuration

DEST = OUT / "analysis_models"
TARGET_FPR = 0.0001           # `make eval` passes --target-fpr 0.0001
SEED, EPOCHS, SEQ_LEN, HIDDEN = 7, 14, 10, 32


def main() -> int:
    _, fit, val, test = load_splits()
    idx = deployed_features(fit)
    print(f"deployed feature columns: {len(idx)}  (indices {idx.min()}..{idx.max()})")

    t0 = time.time()
    r = run_configuration(fit, val, test, name="IUDBMS (S1+S2)", feature_idx=idx,
                          seed=SEED, epochs=EPOCHS, seq_len=SEQ_LEN, hidden=HIDDEN,
                          target_fpr=TARGET_FPR, keep_scores=True, verbose=False)
    print(f"trained in {time.time() - t0:.0f} s")

    m = json.loads((OUT / "metrics.json").read_text())["full_system"]
    print("\nretrained vs reported (out/metrics.json):")
    print(f"  window AUC   {r.window['auc']:.4f}   vs {m['window']['auc']:.4f}")
    print(f"  detection    {r.session['attack_detect']:.4f}   vs {m['session']['attack_detect']:.4f}")
    print(f"  benign FP    {r.session['benign_fp']:.4f}   vs {m['session']['benign_fp']:.4f}")
    print(f"  threshold    {r.threshold:.6f} vs {m['threshold']:.6f}")

    DEST.mkdir(exist_ok=True)
    s1m, s2m = r.models
    s1m.save(DEST / "stage1.pkl")
    s2m.save(DEST / "stage2.pt")
    (DEST / "deploy.json").write_text(json.dumps({
        "feature_idx": [int(i) for i in idx], "threshold": r.threshold,
        "target_fpr": TARGET_FPR, "seq_len": SEQ_LEN, "k": 4, "n": 6}, indent=2))

    # ---- the shipped checkpoint, scored the same way ----------------------
    cfg, s1, lstm = deployed(ORIGINAL)
    S1, S2, _ = score_with(s1, lstm, test, np.array(cfg["feature_idx"]), SEQ_LEN)
    first = run_gate(S2, test, cfg["threshold"])
    sh = session_rates(first, test)
    both = [s for s in test if s.label == 1]
    mx = float(np.median([S2[s.sid][s.y == 1].max() for s in both]))
    print("\nthe checkpoint in out/models/ (shipped, and deployed to the VM):")
    print(f"  threshold in deploy.json     {cfg['threshold']:.4f}")
    print(f"  detection at that threshold  {sh['detect']:.1%}  ({sh['detect_k']}/{sh['n_attack']})")
    print(f"  median per-session post-onset max Stage-2 score  {mx:.3f}")

    # do the two checkpoints disagree, and is Stage 1 the same?
    s1_same = float(np.abs(
        s1m.score_session(test[0].X[:, idx], test[0].descriptor)[0]
        - s1.score_session(test[0].X[:, idx], test[0].descriptor)[0]).max())
    print(f"  Stage-1 max |diff| vs retrained: {s1_same:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
