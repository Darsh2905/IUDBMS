"""
Experiment 6 -- do any model-side changes actually move the results?

    python3 -m eval.analysis_variants

Trains on the v2 realistic corpus (seed 7) and evaluates on the fresh seed-2025 set
(100 sessions per attack profile, 3,000 benign). Variants:

    base          the 28 deployed features, Stage 1 + LSTM (the evaluated architecture)
    +repeat       base + ``repeat_ngram`` (eval.analysis_ngram): long exact repeats
    +context      base + long-memory context columns fed to the LSTM only: EWMAs of
                  log activity at 5/20/60 windows, run length of the current idle or
                  active spell, and the active fraction of the last 30 windows. Gives
                  the 10-window LSTM a view of the last minute without lengthening it
                  (a longer horizon was tried before and tied -- finding 13)
    +repeat+ctx   both
    ens5          base, five LSTM seeds averaged
    ens5+repeat   the long-repeat column with a five-seed ensemble

**Noise floor first.** LSTM training is seeded but seed-sensitive, so each single-model
variant is trained with three seeds and reported as mean +/- spread; a variant is only
said to help if it clears the *base* spread. Two numbers per variant: detection at the
shipped operating rule (each model's own per-window 1e-4 threshold, 4-of-6 gate), and
detection at a **matched 0.5% benign false-positive rate** (level recalibrated on a
benign calibration half, scored on attacks), which removes threshold-luck from the
comparison.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from iudbms.stage1 import Stage1, class_group
from iudbms.stage2 import LSTMDetector, TrainConfig, build_sequences

from .analysis_common import OUT, wilson
from .analysis_gates import first_cross, kth_of_n, level_for
from .analysis_ngram import EXT_COL
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE
from .corpus import load_corpus, stratified_split
from .pipeline import deployed_features

SEED, EPOCHS, HIDDEN, SEQ = 7, 14, 32, 10
TARGET_FPR = 0.0001


# --- long-memory context, strictly causal -------------------------------------------
def context_cols(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]                                         # column 0 = n_events
    act = (n > 0).astype(np.float64)
    cols = []
    for tau in (5.0, 20.0, 60.0):
        a, acc, e = 1.0 / tau, 0.0, np.empty(len(n))
        for t in range(len(n)):
            acc += a * (np.log1p(n[t]) - acc)
            e[t] = acc
        cols.append(e)
    idle = np.empty(len(n)); active = np.empty(len(n)); ri = ra = 0
    for t in range(len(n)):
        ri, ra = (0, ra + 1) if act[t] else (ri + 1, 0)
        idle[t], active[t] = np.log1p(min(ri, 120)), np.log1p(min(ra, 120))
    cs = np.concatenate([[0.0], np.cumsum(act)])
    frac = np.array([(cs[t + 1] - cs[max(0, t - 29)]) / min(t + 1, 30) for t in range(len(n))])
    return np.column_stack(cols + [idle, active, frac])


class Variant:
    def __init__(self, name, *, repeat=False, context=False, seeds=(7, 17, 27)):
        self.name, self.repeat, self.context, self.seeds = name, repeat, context, seeds

    def idx(self, base_idx):
        return np.append(base_idx, EXT_COL) if self.repeat else base_idx


def fit_stage1(fit, idx):
    names = tuple(f"f{i}" for i in idx)
    Xb = np.vstack([s.X[:, idx] for s in fit if s.label == 0])
    groups = [class_group(s.descriptor) for s in fit if s.label == 0 for _ in range(len(s.X))]
    s1 = Stage1(seed=SEED, feature_names=names).fit(Xb, groups)
    s1.calibrate_dev(np.concatenate([s1.score_session(s.X[:, idx], s.descriptor)[2]
                                     for s in fit if s.label == 0]))
    return s1


def design(s1, sessions, idx, context):
    out = {}
    for s in sessions:
        sc = s1.score_session(s.X[:, idx], s.descriptor)[0]
        cols = [s.X[:, idx], sc[:, None]] + ([context_cols(s.X)] if context else [])
        out[s.sid] = np.hstack(cols)
    return out


def train_one(F, fit, val, seed):
    def sets(group):
        return (np.concatenate([build_sequences(F[s.sid], SEQ) for s in group]),
                np.concatenate([s.y for s in group]))
    Xtr, ytr = sets(fit)
    Xva, yva = sets(val)
    m = LSTMDetector(d_in=Xtr.shape[2], hidden=HIDDEN, seed=seed)
    m.fit(Xtr, ytr.astype(np.float32), cfg=TrainConfig(epochs=EPOCHS, seed=seed, hidden=HIDDEN, lr=2e-3),
          val=(Xva, yva), verbose=False)
    return m


def predict(models, F, sessions):
    return {s.sid: np.mean([m.predict_proba(build_sequences(F[s.sid], SEQ)) for m in models], axis=0)
            for s in sessions}


def evaluate(S2, val_scores, val, powered, rng):
    thr = float(np.quantile(np.concatenate([val_scores[s.sid][s.y == 0] for s in val if s.label == 0]),
                            1 - TARGET_FPR))
    ben = [s for s in powered if s.label == 0]
    atk = [s for s in powered if s.label == 1]
    stat = {s.sid: kth_of_n(S2[s.sid], 4, 6) for s in powered}
    fp = np.mean([first_cross(stat[s.sid], thr) is not None for s in ben])
    det = np.mean([first_cross(stat[s.sid], thr) is not None for s in atk])
    # matched-FP: recalibrate on half of the benign sessions
    perm = rng.permutation(len(ben))
    cal = [ben[i] for i in perm[:len(ben) // 2]]
    level = level_for(np.array([stat[s.sid].max() for s in cal]), 0.005)
    det_m = np.mean([first_cross(stat[s.sid], level) is not None for s in atk])
    y = np.concatenate([s.y for s in powered]); p = np.concatenate([S2[s.sid] for s in powered])
    return {"auc": float(roc_auc_score(y, p)), "threshold": thr, "fp": float(fp), "detect": float(det),
            "detect_at_0.5pct": float(det_m),
            "per_profile": {pr: float(np.mean([first_cross(stat[s.sid], thr) is not None
                                              for s in atk if s.profile == pr]))
                            for pr in sorted({s.profile for s in atk})},
            "per_profile_matched": {pr: float(np.mean([first_cross(stat[s.sid], level) is not None
                                                      for s in atk if s.profile == pr]))
                                    for pr in sorted({s.profile for s in atk})}}


def main() -> int:
    t0 = time.time()
    fit, val, _ = stratified_split(load_corpus(TRAIN_CACHE).sessions, seed=SEED)
    powered, _ = load_or_build()
    base_idx = deployed_features(fit, verbose=False)
    rng = np.random.default_rng(5)

    variants = [Variant("base"), Variant("+repeat", repeat=True), Variant("+context", context=True),
                Variant("+repeat+ctx", repeat=True, context=True),
                Variant("ens5", seeds=(7, 17, 27, 37, 47)),
                Variant("ens5+repeat", repeat=True, seeds=(7, 17, 27, 37, 47))]
    results = {}
    s1_cache = {}
    for v in variants:
        idx = v.idx(base_idx)
        key = tuple(idx)
        if key not in s1_cache:
            s1_cache[key] = fit_stage1(fit, idx)
        s1 = s1_cache[key]
        # Two tables, never one: the fresh evaluation set reuses the training corpus's
        # session ids (S0001, A0007, ...), so a single dict keyed by sid let evaluation
        # sessions silently overwrite training sessions.
        F = design(s1, fit + val, idx, v.context)
        Fp = design(s1, powered, idx, v.context)
        runs = []
        if v.name.startswith("ens"):
            models = [train_one(F, fit, val, sd) for sd in v.seeds]
            runs.append(evaluate(predict(models, Fp, powered), predict(models, F, val), val, powered, rng))
        else:
            for sd in v.seeds:
                m = train_one(F, fit, val, sd)
                runs.append(evaluate(predict([m], Fp, powered), predict([m], F, val), val, powered, rng))
        results[v.name] = runs
        d = np.array([r["detect"] for r in runs]); dm = np.array([r["detect_at_0.5pct"] for r in runs])
        fpm = np.mean([r["fp"] for r in runs]); au = np.mean([r["auc"] for r in runs])
        sp = f"+/-{d.std():.1%}" if len(runs) > 1 else "  (1 model)"
        spm = f"+/-{dm.std():.1%}" if len(runs) > 1 else ""
        print(f"{v.name:12s} AUC {au:.4f} | shipped rule: detect {d.mean():5.1%} {sp:>10s} FP {fpm:6.2%} "
              f"| matched 0.5% FP: detect {dm.mean():5.1%} {spm:>8s}   ({time.time() - t0:.0f} s)", flush=True)

    print("\nper-family detection at matched 0.5% benign FP (mean over seeds)")
    profs = sorted(results["base"][0]["per_profile_matched"])
    print(f"  {'variant':12s}" + "".join(f"{p[:11]:>12s}" for p in profs))
    for name, runs in results.items():
        print(f"  {name:12s}" + "".join(
            f"{np.mean([r['per_profile_matched'][p] for r in runs]):12.0%}" for p in profs))

    b = np.array([r["detect_at_0.5pct"] for r in results["base"]])
    print(f"\nnoise floor: base detection at matched FP = {b.mean():.1%} +/- {b.std():.1%} over "
          f"{len(b)} seeds (range {b.min():.1%}-{b.max():.1%}); one session of 600 = {1/600:.2%}")
    (OUT / "analysis_variants.json").write_text(json.dumps(results, indent=2, default=float))
    print("wrote out/analysis_variants.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
