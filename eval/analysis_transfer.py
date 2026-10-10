"""
Why does detection fall from 82.7% (families seen in training) to 34.8% (held out)?

    python3 -m eval.analysis_transfer          # ~8 min

Leave-one-family-out (eval.analysis_lopo) measures THAT generalisation fails; this
measures WHY, with two instruments:

1. **A transfer matrix.** For each attack family F, train Stage 1 + Stage 2 on
   benign traffic plus F ALONE, calibrate exactly as the deployed model is, and
   score every family G on the fresh seed-2025 set (Corpus C). Cell [F, G] is the
   detection rate of G by a model that has only ever seen F. The diagonal is
   in-distribution; the off-diagonal is transfer.
2. **Signatures.** For each family, the features that separate its post-onset
   windows from benign windows of the same declared class group, ranked by the
   two-sample Kolmogorov-Smirnov statistic D on Corpus B's training split. Two
   families whose signatures do not overlap give a supervised model nothing to
   carry from one to the other.

The prediction tested: transfer F -> G is high when, and only when, G's signature
overlaps F's. Writes out/analysis_transfer.json. Trains nothing that is deployed.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES
from iudbms.contract import FEATURE_NAMES
from iudbms.stage1 import class_group

from .analysis_common import OUT, run_gate, score_with, wilson
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE
from .corpus import load_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

SEED, EPOCHS, TOP = 7, 14, 5


def ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def signatures(fit, idx) -> dict:
    out = {}
    for fam in ATTACK_PROFILES:
        sess = [s for s in fit if s.profile == fam]
        grp = class_group(sess[0].descriptor)
        atk = np.vstack([s.X[s.y == 1] for s in sess])
        ben = np.vstack([s.X for s in fit if s.label == 0 and class_group(s.descriptor) == grp])
        d = {FEATURE_NAMES[i]: ks(atk[:, i], ben[:, i]) for i in idx}
        top = sorted(d.items(), key=lambda kv: -kv[1])[:TOP]
        out[fam] = {"class_group": grp, "top": top,
                    "direction": {n: float(np.median(atk[:, FEATURE_NAMES.index(n)])
                                           - np.median(ben[:, FEATURE_NAMES.index(n)]))
                                  for n, _ in top}}
    return out


def main() -> int:
    t0 = time.time()
    B = load_corpus(TRAIN_CACHE).sessions
    fit, val, test = stratified_split(B, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    C, _ = load_or_build()
    ben = [s for s in C if s.label == 0]
    fams = list(ATTACK_PROFILES)

    sig = signatures(fit, idx)
    print("SIGNATURES (top features by KS D, post-onset vs benign of the same class group)")
    for f, v in sig.items():
        print(f"  {f:17s} [{v['class_group']}] " + ", ".join(f"{n} {d:.2f}" for n, d in v["top"]))

    matrix, fp = {}, {}
    for F in fams:
        keep = lambda s: s.label == 0 or s.profile == F          # noqa: E731
        r = run_configuration([s for s in fit if keep(s)], [s for s in val if keep(s)], test,
                              name=f"only {F}", feature_idx=idx, seed=SEED, epochs=EPOCHS,
                              seq_len=10, hidden=32, target_fpr=0.0001, verbose=False)
        S2 = score_with(r.models[0], r.models[1], C, idx, 10)[1]
        first = run_gate(S2, C, r.threshold, 4, 6)
        matrix[F] = {}
        for G in fams:
            g = [s for s in C if s.profile == G]
            k = sum(first[s.sid] is not None for s in g)
            matrix[F][G] = {"rate": k / len(g), "k": k, "n": len(g), "ci": wilson(k, len(g))}
        k = sum(first[s.sid] is not None for s in ben)
        fp[F] = {"k": k, "n": len(ben), "rate": k / len(ben)}
        print(f"  trained on {F:17s} done  ({time.time() - t0:.0f} s)", flush=True)

    print("\nTRANSFER MATRIX  rows: the only attack family in training; columns: family tested "
          "(Corpus C, n=100 each)")
    print(f"  {'trained on':17s}" + "".join(f"{g[:10]:>11s}" for g in fams) + "   benign FP")
    for F in fams:
        print(f"  {F:17s}" + "".join(f"{matrix[F][G]['rate']:>11.0%}" for G in fams)
              + f"   {fp[F]['k']}/{fp[F]['n']}")

    # does signature overlap predict transfer?
    def topset(f):
        return {n for n, _ in sig[f]["top"]}
    pairs = []
    for F in fams:
        for G in fams:
            if F != G:
                j = len(topset(F) & topset(G)) / len(topset(F) | topset(G))
                pairs.append((F, G, j, matrix[F][G]["rate"]))
    jj, tt = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    rho = float(np.corrcoef(np.argsort(np.argsort(jj)), np.argsort(np.argsort(tt)))[0, 1])
    hi = tt[jj >= 0.4].mean() if (jj >= 0.4).any() else float("nan")
    lo = tt[jj < 0.4].mean() if (jj < 0.4).any() else float("nan")
    print(f"\nsignature overlap (Jaccard of top-{TOP}) vs off-diagonal transfer: "
          f"Spearman rho = {rho:.2f};  mean transfer {hi:.0%} where overlap >= 0.4, "
          f"{lo:.0%} where < 0.4")

    (OUT / "analysis_transfer.json").write_text(json.dumps(
        {"signatures": sig, "matrix": matrix, "benign_fp": fp,
         "overlap_vs_transfer": {"pairs": pairs, "spearman": rho,
                                 "mean_transfer_high_overlap": hi, "mean_transfer_low_overlap": lo},
         "runtime_s": time.time() - t0}, indent=1, default=float))
    print(f"wrote out/analysis_transfer.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
