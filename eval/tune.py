"""
Hyperparameter sweep — **validation only**.

    python3 -m eval.tune [--n-benign 420] [--n-attack 280]

The test split is never constructed here. Sweeping on test and then reporting
test is the single easiest way to manufacture a result, and it is precisely the
failure mode this project exists to argue against.

Three disjoint splits are carved out of the non-test sessions:

    inner_fit   trains Stage 1 and the LSTM
    inner_cal   sets the decision threshold at the target per-window FPR
    inner_sel   the sweep is scored here, and the winner is chosen here

``run_configuration`` refuses to run if the calibration and scoring splits are
the same, so the circularity cannot be reintroduced by accident.

The winning configuration is printed, not applied. Put it into
``eval/run_eval.py`` deliberately, then run that once against test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .corpus import build_corpus, load_corpus, save_corpus, stratified_split
from .pipeline import run_configuration

OUT = Path(__file__).resolve().parent.parent / "out"


def inner_splits(fit, val, seed: int = 7):
    """Re-split the non-test sessions into train / calibrate / select."""
    pool = fit + val
    rng = np.random.default_rng(seed + 1000)
    by_profile: dict[str, list] = {}
    for s in pool:
        by_profile.setdefault(s.profile, []).append(s)

    inner_fit, inner_cal, inner_sel = [], [], []
    for profile in sorted(by_profile):
        group = by_profile[profile]
        idx = rng.permutation(len(group))
        n = len(group)
        a, b = int(round(0.64 * n)), int(round(0.82 * n))
        inner_fit += [group[i] for i in idx[:a]]
        inner_cal += [group[i] for i in idx[a:b]]
        inner_sel += [group[i] for i in idx[b:]]
    return inner_fit, inner_cal, inner_sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-benign", type=int, default=420)
    ap.add_argument("--n-attack", type=int, default=280)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seq-len", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--target-fpr", type=float, nargs="+",
                    default=[0.0001, 0.0005, 0.001, 0.005])
    ap.add_argument("--cache", type=Path, default=OUT / "corpus_cache.npz")
    ap.add_argument("--incumbent", type=int, nargs=2, metavar=("SEQ_LEN", "HIDDEN"),
                    default=[10, 32],
                    help="the currently deployed configuration; a challenger must "
                         "beat it by more than the noise floor to be recommended")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    if args.cache.exists():
        corpus = load_corpus(args.cache)
        print(f"  loaded cached corpus: {len(corpus.sessions)} sessions "
              f"({corpus.n_windows:,} windows)")
    else:
        print(f"  building corpus ({args.n_benign} benign / {args.n_attack} attack)...", flush=True)
        corpus = build_corpus(args.n_benign, args.n_attack, seed=args.seed)
        save_corpus(corpus, args.cache)
        print(f"  cached to {args.cache.name}", flush=True)

    fit, val, test = stratified_split(corpus.sessions, seed=args.seed)
    i_fit, i_cal, i_sel = inner_splits(fit, val, seed=args.seed)
    print(f"  test held back      : {len(test)} sessions (NOT touched here)", flush=True)
    print(f"  inner train/cal/sel : {len(i_fit)} / {len(i_cal)} / {len(i_sel)}")
    n_sel_attack = sum(1 for s in i_sel if s.label == 1)
    print(f"  selection set has   : {n_sel_attack} attack sessions\n", flush=True)

    grid = list(itertools.product(args.seq_len, args.hidden, args.target_fpr))
    print(f"{'seq_len':>8s} {'hidden':>7s} {'tgtFPR':>8s} {'AUC':>8s} {'recall':>8s} "
          f"{'detect':>8s} {'FP':>7s} {'adaptive':>9s} {'lat':>7s}", flush=True)
    rows = []
    for seq_len, hidden, tfpr in grid:
        r = run_configuration(
            i_fit, i_cal, i_sel, name=f"T={seq_len},H={hidden},f={tfpr}",
            seed=args.seed, epochs=args.epochs, seq_len=seq_len, hidden=hidden,
            target_fpr=tfpr, verbose=False,
        )
        # The three positional arguments are roles: train / calibrate / report.
        # Here they carry inner_fit / inner_cal / inner_sel, so the defaults
        # already calibrate on inner_cal and score on inner_sel -- three
        # disjoint splits, none of them the real test set.
        mim = r.per_profile.get("mimicry_adaptive", r.per_profile.get("mimicry", {}))
        lat = r.session["median_latency"]
        rows.append({"seq_len": seq_len, "hidden": hidden, "target_fpr": tfpr,
                     "auc": r.window["auc"], "recall": r.window["recall"],
                     "detect": r.session["attack_detect"],
                     "fp": r.session["benign_fp"],
                     "mimicry": mim.get("detect_rate"),
                     "latency": lat})
        print(f"{seq_len:8d} {hidden:7d} {tfpr:8.4%} {r.window['auc']:8.4f} "
              f"{r.window['recall']:8.3f} {r.session['attack_detect']:8.1%} "
              f"{r.session['benign_fp']:7.1%} "
              f"{(mim.get('detect_rate') or 0):9.1%} "
              f"{(f'{lat:.1f}s' if lat else '-'):>7s}", flush=True)

    # Detection first, then the FEWEST false positives, then AUC. Without the
    # false-positive term a sweep happily picks the loosest threshold that
    # saturates detection and ships an unusable alert rate.
    best = max(rows, key=lambda r: (r["detect"], -r["fp"], r["auc"]))
    print(f"\n  best on the selection split: seq_len={best['seq_len']} "
          f"hidden={best['hidden']} target_fpr={best['target_fpr']:.4%}  "
          f"detect={best['detect']:.1%}  FP={best['fp']:.1%}  AUC={best['auc']:.4f}", flush=True)
    print("  This is NOT applied automatically. Set it in eval/run_eval.py and run")
    print("  that once against the held-out test split.", flush=True)

    (OUT / "tuning.json").write_text(json.dumps(
        {"grid": rows, "best": best, "recommended": recommended,
         "noise_floor": noise, "epochs": args.epochs,
         "corpus": len(corpus.sessions),
         "splits": {"inner_fit": len(i_fit), "inner_cal": len(i_cal),
                    "inner_sel": len(i_sel), "test_held_back": len(test)}},
        indent=2, default=float))
    print(f"\n  wrote out/tuning.json    ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
