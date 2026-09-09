"""
Full evaluation: every table in plan §6 and §8.3, reproducible from one command.

    python3 -m eval.run_eval [--n-benign 210] [--n-attack 140] [--seed 7]

Produces out/metrics.json plus figures. Every number in a report or slide must
come from here — nothing is transcribed by hand.

Ablations run through the identical code path as the full system, differing
only in arguments:

  * **Stage-1 only** isolates pillar N3 (temporal modelling). Stage 1 is
    memoryless, so it can only catch attacks with a static per-window
    signature.
  * **Single-modality** variants isolate pillar N1 (multi-modal fusion). Each
    keeps one modality's columns and drops the rest.
  * **Identity whitelist** is the USBGuard-style baseline, expected to score 0 %
    because every attack presents a descriptor drawn from the benign pool.
  * **Leave-one-out on log_time_since_enum** measures how much the one feature
    that could encode session position actually contributes, rather than
    assuming it is harmless (CLAUDE.md §5, decision C).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iudbms.contract import MODALITIES, N_FEATURES, feature_index, modality_indices  # noqa: F401

from .corpus import build_corpus, load_corpus, stratified_split
from .pipeline import GATE_K, GATE_N, TARGET_FPR, identity_whitelist_baseline, run_configuration

OUT = Path(__file__).resolve().parent.parent / "out"


def banner(msg: str) -> None:
    print(f"\n{'=' * 74}\n{msg}\n{'=' * 74}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-benign", type=int, default=210)
    ap.add_argument("--n-attack", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seq-len", type=int, default=10,
                    help="LSTM sequence horizon in windows; tuned by eval.tune")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--cache", type=Path, default=None,
                    help="reuse a corpus built by eval.tune (out/corpus_cache.npz)")
    ap.add_argument("--quick", action="store_true", help="small corpus, few epochs")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--persist", action="store_true",
                    help="write devices/windows/alerts/actions to out/iudbms.db "
                         "so the API and dashboard have something to serve")
    args = ap.parse_args()
    if args.quick:
        args.n_benign, args.n_attack, args.epochs = 60, 40, 4

    t_start = time.time()
    OUT.mkdir(exist_ok=True)

    banner("[1/5] CORPUS  simulate -> usbmon text -> parse -> 28-dim windows")
    if args.cache and args.cache.exists():
        corpus = load_corpus(args.cache)
        n_b = sum(1 for s in corpus.sessions if s.label == 0)
        print(f"  sessions        : {len(corpus.sessions)}  ({n_b} benign / "
              f"{len(corpus.sessions) - n_b} malicious)   [cached]")
        print(f"  usbmon lines    : {corpus.n_lines:,}")
        print(f"  URB events      : {corpus.n_events:,}")
    else:
        corpus = build_corpus(args.n_benign, args.n_attack, seed=args.seed)
    fit, val, test = stratified_split(corpus.sessions, seed=args.seed)
    print(f"  windows (1.0 s) : {corpus.n_windows:,}   features/window: {N_FEATURES}")
    print(f"  split (sessions): fit={len(fit)}  val={len(val)}  test={len(test)}")
    print("  split is BY SESSION, stratified by profile — never by window (plan §5)")

    results = {}

    banner("[2/5] FULL SYSTEM  Stage 1 + Stage 2 (PyTorch LSTM)")
    # DEPLOYED FEATURE SET = 27 of 28. log_time_since_enum is excluded because
    # it leaks the label: post-onset windows are later in a session by
    # construction, so the feature separates every attack class from benign
    # traffic (KS D = 0.03-0.41) without encoding any behaviour. A real
    # attacker chooses onset adversarially; our corpus does not. Including it
    # is reported below as an ablation rather than shipped.
    deployed = np.array([i for i in range(N_FEATURES)
                         if i != feature_index("log_time_since_enum")])
    print(f"  deployed features: {len(deployed)} of {N_FEATURES} "
          f"(log_time_since_enum excluded -- see ablation below)")
    print(f"  sequence horizon: {args.seq_len} windows   hidden units: {args.hidden}")
    full = run_configuration(fit, val, test, name="IUDBMS (S1+S2)",
                             feature_idx=deployed,
                             seed=args.seed, epochs=args.epochs,
                             seq_len=args.seq_len, hidden=args.hidden,
                             keep_scores=True)
    results["full"] = full

    banner("[3/5] ABLATIONS")
    print("\n-- Stage-1 only (isolates pillar N3: temporal modelling) --")
    s1_only = run_configuration(fit, val, test, name="Stage 1 only",
                                feature_idx=deployed,
                                use_stage2=False, seed=args.seed, verbose=False)
    results["stage1_only"] = s1_only

    print("\n-- single-modality (isolates pillar N1: multi-modal fusion) --")
    modality = {}
    for m in MODALITIES:
        idx = np.array(modality_indices(m))
        r = run_configuration(fit, val, test, name=f"{m} only", feature_idx=idx,
                              seed=args.seed, epochs=args.epochs,
                              seq_len=args.seq_len, hidden=args.hidden, verbose=False)
        modality[m] = r
        print(f"  {m:9s} ({len(idx):2d} features)  AUC={r.window['auc']:.4f}  "
              f"detect={r.session['attack_detect']:.1%}  "
              f"FP={r.session['benign_fp']:.1%}")
    results["modality"] = modality

    print("\n-- adding back log_time_since_enum (the leaking feature) --")
    loo = run_configuration(fit, val, test, name="with log_time_since_enum",
                            feature_idx=None, seed=args.seed, epochs=args.epochs,
                            seq_len=args.seq_len, hidden=args.hidden, verbose=False)
    results["with_time_feature"] = loo
    d_auc = loo.window["auc"] - full.window["auc"]
    d_det = loo.session["attack_detect"] - full.session["attack_detect"]
    print(f"  deployed (27 features) : AUC={full.window['auc']:.4f}  "
          f"detect={full.session['attack_detect']:.1%}")
    print(f"  plus the leak (28)     : AUC={loo.window['auc']:.4f}  "
          f"detect={loo.session['attack_detect']:.1%}")
    print(f"  delta                  : AUC{d_auc:+.4f}  detect{d_det:+.1%}")
    print("  The feature separates every attack class from benign traffic purely")
    print("  because post-onset windows are later in a session. Whatever it adds")
    print("  is measuring our onset distribution, not the device's behaviour.")

    print("\n-- identity whitelist (USBGuard-style VID:PID allowlist) --")
    idw = identity_whitelist_baseline(fit, test)
    results["identity_whitelist"] = idw
    print(f"  attack detection: {idw['attack_detect']:.1%}  "
          f"(expected 0% — attacks present benign descriptors)")

    banner("[4/5] RESULTS")
    print("\nWindow level, held-out test set")
    print(f"  {'model':28s} {'ROC-AUC':>8s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")
    for r in (full, s1_only):
        w = r.window
        print(f"  {r.name:28s} {w['auc']:8.4f} {w['precision']:7.3f} "
              f"{w['recall']:7.3f} {w['f1']:7.3f}")

    print(f"\nSession level (threshold @ {TARGET_FPR:.1%} validation FPR, "
          f"{GATE_K}-of-{GATE_N} gate)")
    print(f"  {'profile':18s} {'n':>3s} {'IUDBMS':>8s} {'S1-only':>8s} "
          f"{'whitelist':>10s} {'latency':>9s}")
    for profile, pp in sorted(full.per_profile.items(),
                              key=lambda kv: (-kv[1]["label"], kv[0])):
        if pp["label"] != 1:
            continue
        s1p = s1_only.per_profile.get(profile, {})
        lat = pp["median_latency"]
        print(f"  {profile:18s} {pp['n']:3d} {pp['detect_rate']:8.1%} "
              f"{s1p.get('detect_rate', 0):8.1%} {0.0:10.1%} "
              f"{(f'{lat:.1f}s' if lat is not None else '-'):>9s}")

    fs, ss = full.session, s1_only.session
    print(f"\n  {'overall attack detection':32s} IUDBMS={fs['attack_detect']:6.1%}   "
          f"S1-only={ss['attack_detect']:6.1%}   whitelist={idw['attack_detect']:6.1%}")
    print(f"  {'benign session FP rate':32s} IUDBMS={fs['benign_fp']:6.1%}   "
          f"S1-only={ss['benign_fp']:6.1%}")
    print(f"  {'alerts per benign hour':32s} IUDBMS={fs['fp_per_benign_hour']:6.2f}")
    if fs["median_latency"] is not None:
        print(f"  {'detection latency':32s} median={fs['median_latency']:.1f}s   "
              f"p90={fs['p90_latency']:.1f}s")

    banner("[4b/5] OPERATING POINT  the trade-off curve, not one number (plan §6)")
    from .operating_curve import format_curves, sweep_gates, sweep_thresholds
    thr_rows = sweep_thresholds(test, val, full,
                                [0.0001, 0.0005, TARGET_FPR, 0.005, 0.01, 0.02])
    gate_rows = sweep_gates(test, full, full.threshold,
                            [(2, 6), (3, 6), (GATE_K, GATE_N), (5, 6), (6, 6)])
    print(format_curves(thr_rows, gate_rows, full.threshold))
    print("\n  The curve is DESCRIPTIVE. The deployed point is the one calibrated on")
    print("  validation benign windows; reading a better point off this table and")
    print("  adopting it would be selecting on the test split.")

    _sanity_warnings(full, s1_only)

    banner("[5/5] ARTEFACTS")
    metrics = {
        "config": {"n_benign": args.n_benign, "n_attack": args.n_attack,
                   "seed": args.seed, "epochs": args.epochs,
                   "seq_len": args.seq_len, "hidden": args.hidden,
                   "target_fpr": TARGET_FPR, "gate": f"{GATE_K}-of-{GATE_N}"},
        "corpus": {"sessions": len(corpus.sessions), "usbmon_lines": corpus.n_lines,
                   "urb_events": corpus.n_events, "windows": corpus.n_windows,
                   "features": N_FEATURES},
        "split": {"fit": len(fit), "val": len(val), "test": len(test),
                  "policy": "by session, stratified by profile"},
        "full_system": {"threshold": full.threshold, "window": full.window,
                        "session": full.session, "per_profile": full.per_profile},
        "stage1_only": {"threshold": s1_only.threshold, "window": s1_only.window,
                        "session": s1_only.session, "per_profile": s1_only.per_profile},
        "modality_ablation": {m: {"window": r.window, "session": r.session}
                              for m, r in modality.items()},
        "deployed_features": len(deployed),
        "with_time_feature": {"window": loo.window, "session": loo.session,
                              "per_profile": loo.per_profile,
                              "delta_auc": d_auc, "delta_detect": d_det},
        "identity_whitelist": idw,
        "operating_curve": {"threshold_sweep": thr_rows, "gate_sweep": gate_rows},
        "runtime_sec": round(time.time() - t_start, 1),
    }
    path = OUT / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"  wrote {path.relative_to(OUT.parent)}")

    if args.persist:
        from .persist import persist_run
        db = persist_run(OUT / "iudbms.db", corpus.sessions, full)
        print(f"  wrote out/iudbms.db  {db}")

    if not args.no_figures:
        from .figures import write_figures
        for f in write_figures(OUT, full, s1_only, modality, test):
            print(f"  wrote {f.relative_to(OUT.parent)}")

    print(f"\n  total runtime {metrics['runtime_sec']}s")
    return 0


def _sanity_warnings(full, s1_only) -> None:
    """Flag results that are too good, rather than reporting them quietly.

    A window AUC of 1.000, or a perfect detection rate at a zero false-positive
    rate, is a leakage signal far more often than it is a win. The project has
    already had two duration leaks that presented exactly this way.
    """
    warnings = []
    if full.window["auc"] >= 0.9995:
        warnings.append(f"window AUC is {full.window['auc']:.4f} — effectively perfect")
    if full.session["attack_detect"] == 1.0 and full.session["benign_fp"] == 0.0:
        warnings.append("100% detection at 0% false positives on the same split")
    if full.history and full.history[0].get("val_acc", 0) >= 0.99:
        warnings.append(f"epoch-1 validation accuracy is "
                        f"{full.history[0]['val_acc']:.4f} — the task may be trivially separable")
    if warnings:
        print("\n  " + "!" * 68)
        print("  SUSPICIOUSLY GOOD RESULTS — investigate before reporting:")
        for w in warnings:
            print(f"    - {w}")
        print("  Re-run `python3 -m eval.leakage_audit` and check the split.")
        print("  " + "!" * 68)


if __name__ == "__main__":
    raise SystemExit(main())
