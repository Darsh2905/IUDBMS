"""
Train the DEPLOYED model set on the realistic corpus (Corpus B) -> out/models/.

    python3 -m eval.train_deployed          # or: make train-deployed

The deployed set used to be whatever ``make eval`` last wrote, trained on Corpus A,
whose benign devices never go idle. Scored on realistic traffic that model flags
46.6% of benign sessions and every keyboard, and it flags a real recorded typing
session (CLAUDE.md finding 32). The deployed set is now trained on Corpus B
(duty cycles, HID key-release/rollover reports, language-like text, per-device
speed): same architecture, seed, epochs, calibration rule (per-window FPR 1e-4,
benign validation windows only) and 4-of-6 gate as everything else. ``make eval``
keeps writing the Corpus-A set, to out/models_corpus_a/, so the headline stays
reproducible without ever touching the deployment.

The set is stamped with output fingerprints (``canary``) which the monitor checks at
start-up, so weights that do not belong to this threshold are refused.
"""

from __future__ import annotations

import json
import time

from .analysis_common import OUT, save_set
from .analysis_realistic import CACHE, build
from .corpus import Corpus, load_corpus, save_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

SEED, EPOCHS, TARGET_FPR = 7, 14, 0.0001


def main() -> int:
    t0 = time.time()
    meta = CACHE.with_suffix(".cond.json")
    if CACHE.exists() and meta.exists():
        sessions = load_corpus(CACHE).sessions
        print(f"Corpus B: {len(sessions)} sessions [cached]")
    else:
        sessions, cond = build()
        save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), CACHE)
        meta.write_text(json.dumps(cond))
        print(f"Corpus B: {len(sessions)} sessions [built in {time.time() - t0:.0f} s]")

    fit, val, test = stratified_split(sessions, seed=SEED)
    idx = deployed_features(fit)
    r = run_configuration(fit, val, test, name="IUDBMS (deployed)", feature_idx=idx, seed=SEED,
                          epochs=EPOCHS, seq_len=10, hidden=32, target_fpr=TARGET_FPR,
                          keep_scores=True, verbose=False)
    dest = save_set(OUT / "models", r, idx, target_fpr=TARGET_FPR)
    s = r.session
    print(f"trained in {time.time() - t0:.0f} s  ->  {dest}")
    print(f"  threshold {r.threshold:.4f}   window AUC {r.window['auc']:.4f}")
    print(f"  Corpus-B test split: detection {s['attack_detect']:.1%}  benign FP {s['benign_fp']:.1%}  "
          f"median latency {s['median_latency']:.1f} s")
    (OUT / "deployed_models.json").write_text(json.dumps(
        {"trained_on": "Corpus B (realistic), seed 7", "threshold": r.threshold,
         "window_auc": r.window["auc"], "session": s, "per_profile": r.per_profile}, indent=2,
        default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
