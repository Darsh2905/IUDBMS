"""
Pillar N2 (per-device adaptive baseline) — a fair test, under device variation.

    python3 -m eval.run_baseline_ablation [--n-benign 600] [--n-attack 400]

Why this script exists
----------------------
CLAUDE.md finding 14 measured N2 and did not support it: removing the per-device
EWMA baseline left full-system detection unchanged within noise and marginally
*better* (90.8% vs 89.2%). The note recorded a hypothesis for why, and declined
to act on it:

    "the forests are already class-conditioned, so a device's class-specific
    normality is largely modelled at population level, leaving the per-device
    term little to add on a corpus where devices within a class are
    near-identical. Real devices vary more, so N2 may earn its keep on real
    captures -- which is a hypothesis, not a defence."

That hypothesis is testable without real captures, because the thing it names —
device homogeneity — is a property of the simulator, not of reality. Every
keyboard in the corpus drew its timing from one shared parameter set, so a
class-conditioned forest can model all of them at once and a per-device
baseline is structurally redundant.

``Simulator(device_variation=True)`` gives each device its own persona drawn
from a hyperprior (typing speed, burst length, pause tendency). Benign and
malicious devices draw alike — if only benign devices varied, "behaves like the
population mean" would itself become a tell.

What this measures
------------------
The N2 ablation (full fusion vs forest-only, ``weights=(1.0, 0.0)``) run on two
corpora that differ in exactly one property:

    homogeneous    every device of a class behaves identically  (status quo)
    heterogeneous  every device draws its own persona

N2 is **supported** iff, on the heterogeneous corpus, the full system beats
forest-only by more than the noise floor (1 / n attack sessions) at session
level, having not done so on the homogeneous one. Any other outcome is reported
as it falls. In particular, if N2 is inert under heterogeneity too, that is a
stronger negative result than finding 14's and should replace it.

Do not rescue this by sweeping the fusion weight. Finding 14 refused that and
the refusal is why the rest of the numbers are worth anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iudbms.capture.simulator import SimulatedSource, Simulator
from iudbms.contract import FEATURE_NAMES, N_FEATURES
from iudbms.features.extractor import WindowFeatureExtractor, window_events

from .corpus import Corpus, SessionData, stratified_split
from .pipeline import deployed_features, run_configuration

OUT = Path(__file__).resolve().parent.parent / "out"


def build(n_benign: int, n_attack: int, seed: int, *, variation: bool) -> Corpus:
    sim = Simulator(seed=seed, device_variation=variation)
    sessions: list[SessionData] = []
    n_lines = n_events = 0
    for sid, profile in sim.plan(n_benign=n_benign, n_attack=n_attack):
        s = sim.session(sid, profile)
        ex = WindowFeatureExtractor()
        n_lines += len(s.lines)
        events = list(SimulatedSource(s).events())
        n_events += len(events)
        X, t, y = [], [], []
        for t0, bucket in window_events(events, duration=s.duration):
            X.append(ex.extract(bucket, t0, s.descriptor))
            t.append(t0)
            y.append(int(s.label == 1 and s.onset is not None
                         and (t0 + ex.WINDOW) > s.onset))
        s.lines = []
        sessions.append(SessionData(
            sid=sid, profile=profile, label=s.label, descriptor=s.descriptor,
            X=np.vstack(X), t=np.asarray(t), y=np.asarray(y),
            onset=s.onset, duration=s.duration))
    return Corpus(sessions=sessions, n_lines=n_lines, n_events=n_events)


def heterogeneity(corpus: Corpus) -> float:
    """Between-device spread of event rate within the keyboard class.

    Reported so the manipulation is visible as a number rather than asserted:
    if this does not move between the two corpora, the experiment did nothing.
    """
    rates = [float(s.X[:, 1].mean()) for s in corpus.sessions
             if s.profile == "keyboard"]
    return float(np.std(rates)) if rates else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-benign", type=int, default=600)
    ap.add_argument("--n-attack", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--target-fpr", type=float, default=0.0001)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    deployed = None   # set per corpus, from its own training split

    out = {}
    print(f"{'corpus':16s} {'kbd rate sd':>12s} {'configuration':22s} "
          f"{'AUC':>8s} {'detect':>9s} {'FP':>7s}")
    print("-" * 80)
    for label, variation in (("homogeneous", False), ("heterogeneous", True)):
        c = build(args.n_benign, args.n_attack, args.seed, variation=variation)
        spread = heterogeneity(c)
        fit, val, test = stratified_split(c.sessions, seed=args.seed)
        deployed = deployed_features(fit, verbose=False)
        n_atk = sum(1 for s in test if s.label == 1)
        noise = 1.0 / max(n_atk, 1)

        rows = {}
        for name, use_baseline in (("full (forest + baseline)", True),
                                   ("forest only", False)):
            r = run_configuration(fit, val, test, name=f"{label}/{name}",
                                  feature_idx=deployed, seed=args.seed,
                                  epochs=args.epochs, target_fpr=args.target_fpr,
                                  use_device_baseline=use_baseline, verbose=False)
            rows[name] = {"auc": r.window["auc"],
                          "detect": r.session["attack_detect"],
                          "benign_fp": r.session["benign_fp"]}
            print(f"{label:16s} {spread:12.4f} {name:22s} "
                  f"{r.window['auc']:8.4f} {r.session['attack_detect']:8.1%} "
                  f"{r.session['benign_fp']:6.1%}")

        margin = rows["full (forest + baseline)"]["detect"] - rows["forest only"]["detect"]
        out[label] = {"keyboard_rate_sd": spread, "n_attack_test": n_atk,
                      "noise_floor": noise, "margin": margin, "rows": rows,
                      "baseline_helps": margin > noise}
        print(f"{'':16s} {'':12s} {'-> baseline contributes':22s} "
              f"{margin:+.1%}  (noise {noise:.1%})"
              f"  {'HELPS' if margin > noise else 'inert'}\n")

    hom, het = out["homogeneous"], out["heterogeneous"]
    if het["keyboard_rate_sd"] <= hom["keyboard_rate_sd"] * 1.5:
        print("  WARNING: the heterogeneous corpus is not meaningfully more varied.")
        print("  The manipulation failed; the comparison below means nothing.")

    supported = het["baseline_helps"] and not hom["baseline_helps"]
    print(f"  homogeneous   : baseline {'helps' if hom['baseline_helps'] else 'inert'} "
          f"({hom['margin']:+.1%} vs noise {hom['noise_floor']:.1%})")
    print(f"  heterogeneous : baseline {'helps' if het['baseline_helps'] else 'inert'} "
          f"({het['margin']:+.1%} vs noise {het['noise_floor']:.1%})")
    print(f"\n  ==> PILLAR N2 {'SUPPORTED under device heterogeneity' if supported else 'NOT SUPPORTED'}.")
    if not supported and het["baseline_helps"]:
        print("      (the baseline helps on BOTH corpora -- finding 14's premise, that")
        print("       it is inert, does not reproduce; investigate before reporting)")
    if not supported and not het["baseline_helps"]:
        print("      Device homogeneity was finding 14's stated explanation for N2")
        print("      being inert. Removing it does not rescue the pillar, so that")
        print("      explanation is wrong and N2's negative result is stronger than")
        print("      previously reported. Do not sweep the fusion weight.")

    (OUT / "baseline_ablation.json").write_text(json.dumps({
        "n_benign": args.n_benign, "n_attack": args.n_attack, "seed": args.seed,
        "target_fpr": args.target_fpr, "results": out, "supported": supported,
    }, indent=2, default=float))
    print(f"\n  wrote out/baseline_ablation.json    ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
