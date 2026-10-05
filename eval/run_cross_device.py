"""
Cross-device correlation — does host context close the mimicry_adaptive bound?

    python3 -m eval.run_cross_device [--n-hosts 500]

Why this script exists
----------------------
CLAUDE.md finding 11 states the project's sharpest negative result. A BadUSB
that types its payload once and stops (``mimicry_adaptive``) is detected at
40%, and a per-feature KS scan separates it from benign keyboard traffic on
**zero** of the 28 deployed features. The note calls the bound architectural
rather than incidental:

    "Within single-device URB behaviour there is no signal -- the adversary is
    statistically the same process as a human typing. The principled next
    defence is cross-device correlation: a human typing produces concurrent
    mouse activity, a BadUSB typing alone does not."

This script builds that defence and measures it. Contract 3.0.0 appends
``concurrent_device_rate``: the log1p event rate of every *other* device on the
same host during this window, supplied through ``HostContext``.

The leakage guard, built before the feature
-------------------------------------------
This feature is the most obvious candidate in the project to become the next
``log_time_since_enum`` — a column that encodes how the corpus was assembled
rather than how a device behaves. It becomes one the moment host composition
correlates with the label. So ``Simulator.P_COMPANION`` is applied identically
to benign and malicious hosts, and both of these must occur on both sides:

  * a person typing at a machine with no mouse attached (lone benign host)
  * a BadUSB plugged into a workstation whose owner is still working

The script asserts the guard held, by measuring companion prevalence per label
before reporting any detection number. If attacked hosts are lonelier than
benign ones the experiment is void and says so.

The comparison
--------------
One corpus of multi-device hosts, scored twice — with and without the host
column. Nothing else differs, so any change is attributable to the feature.
The headline cell is ``mimicry_adaptive``; the guard cell is benign FP.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from iudbms.capture.simulator import (
    ATTACK_PROFILES,
    SimulatedSource,
    Simulator,
)
from iudbms.contract import FEATURE_NAMES, N_FEATURES, feature_index
from iudbms.features.extractor import HostContext, WindowFeatureExtractor, window_events

from .corpus import Corpus, SessionData, stratified_split
from .leakage_audit import ks_statistic
from .pipeline import run_configuration

OUT = Path(__file__).resolve().parent.parent / "out"

WINDOW = WindowFeatureExtractor.WINDOW
HOST_COL = feature_index("concurrent_device_rate")

#: The keyboard on each host is drawn from these. Benign keyboards and attack
#: profiles alike sit on hosts assembled by the same rule.
HOST_KEYBOARDS = ("keyboard",) + ATTACK_PROFILES


def build_host_corpus(n_hosts: int, seed: int, *, verbose: bool = True
                      ) -> tuple[Corpus, dict]:
    """Multi-device hosts, with every device's features carrying host context."""
    sim = Simulator(seed=seed)
    rng = np.random.default_rng(seed)
    sessions: list[SessionData] = []
    n_lines = n_events = 0
    companion = {"benign": [0, 0], "attack": [0, 0]}   # [with, without]

    for h in range(n_hosts):
        profile = HOST_KEYBOARDS[int(rng.integers(len(HOST_KEYBOARDS)))]
        devices = sim.host_session(f"H{h:04d}", profile)
        side = "benign" if devices[0].label == 0 else "attack"
        companion[side][0 if len(devices) > 1 else 1] += 1

        # --- per-device events, then the host join -------------------------
        per_dev = {}
        for s in devices:
            ev = list(SimulatedSource(s).events())
            n_lines += len(s.lines)
            n_events += len(ev)
            s.lines = []
            per_dev[s.descriptor.device_key] = (s, ev)

        # events/sec per (device, window), then "every other device" by
        # subtracting this device's own contribution from the host total.
        by_win: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for key, (_s, ev) in per_dev.items():
            for e in ev:
                if e.evt in ("C", "E"):
                    by_win[int(e.ts // WINDOW)][key] += 1
        rate = {}
        for w, counts in by_win.items():
            total = sum(counts.values())
            for key in per_dev:
                rate[(key, w)] = (total - counts.get(key, 0)) / WINDOW
        host_ctx = HostContext(rate)

        for key, (s, ev) in per_dev.items():
            ex = WindowFeatureExtractor()
            X, t, y = [], [], []
            for t0, bucket in window_events(ev, duration=s.duration):
                X.append(ex.extract(bucket, t0, s.descriptor, host=host_ctx))
                t.append(t0)
                y.append(int(s.label == 1 and s.onset is not None
                             and (t0 + ex.WINDOW) > s.onset))
            sessions.append(SessionData(
                sid=s.sid, profile=s.profile, label=s.label, descriptor=s.descriptor,
                X=np.vstack(X), t=np.asarray(t), y=np.asarray(y),
                onset=s.onset, duration=s.duration))

    guard = {
        "benign_with_companion": companion["benign"][0] / max(sum(companion["benign"]), 1),
        "attack_with_companion": companion["attack"][0] / max(sum(companion["attack"]), 1),
        "n_benign_hosts": sum(companion["benign"]),
        "n_attack_hosts": sum(companion["attack"]),
    }
    if verbose:
        print(f"  hosts           : {n_hosts}  -> {len(sessions)} device sessions")
        print(f"  usbmon lines    : {n_lines:,}")
        print(f"  companion rate  : benign {guard['benign_with_companion']:.1%}  "
              f"attack {guard['attack_with_companion']:.1%}")
    return Corpus(sessions=sessions, n_lines=n_lines, n_events=n_events), guard


def check_guard(corpus: Corpus, guard: dict) -> bool:
    """Did host composition stay independent of the label?

    Two checks. The coarse one compares companion prevalence per side. The sharp
    one is a KS test of the feature itself across benign windows of each side --
    if the feature's distribution already separates benign from attack *hosts*
    before any model sees it, the feature is reading corpus construction.
    """
    gap = abs(guard["benign_with_companion"] - guard["attack_with_companion"])
    ok_prev = gap < 0.10

    # Compare LIKE WITH LIKE: benign *keyboard* windows against pre-onset
    # windows of attacked keyboards. The first version of this check pooled all
    # benign sessions, which includes the companion mice -- whose host feature
    # is a keyboard's ~2/s while a keyboard's is a mouse's ~100/s -- and so
    # reported D=0.32 against a 0.066 critical value for a population
    # difference that has nothing to do with the label.
    ben = np.vstack([s.X[:, [HOST_COL]] for s in corpus.sessions
                     if s.label == 0 and s.profile == "keyboard"])
    atk_pre = [s.X[s.y == 0][:, [HOST_COL]] for s in corpus.sessions
               if s.label == 1 and (s.y == 0).any()]
    ok_ks, d, crit = True, float("nan"), float("nan")
    if atk_pre:
        pre = np.vstack(atk_pre)
        d, crit = ks_statistic(ben[:, 0], pre[:, 0])
        ok_ks = d <= crit

    print(f"\n-- leakage guard on concurrent_device_rate --")
    print(f"  companion prevalence gap     : {gap:.1%}   "
          f"{'ok' if ok_prev else 'FAIL (composition tracks the label)'}")
    print(f"  KS, benign kbd vs PRE-onset attack kbd: D={d:.4f} crit={crit:.4f}   "
          f"{'ok' if ok_ks else 'FAIL (feature separates before the attack starts)'}")
    if not (ok_prev and ok_ks):
        print("  The feature is reading how the corpus was built. Any detection")
        print("  gain below is an artefact; fix the generator before believing it.")
    return ok_prev and ok_ks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-hosts", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27],
                    help="corpora are redrawn per seed; a one-session delta on a "
                         "single draw is the noise floor, not an improvement")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--target-fpr", type=float, default=0.0001)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    without = np.array([i for i in range(N_FEATURES)
                        if FEATURE_NAMES[i] not in ("log_time_since_enum",
                                                    "concurrent_device_rate")])
    with_host = np.array([i for i in range(N_FEATURES)
                          if FEATURE_NAMES[i] != "log_time_since_enum"])

    reps, guards = [], []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        corpus, guard = build_host_corpus(args.n_hosts, seed)
        guard_ok = check_guard(corpus, guard)
        guards.append({"seed": seed, "guard": guard, "ok": guard_ok})

        fit, val, test = stratified_split(corpus.sessions, seed=seed)
        n_by_profile = {p: sum(1 for s_ in test if s_.profile == p)
                        for p in sorted({s_.profile for s_ in test})}
        rows = {}
        for name, idx in (("without", without), ("with", with_host)):
            r = run_configuration(fit, val, test, name=name, feature_idx=idx,
                                  seed=seed, epochs=args.epochs,
                                  target_fpr=args.target_fpr, verbose=False)
            rows[name] = {"n_features": len(idx), "auc": r.window["auc"],
                          "detect": r.session["attack_detect"],
                          "benign_fp": r.session["benign_fp"],
                          "per_profile": {p: v.get("detect_rate")
                                          for p, v in r.per_profile.items()
                                          if v.get("label") == 1}}
        reps.append({"seed": seed, "guard_ok": guard_ok, "rows": rows,
                     "n_by_profile": n_by_profile})
        a_, b_ = rows["without"], rows["with"]
        print(f"  without {a_['detect']:6.1%} (FP {a_['benign_fp']:.1%})   "
              f"with {b_['detect']:6.1%} (FP {b_['benign_fp']:.1%})   "
              f"mimicry_adaptive "
              f"{(a_['per_profile'].get('mimicry_adaptive') or 0):.1%} -> "
              f"{(b_['per_profile'].get('mimicry_adaptive') or 0):.1%}")

    # ---- aggregate ---------------------------------------------------------
    guard_ok = all(g["ok"] for g in guards)
    profiles = sorted(reps[0]["rows"]["without"]["per_profile"])
    print("\n-- cross-device correlation, mean over seeds "
          f"{args.seeds} (held-out test splits) --")
    print(f"  {'profile':20s} {'without':>14s} {'with':>14s} {'delta':>14s} {'noise':>7s}")
    summary = {}
    for p in profiles:
        va = np.array([r["rows"]["without"]["per_profile"].get(p) or 0.0 for r in reps])
        vb = np.array([r["rows"]["with"]["per_profile"].get(p) or 0.0 for r in reps])
        n_p = float(np.mean([r["n_by_profile"].get(p, 0) for r in reps]))
        noise = 1.0 / max(n_p, 1)
        d = vb - va
        summary[p] = {"without": float(va.mean()), "with": float(vb.mean()),
                      "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
                      "noise": noise, "n_test": n_p}
        print(f"  {p:20s} {va.mean():8.1%}+-{va.std():4.1%} "
              f"{vb.mean():8.1%}+-{vb.std():4.1%} "
              f"{d.mean():+8.1%}+-{d.std():4.1%} {noise:6.1%}")

    da = summary["mimicry_adaptive"]["delta_mean"]
    da_sd = summary["mimicry_adaptive"]["delta_sd"]
    noise_a = summary["mimicry_adaptive"]["noise"]
    ov = np.array([r["rows"]["with"]["detect"] - r["rows"]["without"]["detect"]
                   for r in reps])

    # A delta counts only if it clears the noise floor AND its own spread across
    # seeds. The first single-seed run showed +4.2% against a 4.2% noise floor --
    # exactly one session of 24, which floating point alone rendered as a win.
    EPS = 1e-9
    helps = guard_ok and (da - noise_a) > EPS and da > da_sd

    print(f"\n  mimicry_adaptive : {da:+.1%} +- {da_sd:.1%}   "
          f"(noise floor {noise_a:.1%})")
    print(f"  overall          : {ov.mean():+.1%} +- {ov.std():.1%}")
    print(f"\n  ==> cross-device correlation "
          f"{'HELPS on the single-shot bound' if helps else 'does NOT clear the bound'}.")
    if not guard_ok:
        print("      (void: the leakage guard failed on at least one seed)")
    elif not helps:
        print("      The bound survives host context, and the reason is specific.")
        print()
        print("      `Simulator.host_session` generates each device INDEPENDENTLY --")
        print("      the companion mouse is drawn without reference to what the")
        print("      keyboard is doing. So in this corpus a benign keyboard and its")
        print("      mouse are statistically independent, exactly like a BadUSB and")
        print("      a bystander's mouse. The coupling this feature exists to detect")
        print("      is not present in the data on EITHER side, so no feature over")
        print("      these streams could separate them.")
        print()
        print("      What a real human produces, and this corpus does not, is")
        print("      temporal interleaving: one agent alternates between typing and")
        print("      mousing, so the two streams are anti-correlated at sub-second")
        print("      scale, while two independent agents overlap at the chance rate.")
        print("      Testing that needs (1) a coupled benign generator and (2) a")
        print("      simultaneity feature rather than a rate.")
        print()
        print("      Note the trap in doing so: building the coupling into the")
        print("      simulator and then detecting it is circular. Human keyboard/")
        print("      mouse anti-correlation is a real HCI property, but asserting its")
        print("      parameters without a real capture would manufacture the result.")
        print("      This is the point where the question needs real traffic.")

    (OUT / "cross_device.json").write_text(json.dumps({
        "n_hosts": args.n_hosts, "seeds": args.seeds, "guards": guards,
        "guard_ok": guard_ok, "per_profile": summary, "reps": reps,
        "mimicry_adaptive_delta": da, "mimicry_adaptive_delta_sd": da_sd,
        "overall_delta_mean": float(ov.mean()), "overall_delta_sd": float(ov.std()),
        "noise_floor": noise_a, "helps": helps,
    }, indent=2, default=float))
    print(f"\n  wrote out/cross_device.json    ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
