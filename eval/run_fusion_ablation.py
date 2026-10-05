"""
Pillar N1 (multi-modal fusion) — a fair test, on single-axis attack families.

    python3 -m eval.run_fusion_ablation [--per-profile 40]

Why this script exists
----------------------
The headline modality ablation in ``run_eval.py`` has never supported N1: the
payload modality alone matches the fused model exactly (AUC 0.9937, 100%). The
reason is a property of the corpus, not a fact about fusion. Every profile in
``ATTACK_PROFILES`` deviates from benign traffic on several modalities at once
— ``hid_injection`` is faster (timing) *and* scripted (payload) *and* bursty
(timing) — so any one modality is sufficient and fusion cannot demonstrate a
gain it may well possess.

A fair test needs attack families that deviate on **one axis at a time**. That
is what ``AXIS_PROFILES`` are for. Each holds three modalities at benign values
and moves the fourth.

Two instruments, because they answer different questions
--------------------------------------------------------
The four **single-axis** profiles cannot test fusion. If an axis is detectable
by its own modality — which is what makes it single-axis — the fused model can
at best tie, never beat. What they *do* test is that fusion does not lose what
a specialist sees, which is a real failure mode: 28 features diluting the 6 that
matter.

``axis_diffuse`` is the actual N1 instrument: a deviation applied to two axes at
once (timing and payload — the only two with real benign variance for a
keyboard), each weak enough to stay inside the benign distribution. That is
fusion's claim, and the realistic shape of a careful adversary.

Its strength is **swept, not tuned**. A single calibrated strength proved
unusable: the selection split holds too few diffuse sessions to place it
reliably (4/8 vs 2/8 vs 0/8 sessions is noise, not a calibration curve). So the
script reports fused-vs-specialist across the whole subtlety range and prints
every point. Reporting a complete sweep is not selection-on-test; selecting the
best point and reporting only that would be, and is refused.

The pre-registered criterion (fixed before the first run)
---------------------------------------------------------
N1 is **supported** iff, on the held-out test split:

  (a) on every single-axis profile, fused detection >= the best non-degenerate
      single modality's, and
  (b) at SOME swept strength where the comparison is valid, fused detection on
      axis_diffuse exceeds EVERY non-degenerate single modality's by more than
      the noise floor (1 / n sessions of that profile).

A strength is a valid comparison point only when the best specialist is inside
``MARGINAL_BAND``: below it nothing sees the attack and fusion's margin is
vacuous; above it a specialist alone suffices and the attack was never diffuse.

A configuration whose benign session FP exceeds ``MAX_BENIGN_FP`` is
**degenerate** — it alarms rather than detects — and its detection numbers are
excluded from the comparison rather than counted as a win. A 6-feature model
that flags half the benign keyboards has not detected anything.

Two ways this can fail to be a fair test, both checked and reported:
  * if NO single modality detects axis_diffuse, the attack is too weak to
    discriminate and fusion's win is vacuous;
  * if EVERY single modality detects it, the deviation was not subtle and the
    instrument has the same flaw as the headline corpus.

Do not relax this criterion after seeing the numbers, and do not sweep fusion
weights or ``DIFFUSE_STRENGTH`` until it passes. The project has refused that
move twice already (CLAUDE.md findings 13 and 14) and the refusal is the reason
the other numbers are worth anything.

Axis purity is measured, not assumed
------------------------------------
A profile is only a single-axis instrument if it really does move one
modality's features and leave the rest alone. Some coupling is structural:
``class_behaviour_mismatch`` is *defined* over a protocol/identity interaction,
and byte-rate features follow event rate by construction. So the script runs a
per-feature two-sample KS scan of each axis against benign keyboard windows and
prints which features actually separate, grouped by modality. Read that table
before the detection table: a contaminated axis makes its column uninterpretable.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from iudbms.capture.simulator import (
    AXIS_PROFILES,
    SimulatedSource,
    Simulator,
)
from iudbms.contract import FEATURE_NAMES, MODALITIES, N_FEATURES, modality_indices
from iudbms.features.extractor import WindowFeatureExtractor, window_events

from .corpus import Corpus, SessionData, stratified_split
from .leakage_audit import ks_statistic
from .pipeline import deployed_features, run_configuration

OUT = Path(__file__).resolve().parent.parent / "out"

#: The axis corpus is keyboard-only on the benign side. A fair single-axis test
#: compares "keyboard that deviates on one axis" against "keyboard", not
#: against a population containing webcams and mass storage — otherwise the
#: identity axis is trivially separable from the class mix alone.
AXIS_BENIGN = ("keyboard",)

#: A configuration flagging more than this fraction of benign sessions is
#: alarming, not detecting; its per-axis detection is excluded from the
#: comparison. Deliberately generous -- the deployed system runs at 0.0%.
MAX_BENIGN_FP = 0.05

#: The axis whose detection tests fusion proper. The other four test that
#: fusion does not lose what a specialist sees.
DIFFUSE = "axis_diffuse"

#: axis_diffuse is a valid fusion instrument only when every single modality
#: finds it genuinely marginal: the best specialist must land inside this band.
#: Below it, nothing can see the attack and fusion's margin is vacuous; above
#: it, a specialist alone suffices and the attack was never diffuse.
MARGINAL_BAND = (0.15, 0.75)
SINGLE_AXES = tuple(a for a in AXIS_PROFILES if a != DIFFUSE)


def build_axis_corpus(per_profile: int, seed: int = 7, *, verbose: bool = True,
                      diffuse_strength: float | None = None,
                      benign_multiple: int = 5) -> Corpus:
    """One balanced corpus of keyboards and single-axis deviants.

    ``benign_multiple`` sizes the benign pool at ``per_profile * k``. It needs to
    be large: the decision threshold is the 1-1e-4 quantile of benign window
    scores, and estimating that from one profile's worth of sessions puts it
    essentially at the maximum, where it has enormous variance. At k=1 the
    fused configuration landed at 7.4% benign session FP against 0.0% for the
    single modalities -- which would make any detection comparison between them
    a comparison of operating points rather than of feature sets.
    """
    sim = (Simulator(seed=seed) if diffuse_strength is None
           else Simulator(seed=seed, diffuse_strength=diffuse_strength))
    sessions: list[SessionData] = []
    n_lines = n_events = 0

    plan = sim.plan_balanced(per_profile, benign_profiles=AXIS_BENIGN,
                             attack_profiles=AXIS_PROFILES,
                             benign_multiple=benign_multiple)
    for sid, profile in plan:
        s = sim.session(sid, profile)
        ex = WindowFeatureExtractor()
        n_lines += len(s.lines)
        events = list(SimulatedSource(s).events())
        n_events += len(events)
        X, t, y = [], [], []
        for t0, bucket in window_events(events, duration=s.duration):
            X.append(ex.extract(bucket, t0, s.descriptor))
            t.append(t0)
            malicious = s.label == 1 and s.onset is not None and (t0 + ex.WINDOW) > s.onset
            y.append(int(malicious))
        s.lines = []
        sessions.append(SessionData(
            sid=sid, profile=profile, label=s.label, descriptor=s.descriptor,
            X=np.vstack(X), t=np.asarray(t), y=np.asarray(y),
            onset=s.onset, duration=s.duration))

    if verbose:
        n_b = sum(1 for s in sessions if s.label == 0)
        print(f"  axis corpus     : {len(sessions)} sessions "
              f"({n_b} keyboard / {len(sessions) - n_b} single-axis)")
        print(f"  usbmon lines    : {n_lines:,}")
        print(f"  windows         : {sum(len(s.X) for s in sessions):,}")
    return Corpus(sessions=sessions, n_lines=n_lines, n_events=n_events)


def purity_scan(corpus: Corpus, deployed: np.ndarray) -> dict:
    """Per-feature KS of each axis's malicious windows against benign windows.

    Returns, per axis, the features that separate at alpha=0.01 grouped by the
    modality they belong to. An axis is 'pure' when every separating feature
    belongs to its own modality.
    """
    benign = np.vstack([s.X for s in corpus.sessions if s.label == 0])
    owner = {"axis_timing": "timing", "axis_payload": "payload",
             "axis_protocol": "protocol", "axis_identity": "identity"}
    report = {}

    print("\n-- axis purity (per-feature KS vs benign keyboard windows, a=0.01) --")
    for axis in SINGLE_AXES:
        rows = [s for s in corpus.sessions if s.profile == axis]
        mal = np.vstack([s.X[s.y == 1] for s in rows if (s.y == 1).any()])
        moved: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for i in deployed:
            d, crit = ks_statistic(benign[:, i], mal[:, i])
            if d > crit:
                moved[_modality_of_index(i)].append((FEATURE_NAMES[i], float(d)))

        own = owner[axis]
        off = {m: v for m, v in moved.items() if m != own}
        n_own, n_off = len(moved.get(own, [])), sum(len(v) for v in off.values())
        verdict = "PURE" if not off else f"contaminated by {', '.join(sorted(off))}"
        print(f"\n  {axis:14s} owns '{own}' : {n_own} own-modality features move, "
              f"{n_off} off-axis  -> {verdict}")
        for m in MODALITIES:
            if m in moved:
                top = ", ".join(f"{n}({d:.2f})" for n, d in
                                sorted(moved[m], key=lambda kv: -kv[1])[:4])
                mark = " " if m == own else "*"
                print(f"      {mark}{m:9s} {top}")
        report[axis] = {"owns": own,
                        "moved": {m: [[n, d] for n, d in v] for m, v in moved.items()},
                        "n_own": n_own, "n_off_axis": n_off, "pure": not off}

    if any(not r["pure"] for r in report.values()):
        print("\n  '*' marks an off-axis modality. Some coupling is structural: byte-rate")
        print("  features follow event rate, and class_behaviour_mismatch is defined over")
        print("  a protocol/identity interaction. Read the detection table accordingly.")
    return report


def _modality_of_index(i: int) -> str:
    for m in MODALITIES:
        if i in modality_indices(m):
            return m
    raise KeyError(i)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-profile", type=int, default=90,
                    help="sessions per axis profile")
    ap.add_argument("--benign-multiple", type=int, default=5,
                    help="benign sessions = per-profile * this; see build_axis_corpus")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.33, 0.50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27],
                    help="corpora are redrawn per seed; a margin that does not "
                         "survive reseeding is draw-to-draw variance, not fusion")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--target-fpr", type=float, default=0.0001)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    base = build_axis_corpus(args.per_profile, seed=args.seed,
                             benign_multiple=args.benign_multiple)
    base_fit, _bv, _bt = stratified_split(base.sessions, seed=args.seed)
    deployed = deployed_features(base_fit)
    dep = set(int(i) for i in deployed)
    configs = [("fused", deployed)] + [
        (m, np.array([i for i in modality_indices(m) if i in dep])) for m in MODALITIES]

    # ---- purity, measured once on the strongest corpus --------------------
    purity = purity_scan(base, deployed)

    # ---- the sweep, replicated across seeds --------------------------------
    # The first single-seed run of this sweep was non-monotone: fused scored
    # 18.5% at strength 0.20 and 88.9% at 0.33 while the best specialist sat
    # flat at 66.7%. A margin that moves that much between adjacent strengths is
    # draw-to-draw variance, and reporting the favourable point would be exactly
    # the error CLAUDE.md finding 13 records. Each strength is therefore redrawn
    # under several seeds and the spread is printed next to the mean.
    print("\n-- fused vs best specialist on axis_diffuse, across attack subtlety --")
    print(f"   mean over seeds {args.seeds}; every point reported, none selected\n")
    head = (f"{'strength':>9s} {'fused':>14s} {'best spec':>14s} {'margin':>16s} "
            f"{'fusedFP':>8s}  verdict")
    print(head); print("-" * len(head))

    sweep = []
    for strength in args.strengths:
        reps = []
        for seed in args.seeds:
            c = build_axis_corpus(args.per_profile, seed=seed, verbose=False,
                                  diffuse_strength=strength,
                                  benign_multiple=args.benign_multiple)
            fit, val, test = stratified_split(c.sessions, seed=seed)
            n_d = sum(1 for s_ in test if s_.profile == DIFFUSE)
            row = {}
            for name, idx in configs:
                r = run_configuration(fit, val, test, name=name, feature_idx=idx,
                                      seed=seed, epochs=args.epochs,
                                      target_fpr=args.target_fpr, verbose=False)
                row[name] = {
                    "diffuse": r.per_profile.get(DIFFUSE, {}).get("detect_rate") or 0.0,
                    "benign_fp": r.session["benign_fp"]}
            ok = [m for m in MODALITIES if row[m]["benign_fp"] <= MAX_BENIGN_FP]
            spec = max((row[m]["diffuse"] for m in ok), default=0.0)
            reps.append({"seed": seed, "fused": row["fused"]["diffuse"],
                         "best_spec": spec, "n_diffuse_test": n_d,
                         "fused_fp": row["fused"]["benign_fp"],
                         "ok": ok, "detail": row})

        f_mu = float(np.mean([r["fused"] for r in reps]))
        s_mu = float(np.mean([r["best_spec"] for r in reps]))
        f_sd = float(np.std([r["fused"] for r in reps]))
        s_sd = float(np.std([r["best_spec"] for r in reps]))
        margins = [r["fused"] - r["best_spec"] for r in reps]
        m_mu, m_sd = float(np.mean(margins)), float(np.std(margins))
        fp_mu = float(np.mean([r["fused_fp"] for r in reps]))
        noise = 1.0 / max(np.mean([r["n_diffuse_test"] for r in reps]), 1)

        fused_ok = fp_mu <= MAX_BENIGN_FP
        in_band = MARGINAL_BAND[0] <= s_mu < MARGINAL_BAND[1]
        valid = fused_ok and in_band
        # A margin counts only if it survives reseeding: bigger than the noise
        # floor AND bigger than its own spread across seeds.
        wins = valid and m_mu > noise and m_mu > m_sd
        loses = valid and -m_mu > noise and -m_mu > m_sd

        verdict = ("fusion WINS" if wins else
                   "fusion LOSES" if loses else
                   "tie (margin within its own seed spread)" if valid else
                   "invalid: fused degenerate" if not fused_ok else
                   "invalid: nothing sees it" if s_mu < MARGINAL_BAND[0] else
                   "invalid: specialist saturates")
        print(f"{strength:9.2f} {f_mu:8.1%}+-{f_sd:4.1%} {s_mu:8.1%}+-{s_sd:4.1%} "
              f"{m_mu:+9.1%}+-{m_sd:4.1%} {fp_mu:8.1%}  {verdict}")
        sweep.append({"strength": strength, "fused_mean": f_mu, "fused_sd": f_sd,
                      "best_spec_mean": s_mu, "best_spec_sd": s_sd,
                      "margin_mean": m_mu, "margin_sd": m_sd, "noise": noise,
                      "fused_benign_fp_mean": fp_mu, "valid": valid,
                      "fusion_wins": wins, "fusion_loses": loses,
                      "verdict": verdict, "reps": reps})

    # ---- condition (a), on the strongest corpus ---------------------------
    fit, val, test = stratified_split(base.sessions, seed=args.seed)
    n_axis_test = {a: sum(1 for s in test if s.profile == a) for a in AXIS_PROFILES}
    print("\n-- (a) does fusion lose what a specialist sees? --")
    base_row = {}
    for name, idx in configs:
        r = run_configuration(fit, val, test, name=name, feature_idx=idx,
                              seed=args.seed, epochs=args.epochs,
                              target_fpr=args.target_fpr, verbose=False)
        base_row[name] = {"per_axis": {a: r.per_profile.get(a, {}).get("detect_rate")
                                       for a in AXIS_PROFILES},
                          "benign_fp": r.session["benign_fp"]}
    ok_b = [m for m in MODALITIES if base_row[m]["benign_fp"] <= MAX_BENIGN_FP]
    cond_a, detail_a = True, []
    for a in SINGLE_AXES:
        f = base_row["fused"]["per_axis"][a] or 0.0
        b = max((base_row[m]["per_axis"][a] or 0.0) for m in ok_b) if ok_b else 0.0
        who = max(ok_b, key=lambda m: base_row[m]["per_axis"][a] or 0.0) if ok_b else None
        if f < b - 1e-9:
            cond_a = False
        detail_a.append({"axis": a, "fused": f, "best_single": b,
                         "best_single_modality": who, "margin": f - b})
        print(f"      {a:14s} fused {f:6.1%}  best specialist {b:6.1%} ({who})"
              f"  margin {f - b:+.1%}")
    print(f"   (a) {'PASS' if cond_a else 'FAIL'}")

    valid_pts = [r for r in sweep if r["valid"]]
    # Fusion must not LOSE at any valid point either. The first run of this
    # sweep had fused 48 points below the timing specialist at strength 0.20
    # and the verdict logic called it a tie; a configuration that is beaten by
    # a 9-feature subset of itself has not demonstrated fusion.
    lost_at = [r["strength"] for r in sweep if r["fusion_loses"]]
    cond_b = any(r["fusion_wins"] for r in sweep) and not lost_at
    print(f"\n   (b) {'PASS' if cond_b else 'FAIL'}"
          f"   ({len(valid_pts)} of {len(sweep)} swept strengths were valid "
          f"comparison points)")
    if lost_at:
        print(f"       Fusion was BEATEN by a single modality at strength(s) "
              f"{lost_at}, by more than both the noise floor and the seed")
        print("       spread. A model beaten by a subset of its own features has not")
        print("       demonstrated that fusing them helps.")
    if not valid_pts:
        print("       No strength put a specialist inside the marginal band, so the")
        print("       sweep never created a fair comparison. That is a statement about")
        print("       the instrument, not about fusion.")

    supported = cond_a and cond_b
    print(f"\n  ==> PILLAR N1 {'SUPPORTED' if supported else 'NOT SUPPORTED'}.")
    if not supported:
        print("      Report this. Do not sweep fusion weights or re-pick a strength")
        print("      to make it pass; that fits the conclusion rather than testing it.")

    (OUT / "fusion_ablation.json").write_text(json.dumps({
        "per_profile": args.per_profile, "benign_multiple": args.benign_multiple,
        "seed": args.seed, "target_fpr": args.target_fpr,
        "marginal_band": MARGINAL_BAND, "max_benign_fp": MAX_BENIGN_FP,
        "n_axis_test": n_axis_test, "purity": purity,
        "sweep": sweep,
        "seeds": args.seeds, "lost_at_strengths": lost_at,
        "criterion": {"a_fusion_does_not_lose": cond_a,
                      "b_fusion_wins_somewhere": cond_b,
                      "supported": supported, "single_axis_detail": detail_a},
    }, indent=2, default=float))
    print(f"\n  wrote out/fusion_ablation.json    ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
