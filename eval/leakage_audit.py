"""
Leakage audit (plan §5, mandatory before any reported result is trusted).

Confirms benign and malicious sessions are drawn from ONE distribution on every
axis the models could shortcut on. A model that separates classes on session
length, device model or window count is not detecting anything.

This exists because the audit has already caught a real defect twice. The first
version of the simulator drew attack sessions from a longer duration range
outright. That was fixed, aggregate means matched, and it looked clean — but
`staged_delay` still redrew uniform(65, 85) against everyone else's
uniform(40, 85), so the leak survived at the per-profile level where nobody was
looking. **Aggregate agreement is not evidence; check per profile.**

    python3 -m eval.leakage_audit [--n-benign 210] [--n-attack 140]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES, BENIGN_PROFILES, Simulator

#: Two-sample Kolmogorov-Smirnov critical value at alpha=0.01 is
#: 1.63 * sqrt((n+m)/(n*m)). We flag above this.
KS_ALPHA_COEFF = 1.63

#: Smallest effect the audit is expected to catch, as a KS distance. A profile
#: whose critical value exceeds this cannot detect a leak of that size, so the
#: audit reports "underpowered" rather than "ok" -- silence from a test that
#: could not have failed is not evidence.
TARGET_DETECTABLE_D = 0.35


def ks_statistic(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and its alpha=0.01 critical value."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    crit = KS_ALPHA_COEFF * np.sqrt((len(a) + len(b)) / (len(a) * len(b)))
    return d, float(crit)


def audit(n_benign: int = 210, n_attack: int = 140, seed: int = 7) -> dict:
    sim = Simulator(seed=seed)
    durs: dict[str, list[float]] = defaultdict(list)
    n_windows: dict[str, list[int]] = defaultdict(list)
    vidpid: dict[str, set] = defaultdict(set)
    labels: dict[str, int] = {}

    for sid, prof in sim.plan(n_benign=n_benign, n_attack=n_attack):
        s = sim.session(sid, prof)
        durs[prof].append(s.duration)
        n_windows[prof].append(int(np.ceil(s.duration)))
        vidpid[prof].add((s.descriptor.vid, s.descriptor.pid))
        labels[prof] = s.label

    findings: list[str] = []

    print("=" * 74)
    print("LEAKAGE AUDIT — session duration by profile (plan §5)")
    print("=" * 74)
    print(f"{'profile':18s} {'label':>5s} {'n':>4s} {'mean':>7s} {'sd':>6s} "
          f"{'min':>7s} {'max':>7s}")
    for p in list(BENIGN_PROFILES) + list(ATTACK_PROFILES):
        d = np.array(durs[p])
        print(f"{p:18s} {labels[p]:5d} {len(d):4d} {d.mean():7.1f} {d.std():6.1f} "
              f"{d.min():7.1f} {d.max():7.1f}")

    ben = np.concatenate([durs[p] for p in BENIGN_PROFILES])
    mal = np.concatenate([durs[p] for p in ATTACK_PROFILES])

    print(f"\n{'aggregate benign':18s} {'':5s} {len(ben):4d} {ben.mean():7.1f} "
          f"{ben.std():6.1f} {ben.min():7.1f} {ben.max():7.1f}")
    print(f"{'aggregate malicious':18s} {'':5s} {len(mal):4d} {mal.mean():7.1f} "
          f"{mal.std():6.1f} {mal.min():7.1f} {mal.max():7.1f}")

    d_agg, crit_agg = ks_statistic(ben, mal)
    print(f"\naggregate KS D={d_agg:.4f}  critical(a=0.01)={crit_agg:.4f}  "
          f"{'FLAG' if d_agg > crit_agg else 'ok'}")
    if d_agg > crit_agg:
        findings.append(f"aggregate duration distributions differ (KS D={d_agg:.4f})")

    # The check that actually matters: per profile against the benign pool.
    # Aggregate agreement can hide a single leaking profile, which is exactly
    # how defect 4 survived the first round of fixing.
    print("\nper-profile duration vs benign pool (this is the check that matters):")
    print(f"{'profile':18s} {'n':>4s} {'KS D':>8s} {'crit':>8s}  verdict")
    worst = 0.0
    underpowered = []
    for p in ATTACK_PROFILES:
        d, crit = ks_statistic(ben, np.array(durs[p]))
        worst = max(worst, d - crit)
        flag = d > crit
        if flag:
            findings.append(f"{p}: duration differs from benign (KS D={d:.4f} > {crit:.4f})")
        # A test that cannot fail proves nothing. If the critical value for this
        # profile sits above the smallest effect we care about, say so.
        weak = (not flag) and crit > TARGET_DETECTABLE_D
        if weak:
            underpowered.append((p, len(durs[p]), crit))
        verdict = "FLAG" if flag else ("UNDERPOWERED" if weak else "ok")
        print(f"{p:18s} {len(durs[p]):4d} {d:8.4f} {crit:8.4f}  {verdict}")

    if underpowered:
        need = int(np.ceil((KS_ALPHA_COEFF / TARGET_DETECTABLE_D) ** 2)) + 1
        print(f"\n  {len(underpowered)} profile(s) cannot detect a KS shift of "
              f"{TARGET_DETECTABLE_D}; each needs about {need} sessions.")
        print("  Passing here means 'no leak found', NOT 'no leak'. Re-run with a "
              "larger corpus before trusting it.")
        findings.append(
            f"underpowered: {', '.join(p for p, _, _ in underpowered)} "
            f"(need ~{need} sessions each to detect D={TARGET_DETECTABLE_D})")

    # Window count is a direct proxy for duration and is what the model sees.
    wb = np.concatenate([n_windows[p] for p in BENIGN_PROFILES])
    wm = np.concatenate([n_windows[p] for p in ATTACK_PROFILES])
    d_w, crit_w = ks_statistic(wb.astype(float), wm.astype(float))
    print(f"\nwindow-count KS D={d_w:.4f}  critical={crit_w:.4f}  "
          f"{'FLAG' if d_w > crit_w else 'ok'}")
    if d_w > crit_w:
        findings.append(f"window counts differ (KS D={d_w:.4f})")

    # Device model must not be a giveaway: every attack profile's VID:PID pool
    # must be a subset of some benign profile's, or a whitelist would win.
    benign_ids = set().union(*(vidpid[p] for p in BENIGN_PROFILES))
    print("\nVID:PID overlap (an attack model absent from benign traffic is a giveaway):")
    for p in ATTACK_PROFILES:
        novel = vidpid[p] - benign_ids
        if novel:
            findings.append(f"{p}: uses VID:PID absent from benign traffic: {novel}")
        print(f"{p:18s} {len(vidpid[p])} model(s)  "
              f"{'FLAG ' + str(novel) if novel else 'all seen in benign traffic'}")

    print("\n" + "=" * 74)
    if findings:
        print(f"AUDIT FAILED — {len(findings)} finding(s):")
        for f in findings:
            print(f"  - {f}")
    else:
        print("AUDIT PASSED — no separable non-behavioural axis found.")
    print("=" * 74)

    return {"findings": findings, "aggregate_ks": d_agg, "worst_profile_margin": worst}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-benign", type=int, default=210)
    ap.add_argument("--n-attack", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    return 1 if audit(args.n_benign, args.n_attack, args.seed)["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
