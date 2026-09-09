"""
Threshold and persistence-gate trade-off curve (plan §6).

"Threshold calibration is a reported result, not a hidden knob." Per-window
false positives compound across a session: at 0.5 % per window over ~60 windows
almost every benign session eventually trips, which is why the k-of-n gate
exists and why a single operating point tells you almost nothing.

**This curve is descriptive, not selective.** It is computed on the test split,
which is legitimate for a curve in the same way an ROC curve is — but the
deployed operating point is still the one calibrated on *validation* benign
windows at the target FPR. Reading a better point off this curve and adopting
it would be selecting on test, and is exactly what the curve must not be used
for. The deployed point is marked in the output.
"""

from __future__ import annotations

import numpy as np

from iudbms.policy import PolicyEngine

__all__ = ["sweep_thresholds", "sweep_gates", "format_curves"]


def _session_outcomes(sessions, scores, evidence, stage1, threshold, k, n):
    eng = PolicyEngine(threshold=threshold, k=k, n=n)
    detected_attacks = benign_alerts = 0
    latencies = []
    n_atk = n_ben = 0
    for s in sessions:
        ctx = eng.new_device(s.sid)
        first = None
        for i, t in enumerate(s.t):
            a = eng.step(ctx, float(t), float(stage1[s.sid][i]),
                         float(scores[s.sid][i]), evidence[s.sid][i], s.descriptor)
            if a is not None and first is None:
                first = float(t) + 1.0
        if s.label == 1:
            n_atk += 1
            if first is not None:
                detected_attacks += 1
                if s.onset is not None:
                    latencies.append(first - s.onset)
        else:
            n_ben += 1
            benign_alerts += first is not None
    return {
        "detect": detected_attacks / max(n_atk, 1),
        "benign_fp": benign_alerts / max(n_ben, 1),
        "median_latency": float(np.median(latencies)) if latencies else None,
    }


def sweep_thresholds(sessions, calib_sessions, result, target_fprs, k=4, n=6):
    """Vary the per-window FPR the threshold is calibrated to."""
    cal_benign = np.concatenate([result.stage2[s.sid][s.y == 0] for s in calib_sessions])
    rows = []
    for fpr in target_fprs:
        thr = float(np.quantile(cal_benign, 1 - fpr))
        out = _session_outcomes(sessions, result.stage2, result.evidence,
                                result.stage1, thr, k, n)
        rows.append({"target_fpr": fpr, "threshold": thr, **out})
    return rows


def sweep_gates(sessions, result, threshold, gates):
    """Vary the k-of-n persistence gate at a fixed threshold."""
    rows = []
    for k, n in gates:
        out = _session_outcomes(sessions, result.stage2, result.evidence,
                                result.stage1, threshold, k, n)
        rows.append({"k": k, "n": n, **out})
    return rows


def format_curves(thr_rows, gate_rows, deployed_threshold) -> str:
    lines = []
    lines.append("  Per-window FPR the threshold is calibrated to (4-of-6 gate):")
    lines.append(f"    {'target FPR':>11s} {'threshold':>10s} {'detection':>10s} "
                 f"{'benign FP':>10s} {'latency':>8s}")
    for r in thr_rows:
        mark = "  <- deployed" if abs(r["threshold"] - deployed_threshold) < 1e-12 else ""
        lat = f"{r['median_latency']:.1f}s" if r["median_latency"] else "-"
        lines.append(f"    {r['target_fpr']:11.4%} {r['threshold']:10.4f} "
                     f"{r['detect']:10.1%} {r['benign_fp']:10.1%} {lat:>8s}{mark}")
    lines.append("")
    lines.append(f"  Persistence gate at the deployed threshold {deployed_threshold:.4f}:")
    lines.append(f"    {'gate':>7s} {'detection':>10s} {'benign FP':>10s} {'latency':>8s}")
    for r in gate_rows:
        gate = f"{r['k']}-of-{r['n']}"
        lat = f"{r['median_latency']:.1f}s" if r["median_latency"] else "-"
        lines.append(f"    {gate:>7s} {r['detect']:10.1%} {r['benign_fp']:10.1%} {lat:>8s}")
    return "\n".join(lines)
