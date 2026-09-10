"""
Export the data behind every figure, so the report can draw native charts.

Matplotlib PNGs are fine for a paper and wrong for a web report: they carry a
baked-in light background, blur on a projector, and cannot respond to the
reader's theme. This dumps the underlying series instead, and the report renders
them as inline SVG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from iudbms.contract import feature_index

__all__ = ["export_figure_data"]


def _downsample(xs, ys, n=140):
    """Keep the curve's shape with far fewer points."""
    if len(xs) <= n:
        return list(map(float, xs)), list(map(float, ys))
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return [float(xs[i]) for i in idx], [float(ys[i]) for i in idx]


def export_figure_data(out: Path, full, s1_only, modality, corpus, test,
                       y_true, p_full, p_s1, thr_rows, gate_rows) -> Path:
    data: dict = {}

    data["headline"] = {
        "sessions": len(corpus.sessions), "windows": corpus.n_windows,
        "lines": corpus.n_lines, "threshold": full.threshold,
        "auc": full.window["auc"], "auc_s1": s1_only.window["auc"],
        "precision": full.window["precision"], "recall": full.window["recall"],
        "detect": full.session["attack_detect"],
        "detect_s1": s1_only.session["attack_detect"],
        "benign_fp": full.session["benign_fp"],
        "n_benign": full.session["n_benign"], "n_attack": full.session["n_attack"],
        "latency": full.session["median_latency"], "p90": full.session["p90_latency"],
    }

    data["by_class"] = [
        {"profile": k, "n": v["n"], "iudbms": v["detect_rate"],
         "stage1": s1_only.per_profile.get(k, {}).get("detect_rate", 0.0),
         "whitelist": 0.0, "latency": v["median_latency"]}
        for k, v in sorted(full.per_profile.items()) if v["label"] == 1
    ]

    sizes = {"timing": 9, "payload": 8, "protocol": 6, "identity": 6}
    data["modality"] = [
        {"name": k, "features": sizes.get(k, 0),
         "auc": r.window["auc"], "detect": r.session["attack_detect"]}
        for k, r in modality.items()
    ]
    data["modality_fused"] = {"auc": full.window["auc"],
                              "detect": full.session["attack_detect"]}

    data["roc"] = {}
    for key, p in (("iudbms", p_full), ("stage1", p_s1)):
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, p)
        x, y = _downsample(fpr, tpr)
        data["roc"][key] = {"fpr": x, "tpr": y}

    data["timelines"] = []
    for profile in ("staged_delay", "mimicry_adaptive"):
        cand = [s for s in test if s.profile == profile and s.sid in full.stage2]
        chosen = None
        for s in cand:
            sc = np.asarray(full.stage2[s.sid])
            if (sc >= full.threshold).sum() >= 4:
                chosen = s
                break
        chosen = chosen or (cand[0] if cand else None)
        if chosen is None:
            continue
        sc2 = np.asarray(full.stage2[chosen.sid])
        above = (sc2 >= full.threshold).astype(int)
        gate = None
        for i in range(len(above)):
            if above[max(0, i - 5):i + 1].sum() >= 4:
                gate = float(chosen.t[i]) + 1.0
                break
        data["timelines"].append({
            "profile": profile, "sid": chosen.sid,
            "t": [float(v) for v in chosen.t],
            "stage1": [round(float(v), 4) for v in np.asarray(full.stage1[chosen.sid])],
            "stage2": [round(float(v), 4) for v in sc2],
            "label": [int(v) for v in chosen.y],
            "onset": float(chosen.onset) if chosen.onset is not None else None,
            "fired": gate,
        })

    i_nov = feature_index("payload_seq_novelty")
    edges = list(range(0, 90, 5))
    series = {}
    for profile, post_only in (("keyboard", False), ("mimicry", True),
                               ("mimicry_adaptive", True)):
        group = [s for s in corpus.sessions if s.profile == profile]
        xs, ys = [], []
        for lo, hi in zip(edges, edges[1:]):
            vals = []
            for s in group:
                m = (s.t >= lo) & (s.t < hi)
                if post_only:
                    m &= s.y == 1
                vals.extend(s.X[m][:, i_nov].tolist())
            if len(vals) >= 25:
                xs.append((lo + hi) / 2)
                ys.append(round(float(np.mean(vals)), 4))
        series[profile] = {"x": xs, "y": ys}
    data["novelty"] = series

    data["threshold_sweep"] = thr_rows
    data["gate_sweep"] = gate_rows

    path = out / "figure_data.json"
    path.write_text(json.dumps(data, default=float, separators=(",", ":")))
    return path
