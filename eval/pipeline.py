"""
One evaluation configuration, end to end.

Everything a reported table needs comes through here, so ablations differ from
the full system only in their arguments — never in their code path. That is
what makes the comparison honest: a Stage-1-only row and a timing-only row are
scored, thresholded and gated by exactly the same machinery.

Threshold calibration is a **reported result, not a hidden knob** (plan §6).
Every threshold is the quantile of *validation benign* windows at the target
per-window FPR. The test set is touched once, at reporting time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from iudbms.contract import FEATURE_NAMES, N_FEATURES
from iudbms.policy import PolicyEngine
from iudbms.stage1 import Stage1, class_group
from iudbms.stage2 import SEQ_LEN, LSTMDetector, TrainConfig, build_sequences

from .corpus import SessionData

__all__ = ["Result", "run_configuration", "TARGET_FPR", "GATE_K", "GATE_N"]

TARGET_FPR = 0.001      # per-window; session FP compounds over ~60 windows
GATE_K, GATE_N = 4, 6   # k-of-n persistence gate


@dataclass
class Result:
    name: str
    threshold: float
    window: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)
    per_profile: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    #: Populated only when keep_scores=True, so the default path does not
    #: retain per-window arrays for every session in the corpus.
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    #: (Stage1, LSTMDetector|None) — saved so the live detector runs the same
    #: weights the reported numbers were produced by.
    models: tuple = ()


def _stack(sessions: list[SessionData], idx: np.ndarray | None, key: str) -> np.ndarray:
    arrs = [getattr(s, key) for s in sessions]
    if key == "X" and idx is not None:
        arrs = [a[:, idx] for a in arrs]
    return np.vstack(arrs) if key == "X" else np.concatenate(arrs)


def run_configuration(
    train: list[SessionData],
    calib: list[SessionData],
    report: list[SessionData],
    *,
    name: str,
    feature_idx: np.ndarray | None = None,
    use_stage2: bool = True,
    seed: int = 7,
    epochs: int = 14,
    seq_len: int = SEQ_LEN,
    hidden: int = 32,
    lr: float = 2e-3,
    verbose: bool = True,
    keep_scores: bool = False,
    score_on: str = "report",
    calibrate_on: str = "calib",
    target_fpr: float = TARGET_FPR,
) -> Result:
    """Train and evaluate one configuration.

    The three splits are **roles, not fixed datasets**: ``train`` fits the
    models, ``calib`` sets the decision threshold, and ``report`` is what gets
    measured. The final evaluation passes (fit, val, test); a hyperparameter
    sweep passes its own inner splits and never constructs the test set at all.
    Naming them fit/val/test invited exactly one bug — a sweep that pointed
    ``score_on`` at its calibration split — so they are named by role.

    ``feature_idx`` selects a column subset for the modality ablations; ``None``
    means all 28. ``use_stage2=False`` is the Stage-1-only ablation that
    isolates novelty pillar N3.
    """
    all_sessions = train + calib + report
    sel = feature_idx if feature_idx is not None else np.arange(N_FEATURES)

    # ---------------- Stage 1: fit on benign fit-set windows only ----------
    Xb = np.vstack([s.X[:, sel] for s in train if s.label == 0])
    groups = [class_group(s.descriptor) for s in train if s.label == 0
              for _ in range(len(s.X))]
    names = tuple(FEATURE_NAMES[i] for i in sel)
    s1 = Stage1(seed=seed, feature_names=names).fit(Xb, groups)

    dev_raw = []
    for s in train:
        if s.label == 0:
            _, _, dr = s1.score_session(s.X[:, sel], s.descriptor)
            dev_raw.append(dr)
    s1.calibrate_dev(np.concatenate(dev_raw))

    S1: dict[str, np.ndarray] = {}
    EXPL: dict[str, list] = {}
    for s in all_sessions:
        sc, ex, _ = s1.score_session(s.X[:, sel], s.descriptor)
        S1[s.sid], EXPL[s.sid] = sc, ex

    # ---------------- Stage 2 ----------------
    history: list = []
    model = None
    if use_stage2:
        def seqset(group):
            Xs = [build_sequences(np.hstack([s.X[:, sel], S1[s.sid].reshape(-1, 1)]),
                                  seq_len=seq_len) for s in group]
            return np.concatenate(Xs), np.concatenate([s.y for s in group])

        Xtr, ytr = seqset(train)
        Xva, yva = seqset(calib)
        model = LSTMDetector(d_in=Xtr.shape[2], hidden=hidden, seed=seed)
        history = model.fit(Xtr, ytr.astype(np.float32),
                            cfg=TrainConfig(epochs=epochs, seed=seed, hidden=hidden, lr=lr),
                            val=(Xva, yva), verbose=verbose)
        scores = {
            s.sid: model.predict_proba(
                build_sequences(np.hstack([s.X[:, sel], S1[s.sid].reshape(-1, 1)]),
                                seq_len=seq_len))
            for s in all_sessions
        }
    else:
        scores = S1

    # ---------------- threshold from held-out benign windows only ----------
    # calibrate_on and score_on MUST name disjoint splits. Calibrating and
    # scoring on the same sessions makes the false-positive rate hit the target
    # by construction, which would make every sweep look equally good.
    if calibrate_on == score_on:
        raise ValueError(
            f"calibrate_on and score_on are both {score_on!r}; the threshold would "
            f"be fit to the same sessions it is measured on"
        )
    cal = {"report": report, "calib": calib, "train": train}[calibrate_on]
    cal_benign = np.concatenate([scores[s.sid][s.y == 0] for s in cal])
    threshold = float(np.quantile(cal_benign, 1 - target_fpr))

    # ---------------- window level ----------------------------------------
    # score_on="val" is what every hyperparameter sweep uses. The test split is
    # touched exactly once, at reporting time, by score_on="test" (the default).
    target = {"report": report, "calib": calib, "train": train}[score_on]
    y_te = _stack(target, None, "y")
    p_te = np.concatenate([scores[s.sid] for s in target])
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_te, (p_te >= threshold).astype(int), average="binary", zero_division=0)
    window = {
        "auc": float(roc_auc_score(y_te, p_te)),
        "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "n_windows": int(len(y_te)), "n_malicious": int(y_te.sum()),
    }

    # ---------------- session-level, k-of-n gated --------------------------
    eng = PolicyEngine(threshold=threshold, k=GATE_K, n=GATE_N)
    first_alert: dict[str, float | None] = {}
    states: dict[str, str] = {}
    for s in target:
        ctx = eng.new_device(s.sid)
        first = None
        for i, t in enumerate(s.t):
            a = eng.step(ctx, float(t), float(S1[s.sid][i]), float(scores[s.sid][i]),
                         EXPL[s.sid][i], s.descriptor)
            if a and first is None:
                first = float(t) + 1.0     # alert lands at the end of the window
        first_alert[s.sid] = first
        states[s.sid] = ctx.state

    atk = [s for s in target if s.label == 1]
    ben = [s for s in target if s.label == 0]
    lat = [first_alert[s.sid] - s.onset for s in atk if first_alert[s.sid] is not None]
    ben_hours = sum(s.duration for s in ben) / 3600

    session = {
        "attack_detect": sum(first_alert[s.sid] is not None for s in atk) / max(len(atk), 1),
        "benign_fp": sum(first_alert[s.sid] is not None for s in ben) / max(len(ben), 1),
        "revoked": sum(states[s.sid] == "revoked" for s in atk) / max(len(atk), 1),
        "median_latency": float(np.median(lat)) if lat else None,
        "p90_latency": float(np.percentile(lat, 90)) if lat else None,
        "fp_per_benign_hour": (sum(first_alert[s.sid] is not None for s in ben)
                               / max(ben_hours, 1e-9)),
        "n_attack": len(atk), "n_benign": len(ben),
    }

    per_profile: dict[str, dict] = {}
    grouped: dict[str, list[SessionData]] = defaultdict(list)
    for s in target:
        grouped[s.profile].append(s)
    for profile, group in grouped.items():
        hits = [first_alert[s.sid] is not None for s in group]
        lats = [first_alert[s.sid] - s.onset for s in group
                if first_alert[s.sid] is not None and s.onset is not None]
        per_profile[profile] = {
            "n": len(group),
            "label": group[0].label,
            "detect_rate": sum(hits) / len(group),
            "median_latency": float(np.median(lats)) if lats else None,
        }

    result = Result(name=name, threshold=threshold, window=window,
                    session=session, per_profile=per_profile, history=history)
    if keep_scores:
        result.stage1, result.stage2, result.evidence = S1, scores, EXPL
    result.models = (s1, model if use_stage2 else None)
    return result


def identity_whitelist_baseline(train: list[SessionData], report: list[SessionData]) -> dict:
    """USBGuard-style VID:PID allowlist learned from benign training devices.

    Expected to score 0 % by construction: every attack profile deliberately
    presents a descriptor drawn from the same pool as benign devices, which is
    the whole reason behavioural monitoring is needed.
    """
    known = {(s.descriptor.vid, s.descriptor.pid) for s in train if s.label == 0}
    atk = [s for s in report if s.label == 1]
    ben = [s for s in report if s.label == 0]
    return {
        "attack_detect": sum((s.descriptor.vid, s.descriptor.pid) not in known
                             for s in atk) / max(len(atk), 1),
        "benign_fp": sum((s.descriptor.vid, s.descriptor.pid) not in known
                         for s in ben) / max(len(ben), 1),
        "n_attack": len(atk), "n_benign": len(ben),
    }
