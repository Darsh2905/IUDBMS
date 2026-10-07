"""
Shared scaffolding for the improvement-analysis experiments (eval/analysis_*.py).

Every experiment starts from the same place -- the cached 600/400 corpus, the
seed-7 session split, and (where a trained model is needed) either the deployed
checkpoint or a configuration trained through ``run_configuration`` -- so that
differences between experiments are differences in the thing being tested, never
in the plumbing. Nothing here writes to ``out/models`` or touches a committed
artefact.

Discipline carried over from the rest of the harness:

* splits are by session, stratified by profile (plan §5);
* a *decision* (which variant wins) is made on the inner validation split, never
  on test; test is reported once, after the decision;
* a margin smaller than the noise floor (1/n sessions, and seed spread where
  seeds are run) is reported as a tie, not a win.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "src"))

from iudbms.policy import PolicyEngine  # noqa: E402
from iudbms.stage1 import Stage1  # noqa: E402
from iudbms.stage2 import LSTMDetector, build_sequences  # noqa: E402

from .corpus import SessionData, load_corpus, stratified_split  # noqa: E402

CACHE = OUT / "corpus_cache.npz"
MODELS = OUT / "models"
GATE_K, GATE_N = 4, 6


def load_splits(seed: int = 7):
    """(corpus, fit, val, test) from the cached corpus -- the same split run_eval uses."""
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing: run `make eval` (or build the cache) first")
    corpus = load_corpus(CACHE)
    fit, val, test = stratified_split(corpus.sessions, seed=seed)
    return corpus, fit, val, test


def deployed(models: Path | None = None):
    """A checkpoint set: (config, Stage1, LSTMDetector). Default is the deployment's."""
    d = Path(models) if models is not None else MODELS
    cfg = json.loads((d / "deploy.json").read_text())
    return cfg, Stage1.load(d / "stage1.pkl"), LSTMDetector.load(d / "stage2.pt")


def save_set(dest: Path, result, idx, *, target_fpr: float, seq_len: int = 10,
             k: int = GATE_K, n: int = GATE_N) -> Path:
    """Write a trained configuration as a loadable, canary-stamped model set.

    Experiments keep their checkpoints under out/analysis_models/, never in
    out/models/ -- that directory is the deployment's.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    s1m, s2m = result.models
    s1m.save(dest / "stage1.pkl")
    s2m.save(dest / "stage2.pt")
    (dest / "deploy.json").write_text(json.dumps({
        "feature_idx": [int(i) for i in idx], "threshold": result.threshold,
        "target_fpr": target_fpr, "seq_len": seq_len, "k": k, "n": n,
        "canary": {"stage1": s1m.canary(), "stage2": s2m.canary(seq_len)}}, indent=2))
    return dest


def score_with(s1: Stage1, lstm: LSTMDetector | None, sessions: list[SessionData],
               idx: np.ndarray, seq_len: int = 10):
    """Stage-1 and Stage-2 per-window scores for each session, keyed by sid."""
    # Scores are returned in dicts keyed by session id. Twice in one analysis a
    # duplicate id (a fresh evaluation set reusing the training corpus's S0001/A0007
    # names; a loop that repeated "X0000" per profile) silently overwrote one session's
    # scores with another's, so make it a loud error instead.
    sids = [s.sid for s in sessions]
    if len(set(sids)) != len(sids):
        dup = sorted({x for x in sids if sids.count(x) > 1})[:5]
        raise ValueError(f"duplicate session ids in one scoring call, e.g. {dup}")
    S1, S2, EXPL = {}, {}, {}
    for s in sessions:
        sc, ex, _ = s1.score_session(s.X[:, idx], s.descriptor)
        S1[s.sid], EXPL[s.sid] = sc, ex
        if lstm is not None:
            seq = build_sequences(np.hstack([s.X[:, idx], sc.reshape(-1, 1)]), seq_len=seq_len)
            S2[s.sid] = lstm.predict_proba(seq)
    return S1, (S2 if lstm is not None else S1), EXPL


def run_gate(scores: dict, sessions: list[SessionData], threshold: float,
             k: int = GATE_K, n: int = GATE_N) -> dict:
    """First-alert time per session under the k-of-n persistence gate.

    Mirrors ``pipeline.run_configuration``'s session loop (alert lands at the end
    of the window that completed the k-of-n), but needs no Stage-1 evidence.
    """
    eng = PolicyEngine(threshold=threshold, k=k, n=n)
    first = {}
    for s in sessions:
        ctx = eng.new_device(s.sid)
        t_alert = None
        for i, t in enumerate(s.t):
            a = eng.step(ctx, float(t), 0.0, float(scores[s.sid][i]), [], s.descriptor)
            if a and t_alert is None:
                t_alert = float(t) + 1.0
        first[s.sid] = t_alert
    return first


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (honest at small n and p near 0/1)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def session_rates(first: dict, sessions: list[SessionData]) -> dict:
    atk = [s for s in sessions if s.label == 1]
    ben = [s for s in sessions if s.label == 0]
    hit = sum(first[s.sid] is not None for s in atk)
    fp = sum(first[s.sid] is not None for s in ben)
    lat = [first[s.sid] - s.onset for s in atk if first[s.sid] is not None]
    return {
        "detect": hit / max(len(atk), 1), "detect_k": hit, "n_attack": len(atk),
        "detect_ci": wilson(hit, len(atk)),
        "fp": fp / max(len(ben), 1), "fp_k": fp, "n_benign": len(ben),
        "fp_ci": wilson(fp, len(ben)),
        "median_latency": float(np.median(lat)) if lat else None,
        "p90_latency": float(np.percentile(lat, 90)) if lat else None,
    }


def per_profile(first: dict, sessions: list[SessionData]) -> dict:
    out: dict[str, dict] = {}
    for prof in sorted({s.profile for s in sessions}):
        g = [s for s in sessions if s.profile == prof]
        k = sum(first[s.sid] is not None for s in g)
        out[prof] = {"n": len(g), "k": k, "rate": k / len(g), "ci": wilson(k, len(g)),
                     "label": g[0].label}
    return out


def featurize(sim_session, sid: str, profile: str, extractor_cls=None) -> SessionData:
    """One simulated session -> SessionData, exactly as eval.corpus.build_corpus does it.

    ``extractor_cls`` lets an experiment append columns (see eval.analysis_ngram); the
    default is the frozen 30-feature extractor.
    """
    from iudbms.capture.simulator import SimulatedSource
    from iudbms.features.extractor import WindowFeatureExtractor, window_events

    ex = (extractor_cls or WindowFeatureExtractor)()
    X, t, y = [], [], []
    for t0, bucket in window_events(list(SimulatedSource(sim_session).events()),
                                    duration=sim_session.duration):
        X.append(ex.extract(bucket, t0, sim_session.descriptor))
        t.append(t0)
        y.append(int(sim_session.label == 1 and sim_session.onset is not None
                     and (t0 + ex.WINDOW) > sim_session.onset))
    return SessionData(sid=sid, profile=profile, label=sim_session.label,
                       descriptor=sim_session.descriptor, X=np.vstack(X), t=np.asarray(t),
                       y=np.asarray(y), onset=sim_session.onset, duration=sim_session.duration)


def featurize_multi(sim_session, sid: str, profile: str, transforms: dict,
                    extractor_cls=None) -> dict:
    """Featurise ONE simulated session under several event-level transforms.

    ``transforms`` maps a name to ``f(events) -> events`` (use ``None`` for identity).
    The events are generated and parsed once and discarded afterwards, so memory stays
    flat however many perturbations are applied.
    """
    from iudbms.capture.simulator import SimulatedSource
    from iudbms.features.extractor import WindowFeatureExtractor, window_events

    base = list(SimulatedSource(sim_session).events())
    out = {}
    for name, fn in transforms.items():
        ev = base if fn is None else fn(base)
        ex = (extractor_cls or WindowFeatureExtractor)()
        X, t, y = [], [], []
        for t0, bucket in window_events(ev, duration=sim_session.duration):
            X.append(ex.extract(bucket, t0, sim_session.descriptor))
            t.append(t0)
            y.append(int(sim_session.label == 1 and sim_session.onset is not None
                         and (t0 + ex.WINDOW) > sim_session.onset))
        out[name] = SessionData(sid=sid, profile=profile, label=sim_session.label,
                                descriptor=sim_session.descriptor, X=np.vstack(X),
                                t=np.asarray(t), y=np.asarray(y), onset=sim_session.onset,
                                duration=sim_session.duration)
    return out
