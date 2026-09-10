#!/usr/bin/env python3
"""
Model-family comparison — does each chosen algorithm earn its place?

    python3 -m eval.model_zoo [--quick]

Three questions the project had never answered:

1. **Stage 1.** Is an Isolation Forest the right unsupervised detector, or would
   Local Outlier Factor / One-Class SVM / Elliptic Envelope do better?
2. **Stage 2.** Does an LSTM beat a GRU, a 1-D CNN, or gradient boosting on the
   flattened sequence? Recurrence is an architectural claim and needs testing.
3. **The architecture itself.** Would a single supervised classifier over one
   window match the whole two-stage pipeline? If so, the pipeline is decoration.

**A fairness note that governs how these are read.** Stage 1 trains on benign
windows only, because a deployment has benign traffic and not a labelled attack
corpus. Every *supervised* entry below sees attack labels, so it is an upper
bound on extractable signal rather than a deployable alternative. Comparing an
unsupervised detector against a supervised one and declaring the supervised one
better is a category error; the useful reading is the size of the gap, which
says how much the labelling constraint costs.

Every model is scored, thresholded and gated by the identical machinery, and
every threshold is calibrated on the validation split alone.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from iudbms.contract import N_FEATURES, feature_index
from iudbms.policy import PolicyEngine
from iudbms.stage1 import class_group
from iudbms.stage2 import LSTMDetector, TrainConfig, build_sequences

from .corpus import load_corpus, stratified_split
from .pipeline import GATE_K, GATE_N, TARGET_FPR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DEPLOYED = np.array([i for i in range(N_FEATURES)
                     if i != feature_index("log_time_since_enum")])


# --------------------------------------------------------------------------
# shared scoring: identical for every model, so comparisons are honest
# --------------------------------------------------------------------------

def evaluate(scores: dict, val, test, name: str, supervised: bool) -> dict:
    cal = np.concatenate([scores[s.sid][s.y == 0] for s in val])
    thr = float(np.quantile(cal, 1 - TARGET_FPR))

    y = np.concatenate([s.y for s in test])
    p = np.concatenate([scores[s.sid] for s in test])
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")

    eng = PolicyEngine(threshold=thr, k=GATE_K, n=GATE_N)
    det = fp = 0
    n_atk = n_ben = 0
    per: dict[str, list[int]] = {}
    for s in test:
        ctx = eng.new_device(s.sid)
        hit = False
        for i, t in enumerate(s.t):
            a = eng.step(ctx, float(t), 0.0, float(scores[s.sid][i]), [], s.descriptor)
            if a is not None:
                hit = True
        if s.label == 1:
            n_atk += 1
            det += hit
            row = per.setdefault(s.profile, [0, 0])
            row[0] += hit
            row[1] += 1
        else:
            n_ben += 1
            fp += hit
    return {
        "name": name, "supervised": supervised, "auc": auc, "threshold": thr,
        "detect": det / max(n_atk, 1), "benign_fp": fp / max(n_ben, 1),
        "per_profile": {k: v[0] / v[1] for k, v in per.items()},
    }


def _sessionwise(vec: np.ndarray, sessions) -> dict:
    """Split a flat per-window vector back into per-session arrays."""
    out, i = {}, 0
    for s in sessions:
        n = len(s.t)
        out[s.sid] = vec[i:i + n]
        i += n
    return out


# --------------------------------------------------------------------------
# Stage 1 candidates — unsupervised, fit on benign windows only
# --------------------------------------------------------------------------

def stage1_candidates(fit, val, test, seed, quick):
    allsess = fit + val + test
    Xb = np.vstack([s.X[:, DEPLOYED] for s in fit if s.label == 0])
    mu, sd = Xb.mean(0), np.maximum(Xb.std(0), 1e-2)
    Zb = np.clip((Xb - mu) / sd, -10, 10)
    Xall = np.vstack([s.X[:, DEPLOYED] for s in allsess])
    Zall = np.clip((Xall - mu) / sd, -10, 10)

    # One-Class SVM is O(n^2); subsample or it never finishes.
    cap = 3000 if quick else 6000
    rng = np.random.default_rng(seed)
    sub = Zb[rng.choice(len(Zb), min(cap, len(Zb)), replace=False)]

    models = [
        ("IsolationForest (incumbent)",
         lambda: IsolationForest(n_estimators=200, max_samples=min(512, len(Zb)),
                                 contamination=0.02, random_state=seed).fit(Zb),
         lambda m, Z: -m.score_samples(Z)),
        ("LocalOutlierFactor",
         lambda: LocalOutlierFactor(n_neighbors=20, novelty=True).fit(sub),
         lambda m, Z: -m.score_samples(Z)),
        ("OneClassSVM (rbf)",
         lambda: OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(sub),
         lambda m, Z: -m.score_samples(Z)),
        ("EllipticEnvelope",
         lambda: EllipticEnvelope(contamination=0.02, support_fraction=0.9,
                                  random_state=seed).fit(sub),
         lambda m, Z: -m.score_samples(Z)),
    ]

    results = []
    for name, build, score in models:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build()
                raw = score(m, Zall)
        except Exception as exc:                      # noqa: BLE001
            results.append({"name": name, "supervised": False, "error": str(exc)[:90]})
            print(f"  {name:30s} FAILED: {str(exc)[:70]}", flush=True)
            continue
        # calibrate to the benign 99th percentile, as Stage 1 does
        p99 = float(np.percentile(raw, 99)) or 1.0
        s = np.clip(raw / p99, 0, 3) / 3
        r = evaluate(_sessionwise(s, allsess), val, test, name, supervised=False)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {name:30s} AUC={r['auc']:.4f}  detect={r['detect']:6.1%}  "
              f"FP={r['benign_fp']:5.1%}  {r['seconds']:5.1f}s", flush=True)
    return results


# --------------------------------------------------------------------------
# Stage 2 candidates — supervised, sequence input
# --------------------------------------------------------------------------

def stage2_candidates(fit, val, test, s1_scores, seed, quick, seq_len=10):
    import torch
    from torch import nn

    allsess = fit + val + test

    def seqs(group):
        X = [build_sequences(np.hstack([s.X[:, DEPLOYED], s1_scores[s.sid].reshape(-1, 1)]),
                             seq_len=seq_len) for s in group]
        return np.concatenate(X), np.concatenate([s.y for s in group])

    Xtr, ytr = seqs(fit)
    Xall, _ = seqs(allsess)
    d_in = Xtr.shape[2]
    epochs = 4 if quick else 14
    results = []

    class Recurrent(nn.Module):
        """GRU / LSTM behind one interface, so only the cell type differs."""

        def __init__(self, cell, d_in, hidden=32):
            super().__init__()
            self.rnn = cell(input_size=d_in, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
            self.register_buffer("mu", torch.zeros(d_in))
            self.register_buffer("sd", torch.ones(d_in))

        def forward(self, x):
            x = (x - self.mu) / self.sd
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    class Conv1D(nn.Module):
        """Dilated causal-ish 1-D CNN: same receptive field, no recurrence."""

        def __init__(self, d_in, ch=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(d_in, ch, 3, padding=1), nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1))
            self.head = nn.Linear(ch, 1)
            self.register_buffer("mu", torch.zeros(d_in))
            self.register_buffer("sd", torch.ones(d_in))

        def forward(self, x):
            x = (x - self.mu) / self.sd
            return self.head(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)

    def train_torch(model, tag):
        torch.manual_seed(seed)
        Xt = torch.as_tensor(Xtr, dtype=torch.float32)
        yt = torch.as_tensor(ytr, dtype=torch.float32)
        flat = Xt.reshape(-1, d_in)
        with torch.no_grad():
            model.mu.copy_(flat.mean(0)); model.sd.copy_(flat.std(0).clamp_min(1e-6))
        pos = float(yt.sum())
        lf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(len(yt) - pos) / max(pos, 1.0)]))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            model.train()
            for idx in torch.randperm(len(Xt), generator=g).split(128):
                opt.zero_grad(set_to_none=True)
                loss = lf(model(Xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            Xa = torch.as_tensor(Xall, dtype=torch.float32)
            out = np.concatenate([torch.sigmoid(model(Xa[i:i + 4096])).numpy()
                                  for i in range(0, len(Xa), 4096)])
        return out

    for tag, factory in (("LSTM (incumbent)", lambda: Recurrent(nn.LSTM, d_in)),
                         ("GRU", lambda: Recurrent(nn.GRU, d_in)),
                         ("1-D CNN (no recurrence)", lambda: Conv1D(d_in))):
        t0 = time.time()
        p = train_torch(factory(), tag)
        r = evaluate(_sessionwise(p, allsess), val, test, tag, supervised=True)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {tag:30s} AUC={r['auc']:.4f}  detect={r['detect']:6.1%}  "
              f"FP={r['benign_fp']:5.1%}  {r['seconds']:5.1f}s", flush=True)

    # flat models over the same sequence, to test whether recurrence matters
    Ftr = Xtr.reshape(len(Xtr), -1)
    Fall = Xall.reshape(len(Xall), -1)
    for tag, mk in (("HistGradientBoosting (flat seq)",
                     lambda: HistGradientBoostingClassifier(
                         max_iter=200, random_state=seed,
                         class_weight="balanced")),
                    ("LogisticRegression (flat seq)",
                     lambda: LogisticRegression(max_iter=400, class_weight="balanced"))):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mk().fit(Ftr, ytr)
            p = m.predict_proba(Fall)[:, 1]
        r = evaluate(_sessionwise(p, allsess), val, test, tag, supervised=True)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {tag:30s} AUC={r['auc']:.4f}  detect={r['detect']:6.1%}  "
              f"FP={r['benign_fp']:5.1%}  {r['seconds']:5.1f}s", flush=True)
    return results


# --------------------------------------------------------------------------
# the architecture control: one supervised model, one window, no pipeline
# --------------------------------------------------------------------------

def architecture_control(fit, val, test, seed):
    allsess = fit + val + test
    Xtr = np.vstack([s.X[:, DEPLOYED] for s in fit])
    ytr = np.concatenate([s.y for s in fit])
    Xall = np.vstack([s.X[:, DEPLOYED] for s in allsess])
    results = []
    for tag, mk in (("HistGradientBoosting (1 window)",
                     lambda: HistGradientBoostingClassifier(
                         max_iter=200, random_state=seed, class_weight="balanced")),
                    ("RandomForest (1 window)",
                     lambda: RandomForestClassifier(
                         n_estimators=200, random_state=seed,
                         class_weight="balanced", n_jobs=-1))):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mk().fit(Xtr, ytr)
            p = m.predict_proba(Xall)[:, 1]
        r = evaluate(_sessionwise(p, allsess), val, test, tag, supervised=True)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {tag:30s} AUC={r['auc']:.4f}  detect={r['detect']:6.1%}  "
              f"FP={r['benign_fp']:5.1%}  {r['seconds']:5.1f}s", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=OUT / "corpus_cache.npz")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--on", choices=("selection", "test"), default="selection",
                    help="'selection' (default) scores on inner splits carved from "
                         "training data, so the comparison can drive a decision. "
                         "'test' is exploratory only -- choosing a model from a "
                         "test-scored table is selecting on test.")
    args = ap.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"no corpus at {args.cache}; run `make eval` first")
    corpus = load_corpus(args.cache)
    outer_fit, outer_val, outer_test = stratified_split(corpus.sessions, seed=args.seed)
    if args.on == "selection":
        from .tune import inner_splits
        fit, val, test = inner_splits(outer_fit, outer_val, seed=args.seed)
        print(f"corpus {len(corpus.sessions)} sessions | scoring on the SELECTION split "
              f"({len(outer_test)} test sessions held back and never touched)")
        print(f"  inner train={len(fit)} calibrate={len(val)} select={len(test)}\n",
              flush=True)
    else:
        fit, val, test = outer_fit, outer_val, outer_test
        print(f"corpus {len(corpus.sessions)} sessions | scoring on TEST "
              f"(exploratory only -- do not pick a model from this table)")
        print(f"  fit={len(fit)} val={len(val)} test={len(test)}\n", flush=True)

    t0 = time.time()
    print("STAGE 1 candidates — unsupervised, benign-only training", flush=True)
    s1_rows = stage1_candidates(fit, val, test, args.seed, args.quick)

    # incumbent Stage 1 scores feed every Stage 2 candidate, so the comparison
    # isolates the sequence model rather than confounding it with Stage 1
    from iudbms.stage1 import DeviceBaseline, Stage1
    from iudbms.contract import FEATURE_NAMES
    names = tuple(FEATURE_NAMES[i] for i in DEPLOYED)
    Xb = np.vstack([s.X[:, DEPLOYED] for s in fit if s.label == 0])
    groups = [class_group(s.descriptor) for s in fit if s.label == 0
              for _ in range(len(s.X))]
    s1 = Stage1(seed=args.seed, feature_names=names).fit(Xb, groups)
    s1.calibrate_dev(np.concatenate(
        [s1.score_session(s.X[:, DEPLOYED], s.descriptor)[2] for s in fit if s.label == 0]))
    s1_scores = {s.sid: s1.score_session(s.X[:, DEPLOYED], s.descriptor)[0]
                 for s in fit + val + test}

    print("\nSTAGE 2 candidates — supervised, sequence input", flush=True)
    s2_rows = stage2_candidates(fit, val, test, s1_scores, args.seed, args.quick)

    print("\nARCHITECTURE CONTROL — one supervised model, one window, no pipeline",
          flush=True)
    ctl_rows = architecture_control(fit, val, test, args.seed)

    payload = {"scored_on": args.on, "stage1": s1_rows, "stage2": s2_rows,
               "control": ctl_rows, "seed": args.seed,
               "seconds": round(time.time() - t0, 1)}
    (OUT / "model_zoo.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote out/model_zoo.json   ({payload['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
