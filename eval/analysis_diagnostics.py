"""
Experiment 9 -- three diagnostics on the v2 model: calibration noise, feature reliance, cost.

    python3 -m eval.analysis_diagnostics

A. **Calibration noise.** The threshold is the 1-in-10,000 quantile of benign *validation
   windows*. With ~8,000 of them that quantile is the top one or two samples, an order
   statistic with enormous variance. This bootstraps the validation sessions, recomputes
   the threshold each time, and measures what that does to the operating point on the
   fresh set (benign FP, detection) -- noise in the *threshold alone*, same model.

B. **Feature reliance.** Permutation importance: shuffle one feature across all windows of
   a fixed evaluation sample, rescore end to end (Stage 1 sees the shuffled value too) and
   record the drop in window AUC. Says which features carry the detection, and whether
   anything suspicious (a position-in-session proxy, a constant) is doing the work.

C. **Cost.** Wall-clock per device-window through the streaming detector, i.e. what one
   CPU core can sustain at one window per second per device.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from iudbms.contract import FEATURE_NAMES, modality_of

from .analysis_common import OUT, deployed, score_with
from .analysis_gates import first_cross, kth_of_n
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE
from .corpus import load_corpus, stratified_split

MODEL = OUT / "analysis_models" / "realistic_v2"


def main() -> int:
    cfg, s1, lstm = deployed(MODEL)
    idx = np.array(cfg["feature_idx"])
    fit, val, _ = stratified_split(load_corpus(TRAIN_CACHE).sessions, seed=7)
    powered, _ = load_or_build()
    ben = [s for s in powered if s.label == 0]
    atk = [s for s in powered if s.label == 1]
    S2 = score_with(s1, lstm, powered, idx, cfg["seq_len"])[1]
    stat = {s.sid: kth_of_n(S2[s.sid], cfg["k"], cfg["n"]) for s in powered}
    out = {}

    # ---------------- A. calibration noise ----------------
    Sv = score_with(s1, lstm, val, idx, cfg["seq_len"])[1]
    vben = [s for s in val if s.label == 0]
    nwin = sum(int((s.y == 0).sum()) for s in vben)
    rng = np.random.default_rng(1)
    ths = []
    for _ in range(400):
        pick = rng.integers(len(vben), size=len(vben))
        w = np.concatenate([Sv[vben[i].sid][vben[i].y == 0] for i in pick])
        ths.append(float(np.quantile(w, 1 - 1e-4)))
    ths = np.array(ths)
    print(f"A. CALIBRATION NOISE   ({nwin:,} benign validation windows; 1e-4 of them = {nwin * 1e-4:.1f} window)")
    print(f"   threshold as shipped {cfg['threshold']:.4f};  bootstrap median {np.median(ths):.4f}, "
          f"2.5-97.5%: {np.percentile(ths, 2.5):.4f} - {np.percentile(ths, 97.5):.4f}")
    print(f"   {'threshold':>10s} {'benign FP':>11s} {'detection':>10s}   (fresh set: {len(ben)} benign, {len(atk)} attack)")
    calib = []
    for q in (2.5, 25, 50, 75, 97.5):
        th = float(np.percentile(ths, q))
        fp = np.mean([first_cross(stat[s.sid], th) is not None for s in ben])
        de = np.mean([first_cross(stat[s.sid], th) is not None for s in atk])
        print(f"   {th:10.4f} {fp:11.2%} {de:10.1%}   (bootstrap percentile {q})")
        calib.append({"percentile": q, "threshold": th, "fp": float(fp), "detect": float(de)})
    out["calibration"] = {"shipped": cfg["threshold"], "bootstrap": [float(np.percentile(ths, q)) for q in (2.5, 50, 97.5)],
                          "operating_points": calib}

    # ---------------- B. permutation importance ----------------
    rs = np.random.default_rng(2)
    sub = [powered[i] for i in rs.permutation(len(powered))[:900]]
    base_auc = roc_auc_score(np.concatenate([s.y for s in sub]),
                             np.concatenate([score_with(s1, lstm, sub, idx, cfg["seq_len"])[1][s.sid] for s in sub]))
    pooled = np.vstack([s.X for s in sub])
    bounds = np.cumsum([0] + [len(s.X) for s in sub])
    drops = {}
    for j in idx:
        perm = pooled.copy()
        perm[:, j] = pooled[rs.permutation(len(pooled)), j]
        shuffled = []
        for k, s in enumerate(sub):
            c = type(s)(**{**s.__dict__, "X": perm[bounds[k]:bounds[k + 1]]})
            shuffled.append(c)
        sc = score_with(s1, lstm, shuffled, idx, cfg["seq_len"])[1]
        drops[FEATURE_NAMES[j]] = base_auc - roc_auc_score(np.concatenate([s.y for s in sub]),
                                                           np.concatenate([sc[s.sid] for s in sub]))
    order = sorted(drops.items(), key=lambda kv: -kv[1])
    print(f"\nB. FEATURE RELIANCE   (window AUC {base_auc:.4f} on {len(sub)} sessions; drop when shuffled)")
    for n, d in order[:10]:
        print(f"   {n:26s} {modality_of(n) if callable(modality_of) else '':9s} {d:+.4f}")
    by_mod = {}
    for n, d in drops.items():
        by_mod[modality_of(n)] = by_mod.get(modality_of(n), 0.0) + d
    print("   by modality (sum of drops): " + ", ".join(f"{m} {v:+.3f}" for m, v in sorted(by_mod.items(), key=lambda kv: -kv[1])))
    out["importance"] = {"base_auc": float(base_auc), "drops": drops, "by_modality": by_mod}

    # ---------------- C. cost ----------------
    from iudbms.capture.simulator import SimulatedSource, Simulator
    from iudbms.features.extractor import window_events
    from iudbms.live import StreamingDetector
    from iudbms.policy import PolicyEngine
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"]),
                            feature_idx=idx, seq_len=cfg["seq_len"])
    sim = Simulator(seed=77)
    work = []
    for i, prof in enumerate(("keyboard", "mouse", "mass_storage", "webcam", "audio") * 4):
        sess = sim.session(f"B{i}", prof, duration=30.0)
        det.attach(sess.descriptor, f"B{i}")
        work += [(f"B{i}", t0, b) for t0, b in window_events(list(SimulatedSource(sess).events()), duration=30.0)]
    t = time.perf_counter()
    for did, t0, b in work:
        det.observe_window(did, t0, b)
    dt = time.perf_counter() - t
    per = dt / len(work)
    print(f"\nC. COST   {len(work)} device-windows (20 devices x 30 s) in {dt:.2f} s = {per * 1e3:.2f} ms per device-window")
    print(f"   one core sustains ~{int(1.0 / per)} devices at 1 window/s each")
    out["cost"] = {"ms_per_device_window": per * 1e3, "devices_per_core": int(1.0 / per)}

    (OUT / "analysis_diagnostics.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote out/analysis_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
