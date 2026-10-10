"""
Sample-size-robust timing features and a session-age-stationary novelty feature.

    python3 -m eval.analysis_robust_features            # ~40 min from cold

**The defect (found by eval.analysis_long_sessions).** On hour-long benign keyboard
sessions the deployed model raises 0.149 false alarms per keyboard-hour -- one per
working day per keyboard -- although it showed 0.0% session FP on 40-85 s sessions,
which simply contained too few hours to reveal it. Every alarm window held exactly
two events: one keystroke, its press and its release. With a single inter-arrival
time, ``cv_iat`` = 0 and ``iat_entropy`` = 0, which is the "perfectly regular"
signature of scripted injection. Freezing either feature at its early-session value
removed all of the alarms. "No evidence" was being read as "machine-like".

A second, related non-stationarity: ``payload_seq_novelty`` remembers every
transition since enumeration, so on genuine typing its median falls from 0.50 in the
first 85 s to 0.00 after five minutes. A session-age-dependent feature is out of
distribution for a model trained on 40-85 s sessions, and it is also a channel
through which session position can leak.

**The candidate fix: three appended columns** (experimental; nothing in the frozen
contract changes unless this passes):

* ``cv_iat_shrunk``  = w*CV + (1-w)*1.0 with w = k/(k+M) over k inter-arrival times.
  Shrinkage toward the coefficient of variation of a Poisson process (1.0): the
  uninformative prior for arrival times. One keystroke cannot look perfectly
  regular; an injection window with hundreds of IATs is essentially unchanged.
* ``iat_entropy_norm`` = entropy divided by the maximum its sample size allows,
  ln(min(k, bins)) instead of ln(bins), with the same shrinkage toward 1.0. The
  existing feature caps a five-IAT window at 0.65 and a one-IAT window at 0.
* ``seq_novelty_recent`` = fraction of this window's payload transitions not seen
  in the preceding HORIZON seconds. Language repeats its transitions over minutes,
  which is normal; a cycling script repeats them within seconds, which is not.

The superseded columns (``std_iat``, ``cv_iat``, ``iat_entropy``,
``payload_seq_novelty``) are dropped from the model's input in the configurations
that replace them. Two single-fix ablations attribute the effect.

**Pre-registered acceptance (fixed before the first run).** The candidate is R_both.
It replaces the deployed configuration only if ALL hold:

  (a) keyboard false alarms per device-hour on a FRESH long-session benchmark (seed
      2027, never used before) are lower than the base model's, one-sided exact
      conditional Poisson test p < 0.05 at equal exposure;
  (b) Corpus-C (fresh, seed 2025) detection, mean over training seeds 7/8/9, is not
      lower than the base model's by more than the larger of the seed spread and one
      session per profile, and Corpus-C benign FP is not higher by more than that;
  (c) the held-out real typing capture does not trigger the gate.

Everything else printed (ablations, other device classes, late attacks) is reported,
not selected on. Writes out/analysis_robust_features.json and, for seed 7 of every
configuration, a loadable model set under out/analysis_models/robust_<cfg>/.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from iudbms.contract import FEATURE_NAMES
from iudbms.features.extractor import COUNTED_KINDS

from .analysis_common import (MODELS, OUT, deployed, featurize, run_gate, save_set, score_with,
                              session_rates, wilson)
from .analysis_long_sessions import LATE_ATTACKS, _duty, late_session
from .analysis_ngram import NGramExtractor
from .analysis_realistic import RealisticSim
from .corpus import Corpus, SessionData, load_corpus, save_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

PRIOR_M = 3.0           # prior strength, in inter-arrival times
HORIZON = 60.0          # seconds of transition memory for seq_novelty_recent
_BINS, _LO, _HI = 12, -4.0, 1.0     # the existing iat_entropy histogram
MAX_RECENT = 8192

SEEDS = (7, 8, 9)
EPOCHS, TARGET_FPR = 14, 0.0001
BENCH_SEED = 2027
BENCH_KBD_SESSIONS, BENCH_KBD_DUR = 80, 3600.0
BENCH_OTHER_SESSIONS, BENCH_OTHER_DUR = 12, 1800.0
BENCH_LATE_N, BENCH_LATE_ONSET = 20, 1200.0
REAL_CAPTURE = Path.home() / "Downloads" / "iudbms-real-captures" / "capture-20261006-161418-bus0.txt"
REAL_DESC = REAL_CAPTURE.with_name("descriptors-20261006-161418.json")
REAL_DEVICE = "1-4"

N_BASE = len(FEATURE_NAMES) + 1          # 30 contract columns + the n-gram column
NEW = {"cv_iat_shrunk": N_BASE, "iat_entropy_norm": N_BASE + 1, "seq_novelty_recent": N_BASE + 2}


class RobustExtractor(NGramExtractor):
    """The 30 contract features + the n-gram column + three robust columns."""

    def reset(self) -> None:
        super().reset()
        self._last_seen: dict[tuple[bytes, bytes], float] = {}
        self._rprev: bytes | None = None

    def _recent_novelty(self, stamped: list[tuple[float, bytes]]) -> float:
        seq = ([(None, self._rprev)] if self._rprev is not None else []) + stamped
        if len(seq) < 2:
            if stamped:
                self._rprev = stamped[-1][1]
            return 1.0
        novel = total = 0
        for (_, a), (t, b) in zip(seq, seq[1:]):
            total += 1
            key = (a, b)
            last = self._last_seen.get(key)
            if last is None or t - last > HORIZON:
                novel += 1
            if key in self._last_seen or len(self._last_seen) < MAX_RECENT:
                self._last_seen[key] = t
        if len(self._last_seen) >= MAX_RECENT:          # evict what the horizon has expired
            cutoff = seq[-1][0] - HORIZON
            self._last_seen = {k: v for k, v in self._last_seen.items() if v >= cutoff}
        self._rprev = seq[-1][1]
        return novel / total

    def extract(self, events, t0, descriptor, host=None) -> np.ndarray:
        vec = super().extract(events, t0, descriptor, host)
        counted = [e for e in events if e.evt in COUNTED_KINDS]
        ts = np.sort(np.fromiter((e.ts for e in counted), dtype=np.float64, count=len(counted)))
        k = max(0, len(ts) - 1)
        cv_s = ent_s = 1.0
        if k >= 1:
            iat = np.diff(ts)
            m = float(iat.mean())
            cv = float(iat.std() / m) if m > 0 else 0.0
            w = k / (k + PRIOR_M)
            cv_s = w * cv + (1 - w) * 1.0
            if k >= 2:
                h, _ = np.histogram(np.clip(np.log10(np.maximum(iat, 1e-6)), _LO, _HI),
                                    bins=_BINS, range=(_LO, _HI))
                p = h[h > 0] / h.sum()
                hmax = math.log(min(k, _BINS))
                ent = float(-(p * np.log(p)).sum() / hmax) if hmax > 0 else 1.0
                we = (k - 1) / (k - 1 + PRIOR_M)
                ent_s = we * ent + (1 - we) * 1.0
        stamped = [(e.ts, e.data) for e in counted if e.data]
        nov = self._recent_novelty(stamped)
        return np.append(vec, np.array([cv_s, ent_s, nov], dtype=np.float32))


def _fingerprint() -> str:
    src = inspect.getsource(RobustExtractor) + inspect.getsource(NGramExtractor)
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _cached(name: str, build) -> list[SessionData]:
    cache = OUT / f"analysis_corpus_{name}_robust.npz"
    meta = cache.with_suffix(".cond.json")
    fp = _fingerprint()
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("fp") == fp:
        print(f"  {name}: cached")
        return load_corpus(cache).sessions
    t0 = time.time()
    sessions, cond = build()
    save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), cache)
    meta.write_text(json.dumps({"fp": fp, "cond": cond}))
    print(f"  {name}: built {len(sessions)} sessions in {time.time() - t0:.0f} s")
    return sessions


def feature_sets(fit) -> dict[str, np.ndarray]:
    base = np.array([i for i in deployed_features(fit, verbose=False) if i < len(FEATURE_NAMES)])
    nm = {n: FEATURE_NAMES.index(n) for n in FEATURE_NAMES}
    timing_old = {nm["std_iat"], nm["cv_iat"], nm["iat_entropy"]}
    nov_old = {nm["payload_seq_novelty"]}
    keep = lambda drop: [i for i in base if i not in drop]          # noqa: E731
    return {
        "base": base,
        "R_timing": np.array(keep(timing_old) + [NEW["cv_iat_shrunk"], NEW["iat_entropy_norm"]]),
        "R_novelty": np.array(keep(nov_old) + [NEW["seq_novelty_recent"]]),
        "R_both": np.array(keep(timing_old | nov_old) + list(NEW.values())),
    }


def benchmark_sessions():
    """The fresh long-session benchmark, featurised once with the robust extractor."""
    out = {"keyboard": [], "other": [], "late": []}
    sim, rng = RealisticSim(seed=BENCH_SEED), np.random.default_rng(BENCH_SEED)
    for i in range(BENCH_KBD_SESSIONS):
        _duty(sim, rng)
        sid = f"BK{i:03d}"
        out["keyboard"].append(featurize(sim.session(sid, "keyboard", duration=BENCH_KBD_DUR),
                                         sid, "keyboard", RobustExtractor))
    for prof in ("mouse", "mass_storage", "webcam", "audio"):
        for i in range(BENCH_OTHER_SESSIONS):
            _duty(sim, rng)
            sid = f"BO-{prof}-{i:03d}"
            out["other"].append(featurize(sim.session(sid, prof, duration=BENCH_OTHER_DUR),
                                          sid, prof, RobustExtractor))
    sim2, rng2 = RealisticSim(seed=BENCH_SEED + 1), np.random.default_rng(BENCH_SEED + 1)
    for prof in LATE_ATTACKS:
        for i in range(BENCH_LATE_N):
            _duty(sim2, rng2)
            sid = f"BL-{prof}-{i:03d}"
            out["late"].append(featurize(late_session(sim2, sid, prof, BENCH_LATE_ONSET),
                                         sid, prof, RobustExtractor))
    return out


def real_capture_session() -> SessionData | None:
    if not REAL_CAPTURE.exists():
        return None
    from iudbms.capture.usbmon import UsbmonSource
    from iudbms.features.extractor import window_events
    from iudbms.live.cli import _load_descriptor_snapshot
    desc = _load_descriptor_snapshot(REAL_DESC)[REAL_DEVICE]
    ev = [e for e in UsbmonSource(path=REAL_CAPTURE).events() if e.device_key == REAL_DEVICE]
    ex, X, T = RobustExtractor(), [], []
    for t0, b in window_events(ev, duration=max(e.ts for e in ev)):
        X.append(ex.extract(b, t0, desc))
        T.append(t0)
    return SessionData(sid="real", profile="keyboard", label=0, descriptor=desc,
                       X=np.vstack(X), t=np.array(T), y=np.zeros(len(T), int), onset=None,
                       duration=float(T[-1]) + 1)


def evaluate(models, idx, thr, bench, corpus_c, real) -> dict:
    s1, lstm = models
    r: dict = {}
    _, S2, _ = score_with(s1, lstm, bench["keyboard"], idx)
    first = run_gate(S2, bench["keyboard"], thr)
    hours = sum(s.duration for s in bench["keyboard"]) / 3600.0
    alarms = sum(first[s.sid] is not None for s in bench["keyboard"])
    over = float(np.mean(np.concatenate([S2[s.sid] for s in bench["keyboard"]]) >= thr))
    r["keyboard"] = {"sessions": len(bench["keyboard"]), "hours": hours, "alarmed_sessions": alarms,
                     "alarms_per_hour": alarms / hours, "windows_over": over}
    _, S2, _ = score_with(s1, lstm, bench["other"], idx)
    first = run_gate(S2, bench["other"], thr)
    r["other"] = {p: sum(first[s.sid] is not None for s in bench["other"] if s.profile == p)
                  for p in ("mouse", "mass_storage", "webcam", "audio")}
    r["other_hours"] = sum(s.duration for s in bench["other"]) / 3600.0
    _, S2, _ = score_with(s1, lstm, bench["late"], idx)
    first = run_gate(S2, bench["late"], thr)
    r["late"] = {}
    for p in LATE_ATTACKS:
        g = [s for s in bench["late"] if s.profile == p]
        pre = sum(first[s.sid] is not None and first[s.sid] <= s.onset for s in g)
        hit = sum(first[s.sid] is not None and first[s.sid] > s.onset for s in g)
        r["late"][p] = {"n": len(g), "detected_after_onset": hit, "alarm_before_onset": pre}
    _, S2, _ = score_with(s1, lstm, corpus_c, idx)
    first = run_gate(S2, corpus_c, thr)
    rc = session_rates(first, corpus_c)
    per = {}
    for p in sorted({s.profile for s in corpus_c if s.label == 1}):
        g = [s for s in corpus_c if s.profile == p]
        per[p] = sum(first[s.sid] is not None for s in g) / len(g)
    r["corpus_c"] = {"detect": rc["detect"], "detect_k": rc["detect_k"], "n_attack": rc["n_attack"],
                     "fp": rc["fp"], "fp_k": rc["fp_k"], "n_benign": rc["n_benign"],
                     "median_latency": rc["median_latency"], "per_profile": per}
    if real is not None:
        _, S2, _ = score_with(s1, lstm, [real], idx)
        f = run_gate(S2, [real], thr)["real"]
        busy = real.X[:, 0] > 0
        r["real_typing"] = {"gate_fires": f is not None, "first_alert_s": f,
                            "windows_over": int((S2["real"][busy] >= thr).sum()),
                            "busy_windows": int(busy.sum()),
                            "median_score": float(np.median(S2["real"][busy]))}
    return r


def poisson_one_sided(x_new: int, x_old: int) -> float:
    """P(X_new <= x_new | X_new + X_old = n) under equal rates and equal exposure."""
    n = x_new + x_old
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(x_new + 1)) / 2 ** n


def main() -> int:
    t_start = time.time()
    from .analysis_powered import build as build_c
    from .analysis_realistic import build as build_b
    print("corpora (robust extractor):")
    B = _cached("realistic", lambda: build_b(extractor_cls=RobustExtractor))
    C = _cached("powered", lambda: build_c(extractor_cls=RobustExtractor))
    fit, val, test = stratified_split(B, seed=7)
    sets = feature_sets(fit)
    dep = json.loads((MODELS / "deploy.json").read_text())["feature_idx"]
    assert list(sets["base"]) == dep, f"base set {list(sets['base'])} != deployed {dep}"
    for k, v in sets.items():
        print(f"  {k:10s} {len(v)} features")

    print("long-session benchmark (fresh seed 2027) ...", flush=True)
    bench = benchmark_sessions()
    real = real_capture_session()
    print(f"  {len(bench['keyboard'])} keyboard sessions x {BENCH_KBD_DUR:.0f} s, "
          f"{len(bench['other'])} other, {len(bench['late'])} late-onset attacks; "
          f"real capture {'loaded' if real is not None else 'MISSING'}  "
          f"[{time.time() - t_start:.0f} s]", flush=True)

    results: dict = {}
    for name, idx in sets.items():
        for seed in SEEDS:
            t0 = time.time()
            r = run_configuration(fit, val, test, name=name, feature_idx=idx, seed=seed,
                                  epochs=EPOCHS, seq_len=10, hidden=32, target_fpr=TARGET_FPR,
                                  verbose=False)
            if name == "base" and seed == 7:
                # identical corpus columns + identical seed => the deployed model, exactly
                assert abs(r.threshold - dep_threshold()) < 1e-6, \
                    f"base/seed 7 threshold {r.threshold} != deployed {dep_threshold()}"
            if seed == 7:
                save_set(OUT / "analysis_models" / f"robust_{name}", r, idx, target_fpr=TARGET_FPR)
            ev = evaluate(r.models, idx, r.threshold, bench, C, real)
            ev["corpus_b_test"] = {"detect": r.session["attack_detect"], "fp": r.session["benign_fp"],
                                   "auc": r.window["auc"]}
            ev["threshold"] = r.threshold
            results.setdefault(name, {})[str(seed)] = ev
            k = ev["keyboard"]
            print(f"  {name:10s} seed {seed}: kbd alarms {k['alarmed_sessions']:2d} in {k['hours']:.0f} h "
                  f"({k['alarms_per_hour']:.3f}/h)  C detect {ev['corpus_c']['detect']:.1%} "
                  f"FP {ev['corpus_c']['fp_k']}/{ev['corpus_c']['n_benign']}  "
                  f"real {'FIRES' if ev.get('real_typing', {}).get('gate_fires') else 'quiet'}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)

    # ------------------------------ verdict ------------------------------
    def agg(name, f):
        v = [f(results[name][str(s)]) for s in SEEDS]
        return float(np.mean(v)), float(np.std(v)), v
    print("\nSUMMARY (mean +/- sd over seeds 7/8/9)")
    print(f"  {'config':10s} {'kbd alarms/h':>14s} {'C detect':>14s} {'C benign FP':>14s} "
          f"{'B-test detect':>14s}  late (det/pre-alarm)")
    summary = {}
    for name in sets:
        a = agg(name, lambda e: e["keyboard"]["alarms_per_hour"])
        d = agg(name, lambda e: e["corpus_c"]["detect"])
        f = agg(name, lambda e: e["corpus_c"]["fp"])
        b = agg(name, lambda e: e["corpus_b_test"]["detect"])
        late = {p: (sum(results[name][str(s)]["late"][p]["detected_after_onset"] for s in SEEDS),
                    sum(results[name][str(s)]["late"][p]["alarm_before_onset"] for s in SEEDS))
                for p in LATE_ATTACKS}
        summary[name] = {"kbd_alarms_per_hour": a, "corpus_c_detect": d, "corpus_c_fp": f,
                         "corpus_b_detect": b, "late": late}
        print(f"  {name:10s} {a[0]:7.3f}+/-{a[1]:.3f} {d[0]:8.1%}+/-{d[1]:.1%} "
              f"{f[0]:8.2%}+/-{f[1]:.2%} {b[0]:8.1%}+/-{b[1]:.1%}  "
              + " ".join(f"{p[:6]}:{late[p][0]}/{late[p][1]}" for p in LATE_ATTACKS))

    # (a) exact conditional Poisson test, pooled over seeds (equal exposure by construction)
    x_new = sum(results["R_both"][str(s)]["keyboard"]["alarmed_sessions"] for s in SEEDS)
    x_old = sum(results["base"][str(s)]["keyboard"]["alarmed_sessions"] for s in SEEDS)
    p_a = poisson_one_sided(x_new, x_old)
    a_ok = x_new < x_old and p_a < 0.05
    # (b) detection / FP on Corpus C within noise
    n_per = min(sum(1 for s in C if s.profile == p) for p in {s.profile for s in C if s.label == 1})
    floor_d = max(summary["base"]["corpus_c_detect"][1], summary["R_both"]["corpus_c_detect"][1], 1 / n_per)
    d_ok = summary["R_both"]["corpus_c_detect"][0] >= summary["base"]["corpus_c_detect"][0] - floor_d
    n_ben = results["base"]["7"]["corpus_c"]["n_benign"]
    floor_f = max(summary["base"]["corpus_c_fp"][1], summary["R_both"]["corpus_c_fp"][1], 1 / n_ben)
    f_ok = summary["R_both"]["corpus_c_fp"][0] <= summary["base"]["corpus_c_fp"][0] + floor_f
    # (c) real typing
    c_ok = all(not results["R_both"][str(s)].get("real_typing", {"gate_fires": True})["gate_fires"]
               for s in SEEDS)
    verdict = {"a_kbd_alarms": {"pass": a_ok, "R_both": x_new, "base": x_old, "p": p_a},
               "b_detection": {"pass": d_ok, "floor": floor_d},
               "b_fp": {"pass": f_ok, "floor": floor_f},
               "c_real_typing": {"pass": c_ok},
               "adopt": a_ok and d_ok and f_ok and c_ok}
    print(f"\nPRE-REGISTERED VERDICT")
    print(f"  (a) keyboard alarms, pooled over seeds: R_both {x_new} vs base {x_old}, "
          f"one-sided p = {p_a:.4f}  -> {'PASS' if a_ok else 'FAIL'}")
    print(f"  (b) Corpus-C detection within {floor_d:.1%}: {'PASS' if d_ok else 'FAIL'};  "
          f"benign FP within {floor_f:.2%}: {'PASS' if f_ok else 'FAIL'}")
    print(f"  (c) real typing capture quiet under every seed: {'PASS' if c_ok else 'FAIL'}")
    print(f"  => {'ADOPT R_both' if verdict['adopt'] else 'DO NOT ADOPT; keep the deployed model'}")

    (OUT / "analysis_robust_features.json").write_text(json.dumps(
        {"pre_registered": __doc__.split("**Pre-registered acceptance")[1].split("Everything else")[0],
         "prior_m": PRIOR_M, "horizon_s": HORIZON, "seeds": SEEDS,
         "feature_sets": {k: [int(i) for i in v] for k, v in sets.items()},
         "benchmark": {"seed": BENCH_SEED, "keyboard_sessions": BENCH_KBD_SESSIONS,
                       "keyboard_duration_s": BENCH_KBD_DUR, "other_sessions_per_class": BENCH_OTHER_SESSIONS,
                       "other_duration_s": BENCH_OTHER_DUR, "late_n": BENCH_LATE_N,
                       "late_onset_s": BENCH_LATE_ONSET},
         "results": results, "summary": summary, "verdict": verdict,
         "runtime_s": time.time() - t_start}, indent=1, default=float))
    print(f"\nwrote out/analysis_robust_features.json  ({time.time() - t_start:.0f} s)")
    return 0


def dep_threshold() -> float:
    return float(json.loads((MODELS / "deploy.json").read_text())["threshold"])


if __name__ == "__main__":
    raise SystemExit(main())
