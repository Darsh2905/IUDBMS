"""
Experiment 3 -- realistic HID keyboards, and a held-out check against real typing.

    python3 -m eval.analysis_realistic

The first real keyboard session this project captured is flagged by the evaluated
model, and by the idle-aware model of experiment 2. The measured gap between a
real keyboard and the simulator's (CLAUDE.md finding 29) is structural, not just
speed: real devices send a report on key-*release* as well as key-press, overlap
keys (rollover), and type language, whose bigrams repeat. This experiment makes the
simulated keyboards behave that way, retrains, and asks whether the real capture is
still flagged.

**The real capture is held out, and every design choice below is made from outside
it.** Nothing here is fitted to that recording; if it were, passing it would mean
nothing. The choices and their sources:

* *Releases + rollover.* The USB HID boot protocol sends a report on every key-state
  change, so a keystroke is a key-down and a key-up report, and overlapping keys
  yield multi-key reports. A property of the specification.
* *Language-like keystrokes.* A first-order Markov source whose conditional entropy is
  tuned to Shannon's (1951) digram estimate for English, ~3.56 bits/symbol, because
  uniformly random keycodes (the old source) never repeat a bigram the way text does.
* *Typing speed.* Persona inter-key median log-uniform over 60-300 ms: fast typists
  sustain ~60-100 ms between keystrokes, hunt-and-peck is >300 ms.
* *Hold time.* Human key-hold ~ log-normal, median 90 ms (a typing-literature figure).
  Scripted injection releases at 45% of its inter-key interval: a script sends a
  press and then a release, not an overlapping chord.
* *Attacks get the same treatment.* Injection also emits releases (a real BadUSB must),
  the mimicry loop still replays a fixed script (its tell is that it cycles), and the
  adaptive payload is language-like text typed once. Benign-only realism would hand
  the model a new, fake "has release reports => benign" shortcut.

Trains nothing outside out/analysis_models/ and out/analysis_*.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES, INTR, DevicePersona

from .analysis_common import (OUT, deployed, featurize, per_profile, run_gate, save_set,
                              score_with, session_rates)
from .analysis_idle import DutyCycleSim
from .analysis_ngram import NGramExtractor
from .analysis_idle_retrain import IDLE_MIX, QUIET_CLASSES
from .corpus import Corpus, load_corpus, save_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

CACHE = OUT / "analysis_corpus_realistic_v2.npz"
SEED, EPOCHS = 7, 14

KEYS = tuple(range(4, 40))                  # 36 usage codes: letters, digits
SHANNON_DIGRAM_BITS = 3.56                  # English, conditional on previous letter
SPEED_RANGE = (0.4, 2.0)                    # persona multiplier on the 150 ms median
HOLD_MEDIAN, HOLD_SIGMA = 0.09, 0.35        # human key-hold, seconds


# ---------------------------------------------------------------------------
# language-like keystroke source
# ---------------------------------------------------------------------------

def _conditional_entropy(P: np.ndarray) -> float:
    pi = np.full(len(P), 1.0 / len(P))
    for _ in range(200):
        pi = pi @ P
    return float(-(pi[:, None] * P * np.log2(np.clip(P, 1e-12, None))).sum())


def build_markov(seed: int = 0, target_bits: float = SHANNON_DIGRAM_BITS) -> np.ndarray:
    """Transition matrix with Zipfian unigrams and a conditional entropy of ``target_bits``.

    Row i is softmax(log zipf + sigma * noise_i); sigma is bisected until the stationary
    conditional entropy matches the target. Peakier rows = more repeated bigrams.
    """
    rng = np.random.default_rng(seed)
    n = len(KEYS)
    log_zipf = -np.log(np.arange(1, n + 1))
    noise = rng.standard_normal((n, n))
    lo, hi = 0.0, 6.0
    for _ in range(40):
        sig = (lo + hi) / 2
        z = log_zipf[None, :] + sig * noise
        P = np.exp(z - z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        h = _conditional_entropy(P)
        lo, hi = (sig, hi) if h > target_bits else (lo, sig)
    return P


# ---------------------------------------------------------------------------
# the realistic simulator
# ---------------------------------------------------------------------------

class RealisticSim(DutyCycleSim):
    """Duty cycles + HID key-release/rollover + language-like keys + speed range."""

    def __init__(self, *a, markov_seed: int = 0, text: str = "words",
                 mimicry_hold: str = "short", **k):
        super().__init__(*a, device_variation=True, **k)
        #: "short": the replaying device presses and releases without overlap (what a
        #: script does). "human": it also imitates human key overlap (rollover), which
        #: makes every cycle's reports differ. The detection result hinges on this.
        self._mimicry_hold = mimicry_hold
        self._P = build_markov(markov_seed)
        self._text = text
        self._vocab, self._word_p = self._build_vocab(markov_seed)
        self._queue: list[int] = []
        self._pending: dict = {}
        self._keysrc = "script"            # "markov": remap key codes; "script": keep them
        self._hold_mode = "human"
        self._script_hold = 0.002
        self._prev_key = 0

    def _sample_persona(self) -> DevicePersona:
        u = self.rng.uniform
        return DevicePersona(
            iat_scale=math.exp(u(math.log(SPEED_RANGE[0]), math.log(SPEED_RANGE[1]))),
            sigma_scale=u(0.8, 1.3), pause_scale=u(0.7, 1.5),
            burst_shift=self.rng.choice((-2, -1, 0, 0, 1, 2)))

    # -- key source / hold ---------------------------------------------------
    def _build_vocab(self, seed: int, n_words: int = 400, s: float = 1.0):
        """A vocabulary of words drawn from the Markov source, used with Zipf frequencies.

        Real text repeats whole words (Zipf's law), so a window of typing contains
        subsequences that occurred earlier in the session. A first-order Markov chain
        alone has bigram structure but no such repeats, and a long-repeat detector
        validated against it would look better than it should.
        """
        rng = np.random.default_rng(seed + 1)
        vocab = []
        for _ in range(n_words):
            state, w = int(rng.integers(len(KEYS))), []
            for _ in range(int(rng.integers(2, 10))):
                state = int(rng.choice(len(KEYS), p=self._P[state]))
                w.append(KEYS[state])
            vocab.append(w)
        p = 1.0 / np.arange(1, n_words + 1) ** s
        return vocab, p / p.sum()

    def _next_key(self) -> int:
        if self._text == "markov":
            self._prev_key = int(self.np_rng.choice(len(KEYS), p=self._P[self._prev_key]))
            return KEYS[self._prev_key]
        if not self._queue:
            w = self._vocab[int(self.np_rng.choice(len(self._vocab), p=self._word_p))]
            self._queue = list(w) + [44]            # 44 = space
        return self._queue.pop(0)

    def _hold_time(self) -> float:
        if self._hold_mode == "script":
            return self._script_hold
        if self._hold_mode == "short":
            # a device replaying a script presses and releases without overlapping keys
            return float(self.np_rng.uniform(0.02, 0.04))
        return float(min(0.4, max(0.03, self.np_rng.lognormal(math.log(HOLD_MEDIAN), HOLD_SIGMA))))

    # -- report emission with releases ----------------------------------------
    def _flush(self, lines, bus, dev, up_to):
        pend = self._pending.get((bus, dev), [])
        pend.sort(key=lambda x: x[0])
        while pend and pend[0][0] <= up_to:
            rt, _k, ep, interval = pend.pop(0)
            down = [kk for _, kk, _, _ in pend][:6]
            rep = bytes([0, 0] + down + [0] * (6 - len(down)))
            super()._urb(lines, rt, INTR, "in", bus, dev, ep, rep, interval=interval)

    def _urb(self, lines, ts, ttype, direction, bus, dev, ep, payload,
             length=None, interval=None, capture=True):
        if ttype == INTR and direction == "in" and len(payload) == 8 and payload[2] != 0:
            self._flush(lines, bus, dev, ts)
            key = self._next_key() if self._keysrc == "markov" else payload[2]
            pend = self._pending.setdefault((bus, dev), [])
            pend.append((ts + self._hold_time(), key, ep, interval))
            down = sorted(k for _, k, _, _ in pend)[:6]
            rep = bytes([payload[0], 0] + down + [0] * (6 - len(down)))
            return super()._urb(lines, ts, ttype, direction, bus, dev, ep, rep,
                                length, interval, capture)
        return super()._urb(lines, ts, ttype, direction, bus, dev, ep, payload,
                            length, interval, capture)

    # -- per-generator mode ------------------------------------------------------
    def _with(self, src, hold, fn, *a, **kw):
        old = (self._keysrc, self._hold_mode)
        self._keysrc, self._hold_mode = src, hold
        try:
            return fn(*a, **kw)
        finally:
            self._keysrc, self._hold_mode = old

    def _human_typing(self, t, dur, bus, dev, lines, ep=1):
        return self._with("markov", "human", super()._human_typing, t, dur, bus, dev, lines, ep)

    def _adaptive_typing(self, t, dur, bus, dev, lines, ep=1):
        return self._with("markov", "human", super()._adaptive_typing, t, dur, bus, dev, lines, ep)

    def _mimicry_typing(self, t, dur, bus, dev, lines, ep=1):
        return self._with("script", self._mimicry_hold, super()._mimicry_typing,
                          t, dur, bus, dev, lines, ep)

    def _injected_typing(self, t, dur, bus, dev, lines, ep=1, mean_iat=0.0045, jitter=0.14):
        self._script_hold = max(0.0005, 0.45 * mean_iat)
        return self._with("script", "script", super()._injected_typing,
                          t, dur, bus, dev, lines, ep, mean_iat, jitter)

    def session(self, sid, profile, duration=None):
        self._pending = {}
        self._prev_key = 0
        self._queue = []
        s = super().session(sid, profile, duration)
        # any key still down at the end is released, then restore time order
        for (bus, dev), pend in list(self._pending.items()):
            self._flush(s.lines, bus, dev, up_to=float("inf"))
        s.lines.sort(key=lambda ln: int(ln.split()[1]))        # stable: keeps S before C
        return s


# ---------------------------------------------------------------------------
# corpus + experiment
# ---------------------------------------------------------------------------

def build(n_benign=600, n_attack=400, seed=SEED):
    sim = RealisticSim(seed=seed)
    rng = np.random.default_rng(seed + 99)
    probs = np.array([m[1] for m in IDLE_MIX])
    sessions, cond = [], {}
    for sid, profile in sim.plan(n_benign=n_benign, n_attack=n_attack):
        if profile in QUIET_CLASSES:
            name, _, kw = IDLE_MIX[int(rng.choice(len(IDLE_MIX), p=probs))]
        else:
            name, kw = "attack", IDLE_MIX[0][2]
        sim._idle, sim._active, sim._fully_idle = kw["idle"], kw["active"], kw["fully_idle"]
        sessions.append(featurize(sim.session(sid, profile), sid, profile, NGramExtractor))
        cond[sid] = name
    return sessions, cond


def keyboard_profile(sim: RealisticSim, n: int = 12) -> dict:
    """Wire-level stats of simulated typing, measured as eval.real_typing measures real."""
    from iudbms.capture.simulator import SimulatedSource

    from .real_typing import _keyboard_stats
    sim._idle, sim._active, sim._fully_idle = (0.0, 0.0), (1e9, 1e9), False
    rows = []
    for i in range(n):
        s = sim.session(f"k{i}", "keyboard")
        ev = list(SimulatedSource(s).events())
        rows.append(_keyboard_stats(ev, s.descriptor.device_key))
    keys = ("reports_per_s", "release_share", "unique_payload_share", "gaps_over_1s")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["iat_median"] = float(np.mean([r["iat_percentiles"][50] for r in rows]))
    return out


def main() -> int:
    t0 = time.time()
    P = build_markov(0)
    h = _conditional_entropy(P)
    print(f"keystroke source: conditional entropy {h:.2f} bits (target {SHANNON_DIGRAM_BITS}), "
          f"uniform would be {math.log2(len(KEYS)):.2f}")
    assert abs(h - SHANNON_DIGRAM_BITS) < 0.05, "Markov source missed its entropy target"

    meta = CACHE.with_suffix(".cond.json")
    if CACHE.exists() and meta.exists():
        sessions, cond = load_corpus(CACHE).sessions, json.loads(meta.read_text())
        print(f"loaded cached realistic corpus: {len(sessions)} sessions")
    else:
        sessions, cond = build()
        save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), CACHE)
        meta.write_text(json.dumps(cond))
        print(f"built realistic corpus: {len(sessions)} sessions in {time.time() - t0:.0f} s")

    # --- how close is the new keyboard to a real one on the wire? -------------
    real = json.loads((OUT / "real_typing.json").read_text())["capture"]
    sim = keyboard_profile(RealisticSim(seed=3))
    print("\nwire-level shape, real mashing session vs realistic simulated typing "
          "(prose-like speed range, so rate is expected to differ):")
    print(f"  {'':28s}{'real':>9s}{'realistic sim':>15s}")
    for label, rk, sk, f in (("reports/s", "reports_per_s", "reports_per_s", "{:.1f}"),
                             ("release (all-zero) share", "release_share", "release_share", "{:.1%}"),
                             ("distinct payload share", "unique_payload_share", "unique_payload_share", "{:.1%}"),
                             ("gaps over 1 s", "gaps_over_1s", "gaps_over_1s", "{:.1%}")):
        print(f"  {label:28s}{f.format(real[rk]):>9s}{f.format(sim[sk]):>15s}")

    fit, val, test = stratified_split(sessions, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    r = run_configuration(fit, val, test, name="realistic", feature_idx=idx, seed=SEED,
                          epochs=EPOCHS, seq_len=10, hidden=32, target_fpr=0.0001,
                          keep_scores=True, verbose=False)
    save_set(OUT / "analysis_models" / "realistic_v2", r, idx, target_fpr=0.0001)

    # three models, one test split
    sets = {"original": OUT / "models", "idle-aware": OUT / "analysis_models" / "idle_aware",
            "realistic": OUT / "analysis_models" / "realistic_v2"}
    first, win_auc = {}, {}
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([s.y for s in test])
    for name, d in sets.items():
        cfg, s1, lstm = deployed(d)
        S2 = score_with(s1, lstm, test, np.array(cfg["feature_idx"]), cfg["seq_len"])[1]
        first[name] = run_gate(S2, test, cfg["threshold"], cfg["k"], cfg["n"])
        win_auc[name] = float(roc_auc_score(y, np.concatenate([S2[s.sid] for s in test])))

    ben = [s for s in test if s.label == 0]
    out = {"benign": {}, "attack": {}, "window_auc": win_auc,
           "wire": {"real": {k: real[k] for k in ("reports_per_s", "release_share",
                                                  "unique_payload_share", "gaps_over_1s")},
                    "corpus_a_sim": {k: real["simulator_measured"][k] for k in (
                        "reports_per_s", "release_share", "unique_payload_share", "gaps_over_1s")},
                    "corpus_b_sim": sim}}
    print(f"\nBENIGN session FP by condition   (test split of the realistic corpus)")
    print(f"  {'condition':10s} {'n':>4s}" + "".join(f"{n:>12s}" for n in sets))
    for name, *_ in IDLE_MIX:
        g = [s for s in ben if cond[s.sid] == name]
        if not g:
            continue
        rates = {m: sum(first[m][s.sid] is not None for s in g) / len(g) for m in sets}
        print(f"  {name:10s} {len(g):4d}" + "".join(f"{rates[m]:12.1%}" for m in sets))
        out["benign"][name] = rates
        out.setdefault("benign_n", {})[name] = len(g)
    ov = {m: session_rates(first[m], test) for m in sets}
    print(f"  {'ALL':10s} {len(ben):4d}" + "".join(f"{ov[m]['fp']:12.1%}" for m in sets))
    kb = [s for s in ben if s.profile == "keyboard"]
    print(f"  {'keyboards':10s} {len(kb):4d}" + "".join(
        f"{sum(first[m][s.sid] is not None for s in kb) / len(kb):12.1%}" for m in sets))
    out["benign"]["keyboards"] = {m: sum(first[m][s.sid] is not None for s in kb) / len(kb) for m in sets}
    out["benign_n"]["keyboards"], out["benign_n"]["ALL"] = len(kb), len(ben)
    out["attack_n"] = {p: sum(1 for s in test if s.profile == p) for p in ATTACK_PROFILES}

    print("\nATTACK detection by profile")
    print(f"  {'profile':18s} {'n':>3s}" + "".join(f"{n:>12s}" for n in sets) + "   realistic 95% CI")
    pp = {m: per_profile(first[m], test) for m in sets}
    for p in sorted(pp["realistic"]):
        if pp["realistic"][p]["label"] != 1:
            continue
        lo, hi = pp["realistic"][p]["ci"]
        print(f"  {p:18s} {pp['realistic'][p]['n']:3d}"
              + "".join(f"{pp[m][p]['rate']:12.1%}" for m in sets) + f"   {lo:.0%}-{hi:.0%}")
        out["attack"][p] = {m: pp[m][p]["rate"] for m in sets}
    print(f"  {'ALL':18s} {ov['original']['n_attack']:3d}"
          + "".join(f"{ov[m]['detect']:12.1%}" for m in sets))
    print(f"  window AUC:        " + "".join(f"{win_auc[m]:>12.4f}" for m in sets))
    out["overall"] = {m: {"detect": ov[m]["detect"], "fp": ov[m]["fp"],
                          "median_latency": ov[m]["median_latency"]} for m in sets}

    # --- the held-out real capture -------------------------------------------
    import glob
    from pathlib import Path

    from .real_typing import rescore_report
    cap_dir = Path.home() / "Downloads" / "iudbms-real-captures"
    cap = sorted(glob.glob(str(cap_dir / "capture-*.txt")))
    desc = sorted(glob.glob(str(cap_dir / "descriptors-*.json")))
    if cap and desc:
        print("\nHELD-OUT REAL TYPING (1,708 reports; never used to build or tune anything above)")
        out["real"] = {}
        for name, d in sets.items():
            rr = rescore_report(Path(cap[-1]), Path(desc[-1]), "1-4", d)
            fired = f"fires at {rr['first_alert_s']:.0f} s" if rr["gate_fires"] else "does NOT fire"
            print(f"  {name:11s} busy windows >= threshold: {rr['at_or_above_threshold']:2d}/{rr['busy']}"
                  f"   median stage-2 {rr['stage2_median']:.3f}   gate {fired}")
            out["real"][name] = {k: rr[k] for k in ("at_or_above_threshold", "busy",
                                                    "stage2_median", "gate_fires", "first_alert_s")}
    else:
        print("\n(real capture not found at ~/Downloads/iudbms-real-captures; skipped)")

    (OUT / "analysis_realistic_v2.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote out/analysis_realistic_v2.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
