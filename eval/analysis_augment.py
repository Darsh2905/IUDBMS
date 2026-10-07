"""
Experiment 8 -- can synthetic "behaviour swap" attacks teach Stage 2 to catch unseen families?

    python3 -m eval.analysis_augment

Leave-one-out (eval.analysis_lopo) showed Stage 2 collapses on attack families it was
not trained on: ``class_exfil`` and ``identity_switch`` fall from 100% to 0%. The
standard remedy is *anomaly injection*: train on synthetic anomalies built from a
**generic principle** instead of only the hand-written families. The principle here is
the one the project's threat model is built on -- a device that declares one class and
then behaves like another.

A swap session declares class A (descriptor, enumeration), behaves as A until a random
onset, then behaves as class B != A. Over five benign classes that is 20 (A, B) pairs.

**The leak to avoid, and how.** If the held-out family *is* one of the swaps, the model
has simply seen it. So for each held-out family the swap equivalent to it is removed:

    class_exfil     (keyboard declared, covert bulk traffic)  ->  drop keyboard->mass_storage
    identity_switch (storage declared, then keystroke injection) -> drop mass_storage->keyboard

The remaining 19 swaps are *different* behaviours; if they teach the model anything that
transfers, it is the general principle -- that is the claim under test, and it can fail.
``identity_switch`` additionally carries a late-appearing interface that no swap has, so
it is a harder test of transfer than ``class_exfil``.

Evaluated on the fresh seed-2025 set (100 sessions per family, 3,000 benign), with the
same 4-of-6 gate and per-window 1e-4 calibration as everything else.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import DURATION_RANGE, SimulatedSession
from iudbms.contract import CLASS_AUDIO, CLASS_HID, CLASS_MASS_STORAGE, CLASS_VIDEO, Endpoint

from .analysis_common import OUT, featurize, run_gate, score_with, wilson
from .analysis_ngram import NGramExtractor
from .analysis_powered import load_or_build
from .analysis_realistic import CACHE as TRAIN_CACHE
from .analysis_realistic import RealisticSim
from .corpus import load_corpus, stratified_split
from .pipeline import deployed_features, run_configuration

CLASSES = ("keyboard", "mouse", "mass_storage", "webcam", "audio")
SEED, EPOCHS = 7, 14
HELD = {"class_exfil": ("keyboard", "mass_storage"), "identity_switch": ("mass_storage", "keyboard")}


class SwapSim(RealisticSim):
    """Adds sessions that declare one class and behave as another after an onset."""

    def _declare(self, cls, bus, dev, lines):
        """Descriptor + enumeration for a declared class, as Simulator.session builds them."""
        if cls == "keyboard":
            d = self._descriptor("keyboard", bus, dev,
                                 [self._iface(CLASS_HID, 0x01, 0x01, self._HID_EPS)])
        elif cls == "mouse":
            d = self._descriptor("mouse", bus, dev, [self._iface(
                CLASS_HID, 0x01, 0x02, (Endpoint(0x81, "intr", 8),))])
        elif cls == "mass_storage":
            d = self._descriptor("mass_storage", bus, dev, [self._iface(
                CLASS_MASS_STORAGE, 0x06, 0x50, self._STORAGE_EPS)])
        elif cls == "webcam":
            d = self._descriptor("webcam", bus, dev, [self._iface(
                CLASS_VIDEO, 0x01, 0x00, (Endpoint(0x83, "iso"),))])
        else:
            d = self._descriptor("audio", bus, dev, [self._iface(
                CLASS_AUDIO, 0x02, 0x00, (Endpoint(0x83, "iso"),))])
        t = self._enumerate(lines, bus, dev, d.vid, d.pid)
        if cls in ("keyboard", "mouse"):
            t = self._hid_class_setup(lines, t, bus, dev)
        return d, t

    def _behave(self, cls, t, dur, bus, dev, lines):
        if cls == "keyboard":
            return self._human_typing(t, dur, bus, dev, lines)
        if cls == "mouse":
            return self._mouse_motion(t, dur, bus, dev, lines)
        if cls == "mass_storage":
            return self._bulk_io(t, dur, bus, dev, lines)
        if cls == "webcam":
            return self._iso_stream(t, dur, bus, dev, lines)
        return self._iso_stream(t, dur, bus, dev, lines, period=0.005, size=192)

    def swap_session(self, sid, declared, behaviour):
        rng = self.rng
        bus = rng.randint(1, 3)
        dev = self._devnum
        self._devnum = 2 + (self._devnum - 1) % 120
        self._persona = self._sample_persona()
        self._pending, self._prev_key, self._queue = {}, 0, []
        self._idle, self._active, self._fully_idle = (0.0, 0.0), (1e9, 1e9), False
        lines: list[str] = []
        dur = rng.uniform(*DURATION_RANGE)
        d, t = self._declare(declared, bus, dev, lines)
        onset = rng.uniform(5.0, 25.0)
        self._behave(declared, t, onset - t, bus, dev, lines)
        self._behave(behaviour, onset, dur - onset, bus, dev, lines)
        for (b, dv), _ in list(self._pending.items()):
            self._flush(lines, b, dv, up_to=float("inf"))
        lines.sort(key=lambda ln: int(ln.split()[1]))
        return SimulatedSession(sid=sid, profile="swap", label=1, descriptor=d, lines=lines,
                                onset=onset, duration=dur)


def swap_corpus(exclude, per_pair=8, seed=SEED + 500):
    sim = SwapSim(seed=seed)
    out = []
    for a in CLASSES:
        for b in CLASSES:
            if a == b or (a, b) == exclude:
                continue
            for i in range(per_pair):
                sid = f"W{a[:2]}{b[:2]}{i}"
                out.append(featurize(sim.swap_session(sid, a, b), sid, "swap", NGramExtractor))
    return out


def swap_detection(models, thr, idx, exclude, per_pair=5):
    """Detection of FRESH swap sessions: pairs it trained on vs the one that was withheld."""
    sim = SwapSim(seed=SEED + 900)
    seen, held = [], []
    for a in CLASSES:
        for b in CLASSES:
            if a == b:
                continue
            ss = [featurize(sim.swap_session(f"F{a[:2]}{b[:2]}{i}", a, b), f"F{a[:2]}{b[:2]}{i}", "swap",
                            NGramExtractor) for i in range(per_pair)]
            S2 = score_with(models[0], models[1], ss, idx, 10)[1]
            first = run_gate(S2, ss, thr, 4, 6)
            rate = float(np.mean([first[x.sid] is not None for x in ss]))
            (held if (a, b) == exclude else seen).append(rate)
    return float(np.mean(seen)), float(np.mean(held))


def main() -> int:
    t0 = time.time()
    fit, val, test = stratified_split(load_corpus(TRAIN_CACHE).sessions, seed=SEED)
    idx = deployed_features(fit, verbose=False)
    powered, _ = load_or_build()
    ben = [s for s in powered if s.label == 0]
    print(f"{len(fit)} fit sessions; fresh evaluation set {len(powered)} ({len(ben)} benign)\n")

    def detect(models, thr, sessions):
        S2 = score_with(models[0], models[1], powered, idx, 10)[1]
        first = run_gate(S2, powered, thr, 4, 6)
        out = {}
        for fam in sessions:
            g = [s for s in powered if s.profile == fam]
            k = sum(first[s.sid] is not None for s in g)
            out[fam] = (k / len(g), wilson(k, len(g)))
        fp = sum(first[s.sid] is not None for s in ben) / len(ben)
        return out, fp

    results = {}
    for fam, pair in HELD.items():
        rows = {}
        for label, extra in (("no augmentation", []), ("+ swaps (held-out pair removed)",
                                                       swap_corpus(pair))):
            tr = [s for s in fit if s.profile != fam] + extra
            r = run_configuration(tr, [s for s in val if s.profile != fam], test, name=label,
                                  feature_idx=idx, seed=SEED, epochs=EPOCHS, seq_len=10, hidden=32,
                                  target_fpr=0.0001, verbose=False)
            res, fp = detect(r.models, r.threshold, [fam, "hid_injection", "mimicry", "class_exfil",
                                                     "identity_switch", "staged_delay"])
            rows[label] = {"heldout": res[fam], "fp": fp, "others": {k: v[0] for k, v in res.items()},
                           "n_swaps": len(extra)}
            if extra:
                seen_rate, held_rate = swap_detection(r.models, r.threshold, idx, pair)
                rows[label]["swap_seen_pairs"], rows[label]["swap_withheld_pair"] = seen_rate, held_rate
        results[fam] = rows

    print("HELD-OUT FAMILY detection on fresh data, trained without it (n=100; 95% CI)")
    print(f"  {'held out':16s} {'training set':36s} {'detection':>20s}  {'benign FP':>9s}  swaps")
    for fam, rows in results.items():
        for label, r in rows.items():
            d, ci = r["heldout"]
            print(f"  {fam:16s} {label:36s} {d:7.0%} [{ci[0]:3.0%}-{ci[1]:3.0%}]  {r['fp']:9.2%}  {r['n_swaps']}")
    print("\ndid the augmented model learn the swaps at all?  (fresh swap sessions, 5 per pair)")
    for fam, rows in results.items():
        r = rows["+ swaps (held-out pair removed)"]
        print(f"  hold {fam:15s} detects {r['swap_seen_pairs']:.0%} of fresh swaps of pairs it TRAINED on, "
              f"{r['swap_withheld_pair']:.0%} of the withheld pair")
    print("\ndoes augmentation cost the families it was NOT about? (detection on the other seen families)")
    for fam, rows in results.items():
        for label, r in rows.items():
            o = {k: f"{v:.0%}" for k, v in r["others"].items() if k != fam}
            print(f"  hold {fam:15s} {label:36s} " + "  ".join(f"{k[:8]}={v}" for k, v in o.items()))
    (OUT / "analysis_augment.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote out/analysis_augment.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
