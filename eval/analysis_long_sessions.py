"""
Long sessions -- does the deployed model hold up past the 40-85 s it was trained on?

    python3 -m eval.analysis_long_sessions [--models out/models] [--n 12] [--quick]

Every number reported so far comes from sessions of 40-85 s. A deployed monitor
watches a keyboard for a working day. Two things can go wrong with length that a
short corpus cannot show, and this experiment measures both:

1. **Benign drift.** ``payload_seq_novelty`` remembers every report transition seen
   in the session, and language repeats its transitions, so on an hour of genuine
   typing it decays toward the "replayed script" signature. A per-device baseline
   also adapts for longer than it ever did in training. The question is the false
   alarm rate *per device-hour*, the unit an operator actually pays in.
2. **Late attacks.** A device that behaves for twenty minutes and then injects is
   the realistic BadUSB (a timed payload). After a long benign prefix the
   transition memory is full and the baseline is settled -- does that hide the
   attack or expose it?

Benign classes use the same realistic simulator and duty-cycle mix as Corpus B
(``eval.analysis_realistic``), so the only thing that changes is length. Sessions
are scored offline, which ``test_streaming_matches_offline_scoring`` pins as
identical to the streaming detector.

Writes out/analysis_long_sessions.json. Reads a model set; trains nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iudbms.capture.simulator import SimulatedSession

from .analysis_common import MODELS, OUT, deployed, featurize, run_gate, score_with, wilson
from .analysis_idle_retrain import IDLE_MIX
from .analysis_realistic import RealisticSim

BENIGN = ("keyboard", "mouse", "mass_storage", "webcam", "audio")
DURATIONS = (60.0, 600.0, 1800.0, 3600.0)
#: keyboard-descriptor attacks whose generator can be started at any time
LATE_ATTACKS = ("hid_injection", "mimicry", "mimicry_adaptive", "class_exfil")
ONSETS = (30.0, 300.0, 1200.0)
ATTACK_TAIL = 120.0          # how long the attack runs after onset
SEED = 31


def _duty(sim: RealisticSim, rng: np.random.Generator) -> str:
    probs = np.array([m[1] for m in IDLE_MIX])
    name, _, kw = IDLE_MIX[int(rng.choice(len(IDLE_MIX), p=probs))]
    sim._idle, sim._active, sim._fully_idle = kw["idle"], kw["active"], kw["fully_idle"]
    return name


def late_session(sim: RealisticSim, sid: str, profile: str, onset: float,
                 tail: float = ATTACK_TAIL) -> SimulatedSession:
    """A keyboard that is used normally for ``onset`` seconds, then attacks.

    Mirrors ``Simulator.session`` for the keyboard-descriptor attacks, with the
    benign prefix lengthened; the prefix uses the duty-cycled human generator, so
    the device has gone idle and come back the way a real keyboard does.
    """
    sim._pending, sim._prev_key, sim._queue = {}, 0, []
    sim._persona = sim._sample_persona()
    bus, dev = sim.rng.randint(1, 3), sim._devnum
    sim._devnum = 2 + (sim._devnum - 1) % 120
    lines: list[str] = []
    d = sim._descriptor("keyboard", bus, dev,
                        [sim._iface(0x03, 0x01, 0x01, sim._HID_EPS)])
    t = sim._enumerate(lines, bus, dev, d.vid, d.pid)
    t = sim._hid_class_setup(lines, t, bus, dev)
    sim._human_typing(t, onset - t, bus, dev, lines)
    dur = onset + tail
    gen = {"hid_injection": sim._injected_typing, "mimicry": sim._mimicry_typing,
           "mimicry_adaptive": sim._adaptive_typing, "class_exfil": sim._covert_bulk_out}[profile]
    gen(onset, tail, bus, dev, lines)
    for (b, dv), _ in list(sim._pending.items()):
        sim._flush(lines, b, dv, up_to=float("inf"))
    lines.sort(key=lambda ln: int(ln.split()[1]))
    return SimulatedSession(sid=sid, profile=profile, label=1, descriptor=d,
                            lines=lines, onset=onset, duration=dur)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--n", type=int, default=12, help="sessions per (class, duration) cell")
    ap.add_argument("--quick", action="store_true", help="durations up to 600 s only")
    ap.add_argument("--out", type=Path, default=OUT / "analysis_long_sessions.json")
    a = ap.parse_args(argv)

    cfg, s1, lstm = deployed(a.models)
    idx, thr, K, N = np.array(cfg["feature_idx"]), cfg["threshold"], cfg["k"], cfg["n"]
    durations = [d for d in DURATIONS if d <= 600] if a.quick else DURATIONS
    print(f"model {a.models}  threshold {thr:.4f}  gate {K}-of-{N}  n={a.n} per cell")
    t_start = time.time()

    # ---------------- benign: false alarms vs session length ----------------
    benign: dict = {}
    print(f"\nBENIGN  {'class':14s}" + "".join(f"{int(d):>16d}s" for d in durations)
          + "   alarms/device-h")
    for prof in BENIGN:
        sim, rng = RealisticSim(seed=SEED), np.random.default_rng(SEED)
        row, hours, alarms = [], 0.0, 0
        for dur in durations:
            sess, cond = [], {}
            for i in range(a.n):
                sid = f"L-{prof}-{int(dur)}-{i}"
                cond[sid] = _duty(sim, rng)
                sess.append(featurize(sim.session(sid, prof, duration=dur), sid, prof))
            _, S2, _ = score_with(s1, lstm, sess, idx, cfg["seq_len"])
            first = run_gate(S2, sess, thr, K, N)
            k = sum(first[s.sid] is not None for s in sess)
            over = np.concatenate([S2[s.sid] for s in sess]) >= thr
            # windows over threshold, by position in the session (drift check)
            pos = np.concatenate([np.arange(len(S2[s.sid])) for s in sess])
            late = over[pos >= 600].mean() if (pos >= 600).any() else None
            early = over[pos < 600].mean()
            hours += a.n * dur / 3600.0
            alarms += k
            benign.setdefault(prof, {})[str(int(dur))] = {
                "n": a.n, "fp_sessions": k, "fp": k / a.n, "ci": wilson(k, a.n),
                "windows_over": float(over.mean()), "windows_over_first_600s": float(early),
                "windows_over_after_600s": None if late is None else float(late),
                "first_alarm_s": sorted(first[s.sid] for s in sess if first[s.sid] is not None),
                "duty": {c: sum(v == c for v in cond.values()) for c in set(cond.values())},
                "median_score_by_10min": [float(np.median(np.concatenate(
                    [S2[s.sid][j * 600:(j + 1) * 600] for s in sess if len(S2[s.sid]) > j * 600])))
                    for j in range(int(dur // 600) or 1)],
            }
            row.append(f"{k:3d}/{a.n:<3d}{'':3s}w{over.mean():6.4f}")
        benign[prof]["device_hours"] = hours
        benign[prof]["alarms"] = alarms
        lo, hi = wilson(alarms, max(1, int(round(hours))))
        benign[prof]["alarms_per_device_hour"] = alarms / hours
        print(f"        {prof:14s}" + "".join(f"{c:>17s}" for c in row)
              + f"   {alarms / hours:6.3f}  ({hours:.1f} h)")

    tot_h = sum(benign[p]["device_hours"] for p in BENIGN)
    tot_a = sum(benign[p]["alarms"] for p in BENIGN)
    print(f"        all classes: {tot_a} sessions alarmed over {tot_h:.1f} device-hours "
          f"= {tot_a / tot_h:.3f} alarms per device-hour")

    # ---------------- late-onset attacks: detection vs onset ----------------
    late: dict = {}
    onsets = [o for o in ONSETS if o <= max(durations)]
    print(f"\nLATE ATTACKS  detection (median latency) by onset, n={a.n} per cell")
    print(f"        {'family':18s}" + "".join(f"{int(o):>18d}s" for o in onsets))
    for prof in LATE_ATTACKS:
        sim, rng = RealisticSim(seed=SEED + 1), np.random.default_rng(SEED + 1)
        row = []
        for onset in onsets:
            sess = []
            for i in range(a.n):
                sid = f"X-{prof}-{int(onset)}-{i}"
                _duty(sim, rng)
                sess.append(featurize(late_session(sim, sid, prof, onset), sid, prof))
            _, S2, _ = score_with(s1, lstm, sess, idx, cfg["seq_len"])
            first = run_gate(S2, sess, thr, K, N)
            pre = sum(first[s.sid] is not None and first[s.sid] <= s.onset for s in sess)
            hit = [s for s in sess if first[s.sid] is not None and first[s.sid] > s.onset]
            lat = [first[s.sid] - s.onset for s in hit]
            late.setdefault(prof, {})[str(int(onset))] = {
                "n": a.n, "detected": len(hit), "rate": len(hit) / a.n, "ci": wilson(len(hit), a.n),
                "alarmed_before_onset": pre,
                "median_latency_s": float(np.median(lat)) if lat else None}
            ml = f"{np.median(lat):5.1f}s" if lat else "   -  "
            row.append(f"{len(hit):3d}/{a.n:<3d} ({ml})")
        print(f"        {prof:18s}" + "".join(f"{c:>19s}" for c in row))

    out = {"models": str(a.models), "threshold": thr, "gate": [K, N], "n_per_cell": a.n,
           "durations_s": list(durations), "benign": benign, "late_attacks": late,
           "total_device_hours": tot_h, "total_alarms": tot_a,
           "alarms_per_device_hour": tot_a / tot_h, "runtime_s": time.time() - t_start}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {a.out}  ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
