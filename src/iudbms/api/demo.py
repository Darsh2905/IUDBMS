"""
Live demo endpoints: stream the DEPLOYED pipeline's verdicts, one window at a time.

    GET /demo                       the demo page
    GET /demo/scenarios             what can be plugged in
    GET /demo/stream?scenario=ID    server-sent events, one per 1 s window

Nothing here scores anything itself. Each stream builds the same
``StreamingDetector`` the systemd monitor builds (``iudbms.live.cli``): the
deployed Stage 1 and Stage 2, the calibrated threshold, the 4-of-6 gate, the model
support list and the human-capability rule, and feeds it one window of URBs at a time.
``tests/test_live_detector.py`` pins its scores to the offline ones.

What differs from a deployment is only the source of the URBs: a simulated session
(``eval.analysis_realistic.RealisticSim``, the generator the deployed model was trained
on) or a window of a recorded real-kernel capture from the testbed (``out/testbed/raw``)
if that is on disk. Enforcement stays dry-run; the page says so.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Iterator

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from ..capture.simulator import SimulatedSource
from ..capture.usbmon import UsbmonSource
from ..contract import FEATURE_NAMES
from ..features.extractor import window_events
from ..live import StreamingDetector
from ..live.cli import _load_descriptor_snapshot, load_models
from ..policy import ModelSupport, PolicyEngine

router = APIRouter()

PAGE = Path(str(files("iudbms.api").joinpath("static", "demo.html")))
REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "out" / "testbed" / "raw"

#: Claim shown beside each scenario, written before looking at its result.
#: ``expect`` is what the measured results say should happen, so a surprise on stage
#: is visible as one rather than explained away afterwards.
SIM = [
    # id, label, kind, profile, expect, note
    ("kbd", "Person typing", "benign", "keyboard", "quiet",
     "A keyboard with a human's rhythm: bursts, pauses, key releases."),
    ("mouse", "Mouse", "benign", "mouse", "quiet", "Pointer motion. Never a keystroke source."),
    ("drive", "USB drive", "benign", "mass_storage", "quiet", "Bulk transfers; another device class."),
    ("inject", "Keystroke injector", "attack", "hid_injection", "alarm",
     "A BadUSB that types a payload at machine speed, after the user has been typing normally."),
    ("staged", "Staged injector", "attack", "staged_delay", "alarm",
     "Behaves like a keyboard, waits, then injects."),
    ("switch", "Identity switch", "attack", "identity_switch", "alarm",
     "Enumerates as one thing, then acts as another."),
    ("exfil", "Storage exfiltration", "attack", "class_exfil", "alarm",
     "A keyboard that also moves bulk data."),
    ("replay", "Replay loop at human speed", "attack", "mimicry", "partial",
     "Types a short script over and over at human speed. Caught in simulation, missed on real kernels."),
    ("slow", "Payload typed once at human speed", "attack", "mimicry_adaptive", "miss",
     "The case nothing here detects. Shown so the limit is visible."),
]


@dataclass
class Scenario:
    id: str
    label: str
    kind: str          # benign | attack
    expect: str        # quiet | alarm | partial | miss
    note: str
    source: str        # simulated | real kernel
    profile: str | None = None
    run: str | None = None
    dev: str | None = None


def _real_scenarios() -> list[Scenario]:
    """A few real-kernel devices, if the testbed captures are on disk."""
    wanted = [("debian-13-benign-s1", "1-2", "Real kernel: keyboard, human typing replay", "benign", "quiet",
               "QEMU keyboard on Debian 13 (kernel 6.12) driven by recorded keystroke timing of real people."),
              ("debian-13-attack-s1", "1-11", "Real kernel: injector", "attack", "alarm",
               "Keystroke injection through a real xHCI controller and the real HID driver."),
              ("debian-13-attack-s1", "1-13", "Real kernel: replay loop", "attack", "miss",
               "Human-speed replay on a real kernel. The learned model does not catch it."),
              ("debian-13-benign-s1", "1-7", "Real kernel: network adapter", "benign", "quiet",
               "A class the model never saw. Scored but never escalated.")]
    out = []
    for run, dev, label, kind, expect, note in wanted:
        d = RAW / run
        if (d / "usbmon.txt").exists() and (d / "descriptors.json").exists():
            out.append(Scenario(f"real-{run}-{dev}", label, kind, expect, note, "real kernel",
                                run=run, dev=dev))
    return out


@lru_cache(maxsize=1)
def scenarios() -> dict[str, Scenario]:
    s = [Scenario(i, l, k, e, n, "simulated", profile=p) for i, l, k, p, e, n in SIM]
    return {x.id: x for x in s + _real_scenarios()}


# ---------------------------------------------------------------------------

_LOAD = threading.Lock()


def models_dir() -> Path:
    return Path(os.environ.get("IUDBMS_MODELS", REPO / "out" / "models"))


@lru_cache(maxsize=2)
def _models(path: str):
    return load_models(Path(path))


def _detector() -> tuple[StreamingDetector, dict]:
    with _LOAD:
        cfg, s1, s2 = _models(str(models_dir()))
    support = ModelSupport(cfg["support"]) if cfg.get("support") else None
    det = StreamingDetector(
        s1, s2, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"], support=support),
        feature_idx=np.array(cfg["feature_idx"]), seq_len=cfg["seq_len"],
        human_max_rps=(cfg.get("capability_rule") or {}).get("human_max_rps"))
    return det, cfg


def _sim_session(sc: Scenario, seed: int):
    """(descriptor, events, duration, onset). Attacks start after a normal prefix."""
    if str(REPO) not in sys.path:               # eval/ is a repo directory, not the package
        sys.path.insert(0, str(REPO))
    try:
        from eval.analysis_long_sessions import LATE_ATTACKS as LATE
        from eval.analysis_long_sessions import _duty, late_session
        from eval.analysis_realistic import RealisticSim
    except ImportError as e:                    # an installed wheel without the checkout
        raise HTTPException(503, f"simulated scenarios need the repository's eval/ package: {e}")
    sim = RealisticSim(seed=seed)
    _duty(sim, np.random.default_rng(seed))
    sid = f"DEMO-{seed}"
    if sc.kind == "attack" and sc.profile in LATE:
        sess = late_session(sim, sid, sc.profile, onset=25.0, tail=45.0)
    else:                                       # staged_delay / identity_switch set their own onset
        sess = sim.session(sid, sc.profile, duration=70.0 if sc.kind == "benign" else None)
    ev = list(SimulatedSource(sess).events())
    return sess.descriptor, ev, sess.duration, sess.onset


@lru_cache(maxsize=4)
def _real_events(run: str):
    cap = RAW / run / "usbmon.txt"
    by_dev: dict[str, list] = {}
    for e in UsbmonSource(path=cap).events():
        by_dev.setdefault(e.device_key, []).append(e)
    return by_dev, _load_descriptor_snapshot(RAW / run / "descriptors.json")


def _real_session(sc: Scenario):
    by_dev, descs = _real_events(sc.run)
    ev, desc = by_dev.get(sc.dev, []), descs.get(sc.dev)
    if not ev or desc is None:
        raise HTTPException(404, f"{sc.dev} is not in {sc.run}")
    first = next((e.ts for e in ev if e.evt == "C" and e.ttype == "intr"
                  and len(e.data) == 8 and any(e.data[2:])), None)
    # Start a little before the first keystroke so the quiet part is visible.
    t0 = max(0.0, (first if first is not None else ev[0].ts) - 15.0)
    t0 = float(int(t0))
    seg = [e for e in ev if t0 <= e.ts < t0 + 75.0]
    labels = json.loads((RAW / sc.run / "labels.json").read_text())
    attack = labels.get(sc.dev, {}).get("label") == 1
    onset = first if attack else None
    return desc, seg, t0, 75.0, onset


# ---------------------------------------------------------------------------

def _verdict_event(v, cfg, n_events: int) -> dict:
    a = v.action
    top = sorted(v.evidence, key=lambda kv: -abs(kv[1]))[:3]
    d = {"t": float(v.t_start), "s1": float(v.stage1), "s2": float(v.stage2),
         "hits": v.gate_hits, "n": v.gate_n, "state": v.trust_state,
         "events": int(n_events), "over": bool(v.stage2 >= cfg["threshold"]),
         "evidence": [[k, float(x)] for k, x in top], "action": None}
    if a is not None:
        d["action"] = {"name": a.action, "severity": a.severity, "ts": float(a.ts),
                       "fired_by": (a.rule or {}).get("fired_by"), "rule": a.rule}
    return d


def _stream(sc: Scenario, speed: float, seed: int) -> Iterator[str]:
    def sse(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    det, cfg = _detector()
    if sc.source == "simulated":
        desc, events, t0, duration, onset = None, None, 0.0, 0.0, None
        desc, events, duration, onset = _sim_session(sc, seed)
    else:
        desc, events, t0, duration, onset = _real_session(sc)

    did = desc.device_key
    det.attach(desc, did)
    ifs = [{"cls": i.bInterfaceClass, "sub": i.bInterfaceSubClass, "proto": i.bInterfaceProtocol,
            "pages": list(i.hid_usage_pages) if i.hid_usage_pages is not None else None}
           for i in desc.interfaces]
    yield sse("init", {
        "scenario": sc.id, "label": sc.label, "kind": sc.kind, "expect": sc.expect,
        "note": sc.note, "source": sc.source, "device": did,
        "product": desc.product, "vid": f"{desc.vid:04x}", "pid": f"{desc.pid:04x}",
        "interfaces": ifs, "threshold": cfg["threshold"], "k": cfg["k"], "n": cfg["n"],
        "features": len(cfg["feature_idx"]), "windows": int(duration),
        "onset": None if onset is None else float(onset) - (t0 if sc.source != "simulated" else 0.0),
        "human_max_rps": (cfg.get("capability_rule") or {}).get("human_max_rps"),
        "dry_run": True, "feature_names": list(FEATURE_NAMES)})

    start = time.monotonic()
    first_alarm = None
    last_state = "observe"
    if sc.source == "simulated":
        buckets = window_events(events, duration=duration)
    else:                                       # absolute timestamps, windows from t0
        n = int(duration)
        idx: list[list] = [[] for _ in range(n)]
        for e in events:
            k = int(e.ts - t0)
            if 0 <= k < n:
                idx[k].append(e)
        buckets = ((t0 + i, b) for i, b in enumerate(idx))

    for i, (t_start, bucket) in enumerate(buckets):
        delay = start + (i + 1) / max(speed, 0.05) - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, 2.0))
        v = det.observe_window(did, t_start, bucket)
        out = _verdict_event(v, cfg, sum(1 for e in bucket if e.evt in ("C", "E")))
        out["i"] = i
        out["t"] = float(i)                     # seconds into the stream, not the capture
        last_state = v.trust_state
        if v.action is not None and first_alarm is None:
            first_alarm = i + 1.0
        yield sse("window", out)

    rel_onset = None if onset is None else (float(onset) - (t0 if sc.source != "simulated" else 0.0))
    yield sse("end", {
        "alarmed": first_alarm is not None, "first_alarm": first_alarm, "onset": rel_onset,
        "latency": None if (first_alarm is None or rel_onset is None) else first_alarm - rel_onset,
        "final_state": last_state, "kind": sc.kind, "expect": sc.expect})


# ---------------------------------------------------------------------------

@router.get("/demo", include_in_schema=False)
def demo_page():
    if not PAGE.exists():
        raise HTTPException(404, f"demo page not found at {PAGE}")
    return FileResponse(PAGE)


@router.get("/demo/scenarios")
def demo_scenarios() -> list[dict]:
    return [{"id": s.id, "label": s.label, "kind": s.kind, "expect": s.expect,
             "note": s.note, "source": s.source} for s in scenarios().values()]


@router.get("/demo/stream")
def demo_stream(scenario: str, speed: float = Query(2.0, gt=0, le=60), seed: int = Query(11, ge=0)):
    sc = scenarios().get(scenario)
    if sc is None:
        raise HTTPException(404, f"unknown scenario {scenario!r}")
    try:
        _detector()                             # fail with a clear message, not mid-stream
    except SystemExit as e:
        raise HTTPException(503, str(e))
    return StreamingResponse(_stream(sc, speed, seed), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
