"""Raw URB trace storage (storage.traces).

The requirement, from plan §5: "store raw traces, not just features, so the
feature set can be revised without re-collecting." The acceptance test is
therefore re-featurisation -- a stored session, read back and run through a
fresh extractor, must reproduce BIT FOR BIT the vectors the live monitor
actually scored. Everything else here (privacy, retention, throughput) is a
constraint on how that is achieved.
"""

import os
import threading
import time

import numpy as np
import pytest

from iudbms.capture.simulator import SimulatedSource, Simulator
from iudbms.contract import (
    FEATURE_NAMES,
    N_FEATURES,
    CaptureStats,
    URBEvent,
    feature_index,
)
from iudbms.live import StreamingDetector
from iudbms.live.monitor import LiveMonitor
from iudbms.policy import PolicyEngine
from iudbms.stage1 import Stage1
from iudbms.storage import (
    PayloadPseudonymizer,
    Store,
    StoreSource,
    TraceRecorder,
    descriptor_from_json,
    descriptor_to_json,
    refeaturise,
    session_events,
    trace_sessions,
)

PAYLOAD_FEATURES = {"payload_novelty", "repeat_ratio", "payload_seq_novelty"}


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

class GatedSource:
    """Yields a fixed event list, and says when it has finished doing so.

    The monitor reads its source on a thread. Paired with ``gated_clock`` this
    holds the first window boundary until every event is queued, so no event is
    late and the exactness comparison covers the whole trace rather than
    whatever the reader thread happened to deliver before the clock raced past.
    """

    def __init__(self, events):
        self._events = events
        self.stats = CaptureStats()
        self.done = threading.Event()

    def events(self):
        yield from self._events
        self.done.set()

    def descriptor(self, key):
        raise KeyError(key)


def gated_clock(src):
    """Whole-window ticks once the source is queued, so each boundary is reached
    in one call and the loop never sleeps."""
    t = {"v": 0.0}

    def tick():
        if not src.done.is_set():
            return 0.0
        t["v"] += 1.0
        return t["v"]
    return tick


@pytest.fixture(scope="module")
def detector_factory():
    X = np.random.default_rng(0).normal(0, 1, (300, N_FEATURES))
    s1 = Stage1(seed=0).fit(X, ["hid"] * 300)

    def make():
        return StreamingDetector(s1, None, PolicyEngine(threshold=0.9, k=4, n=6))
    return make


def simulated(profiles, seed=11, duration=20.0):
    """Several devices active at once, merged into one time-ordered stream."""
    sim = Simulator(seed=seed)
    sessions = [sim.session(f"T{i}", p, duration=duration) for i, p in enumerate(profiles)]
    events = sorted((e for s in sessions for e in SimulatedSource(s).events()),
                    key=lambda e: e.ts)
    descs = {s.descriptor.device_key: s.descriptor for s in sessions}
    return events, descs


def monitor_run(store, events, descriptor_for, detector, *, mode="pseudonymous",
                idle=30, max_events=10_000_000):
    """Run the real LiveMonitor with a recorder; return (recorder, verdicts).

    verdicts maps (device_id, t_start) -> the feature vector the detector
    actually scored for that window.
    """
    src = GatedSource(events)
    rec = TraceRecorder(store, payload_mode=mode, max_events=max_events, source="test")
    mon = LiveMonitor(detector, src, descriptor_for, idle_detach_windows=idle,
                      clock=gated_clock(src), recorder=rec)
    verdicts = {}
    for batch in mon.run():
        for v in batch:
            verdicts[(v.device_id, v.t_start)] = v.features.copy()
    store.commit()
    return rec, mon, verdicts


def assert_reproduces(store, verdicts, *, ignore=frozenset()):
    """Every stored session re-featurises to exactly what the monitor scored."""
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in ignore]
    compared = 0
    for sess in trace_sessions(store):
        r = refeaturise(store, sess["session_id"])
        assert r.complete
        for t0, x in zip(r.t, r.X):
            live = verdicts[(sess["device_id"], float(t0))]
            assert np.array_equal(live[keep], x[keep]), (
                f"{sess['device_id']} window {t0}: re-featurised vector differs from "
                f"the scored one in "
                f"{[FEATURE_NAMES[i] for i in keep if live[i] != x[i]]}")
            compared += 1
    # and nothing the monitor scored went unaccounted for
    assert compared == len(verdicts)
    return compared


# --------------------------------------------------------------------------
# 1. the pseudonymiser
# --------------------------------------------------------------------------

def test_pseudonymiser_is_a_bijection_at_every_short_length():
    """Exhaustive over all 1- and 2-byte strings: injective, length-preserving,
    invertible. Injectivity is the property the payload features need; length
    preservation is what keeps len(data) <= length true after mapping."""
    p = PayloadPseudonymizer(os.urandom(32))
    for n in (1, 2):
        domain = [i.to_bytes(n, "big") for i in range(256 ** n)]
        images = [p(x) for x in domain]
        assert all(len(y) == n for y in images)
        assert len(set(images)) == len(domain), f"collision at length {n}"
        assert all(p.invert(y) == x for x, y in zip(domain, images))


def test_pseudonymiser_round_trips_longer_payloads():
    p = PayloadPseudonymizer(os.urandom(32))
    rng = np.random.default_rng(0)
    for n in list(range(3, 41)) + [64, 512]:
        for _ in range(50):
            x = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            y = p(x)
            assert len(y) == n and p.invert(y) == x


def test_pseudonymiser_is_keyed():
    a, b = PayloadPseudonymizer(b"k" * 32), PayloadPseudonymizer(b"j" * 32)
    x = bytes([0, 0, 0x17, 0, 0, 0, 0, 0])          # an HID report: 't'
    assert a(x) == a(x)                              # consistent within a run
    assert a(x) != b(x)                              # unrelated across keys
    assert a(x) != x


# --------------------------------------------------------------------------
# 2. the acceptance test: exact re-featurisation of what the monitor scored
# --------------------------------------------------------------------------

PROFILES = ["keyboard", "mimicry", "mass_storage", "identity_switch", "mouse"]


@pytest.mark.parametrize("mode", ["pseudonymous", "full"])
def test_stored_trace_reproduces_scored_vectors_exactly(tmp_path, detector_factory, mode):
    """Five devices at once, including mimicry (payload_seq_novelty needs the
    whole session history), mass storage (bulk, truncated payloads) and
    identity_switch (an interface that appears mid-session)."""
    events, descs = simulated(PROFILES)
    with Store(tmp_path / "t.db") as store:
        rec, mon, verdicts = monitor_run(store, events, descs.get, detector_factory(),
                                         mode=mode)
        assert mon.events_late == 0 and rec.n_late == 0
        assert rec.n_recorded == len(events)
        n = assert_reproduces(store, verdicts)
        assert n > 50


def test_none_mode_reproduces_everything_but_the_payload_equality_features(
        tmp_path, detector_factory):
    """Dropping payloads loses exactly the three features that compare payload
    bytes -- and those really do change, so the claim is not vacuous."""
    events, descs = simulated(PROFILES)
    with Store(tmp_path / "t.db") as store:
        _, _, verdicts = monitor_run(store, events, descs.get, detector_factory(),
                                     mode="none")
        assert_reproduces(store, verdicts, ignore=PAYLOAD_FEATURES)
        changed = set()
        for sess in trace_sessions(store):
            r = refeaturise(store, sess["session_id"])
            for t0, x in zip(r.t, r.X):
                live = verdicts[(sess["device_id"], float(t0))]
                changed |= {n for i, n in enumerate(FEATURE_NAMES) if live[i] != x[i]}
        assert changed and changed <= PAYLOAD_FEATURES


def test_reproduction_survives_delayed_enumeration(tmp_path, detector_factory):
    """sysfs often lists a device a window after its first URBs. The monitor
    holds those events and folds them into the first scored window, ahead of
    that window's own; re-featurisation has to do the same."""
    events, descs = simulated(["keyboard", "hid_injection"])
    calls: dict[str, int] = {}

    def late_sysfs(key):
        calls[key] = calls.get(key, 0) + 1
        return descs.get(key) if calls[key] >= 3 else None

    with Store(tmp_path / "t.db") as store:
        _, _, verdicts = monitor_run(store, events, late_sysfs, detector_factory())
        assert_reproduces(store, verdicts)
        first = min(t for (_, t) in verdicts)
        assert first >= 2.0, "test did not actually defer enumeration"


def test_detach_and_reattach_are_separate_sessions(tmp_path, detector_factory):
    """A device idle long enough to be forgotten, then active again on the same
    bus address, is a new session with a fresh extractor -- and must
    re-featurise as one, or the second session inherits the first's history."""
    def e(ts, code):
        return URBEvent(ts=ts, evt="C", ttype="intr", direction="in", bus=1, dev=7,
                        ep=1, status=0, length=8, data=bytes([0, 0, code, 0, 0, 0, 0, 0]))
    evs = [e(0.1 + 0.2 * i, 4 + i % 5) for i in range(10)]
    evs += [e(20.1 + 0.2 * i, 4 + i % 5) for i in range(10)]   # same keys again
    desc = Simulator(seed=3).session("K", "keyboard").descriptor
    with Store(tmp_path / "t.db") as store:
        _, _, verdicts = monitor_run(store, evs, lambda k: desc, detector_factory(),
                                     idle=3)
        sessions = trace_sessions(store)
        assert len(sessions) == 2 and sessions[0]["t_detach"] is not None
        assert_reproduces(store, verdicts)


def test_late_events_are_kept_but_not_scored(tmp_path, detector_factory):
    """An event whose window has closed is never scored. The trace keeps it --
    it is what the kernel emitted -- flagged, and re-featurisation excludes it
    by default so it reproduces the monitor rather than the kernel."""
    desc = Simulator(seed=3).session("K", "keyboard").descriptor

    def e(ts):
        return URBEvent(ts=ts, evt="C", ttype="intr", direction="in", bus=1, dev=7,
                        ep=1, status=0, length=8, data=b"\x01" * 8)

    class Straggler(GatedSource):
        def events(self):
            yield e(0.1)
            yield e(3.2)
            self.done.set()
            time.sleep(0.2)          # let the loop close windows 0..2
            yield e(0.5)             # belongs to window 0: late

    src = Straggler([])
    with Store(tmp_path / "t.db") as store:
        rec = TraceRecorder(store, source="test")
        mon = LiveMonitor(detector_factory(), src, lambda k: desc,
                          clock=gated_clock(src), recorder=rec)
        list(mon.run())
        store.commit()
        assert mon.events_late == 1 and rec.n_late == 1
        sid = trace_sessions(store)[0]["session_id"]
        _, scored, _ = session_events(store, sid)
        _, emitted, _ = session_events(store, sid, include_late=True)
        assert len(emitted) == len(scored) + 1


# --------------------------------------------------------------------------
# 3. privacy
# --------------------------------------------------------------------------

def _db_bytes(path):
    out = b""
    for suffix in ("", "-wal", "-shm"):
        p = path.with_name(path.name + suffix)
        if p.exists():
            out += p.read_bytes()
    return out


SECRET = b"Hunter2!"            # 8 bytes, the length of an HID report


@pytest.mark.parametrize("mode,present", [("pseudonymous", False),
                                          ("none", False),
                                          ("full", True)])
def test_payload_bytes_on_disk_per_mode(tmp_path, mode, present):
    """Scans the raw database files -- including the WAL, where fresh writes sit
    -- for a known payload. The 'full' case proves the scan can find it."""
    ev = URBEvent(ts=0.5, evt="C", ttype="intr", direction="in", bus=1, dev=4, ep=1,
                  status=0, length=8, data=SECRET)
    db = tmp_path / "t.db"
    with Store(db) as store:
        rec = TraceRecorder(store, payload_mode=mode)
        rec.record([ev] * 50)
        store.commit()
        assert (SECRET in _db_bytes(db)) is present
    assert (SECRET in _db_bytes(db)) is present


def test_pseudonymised_events_are_still_valid_events(tmp_path, detector_factory):
    """Length preservation keeps every contract invariant -- in particular
    len(data) <= length, which a hash-sized pseudonym would break."""
    events, descs = simulated(["keyboard", "mass_storage", "webcam"])
    with Store(tmp_path / "t.db") as store:
        monitor_run(store, events, descs.get, detector_factory())
        for sess in trace_sessions(store):
            _, evs, _ = session_events(store, sess["session_id"])
            for e in evs:
                e.validate()


def test_recorder_never_persists_its_key(tmp_path):
    key = os.urandom(32)
    db = tmp_path / "t.db"
    with Store(db) as store:
        rec = TraceRecorder(store, key=key)
        rec.record([URBEvent(ts=0.1, evt="C", ttype="intr", direction="in", bus=1,
                             dev=4, ep=1, status=0, length=8, data=SECRET)])
        store.commit()
    assert key not in _db_bytes(db)


def test_unknown_payload_mode_is_rejected(tmp_path):
    with Store(tmp_path / "t.db") as store, pytest.raises(ValueError):
        TraceRecorder(store, payload_mode="plaintext")


# --------------------------------------------------------------------------
# 4. retention
# --------------------------------------------------------------------------

def _burst(dev, t0, n, step=0.01):
    return [URBEvent(ts=t0 + i * step, evt="C", ttype="intr", direction="in", bus=1,
                     dev=dev, ep=1, status=0, length=8, data=bytes([0, 0, 4 + i % 30, 0, 0, 0, 0, 0]))
            for i in range(n)]


def test_retention_is_bounded_and_spares_pinned_evidence(tmp_path):
    desc = Simulator(seed=3).session("K", "keyboard").descriptor
    with Store(tmp_path / "t.db") as store:
        rec = TraceRecorder(store, max_events=300, evict_every=50)
        rec.on_attach("1-4", 0.0, desc)
        rec.record(_burst(4, 0.0, 200))
        pinned = rec.pin("1-4", 0.0, 0.5)               # the first 50 events
        assert pinned == 50
        rec.record(_burst(4, 2.0, 400))
        rec.evict()
        store.commit()
        total = store.cx.execute("SELECT COUNT(*) FROM urb_events").fetchone()[0]
        kept_pin = store.cx.execute(
            "SELECT COUNT(*) FROM urb_events WHERE pinned=1").fetchone()[0]
        assert total <= 300
        assert kept_pin == 50, "alert evidence was evicted"
        assert rec.n_evicted == 600 - total


def test_eviction_marks_sessions_incomplete(tmp_path):
    """payload_seq_novelty depends on the whole session, so a session that lost
    rows cannot be re-featurised exactly -- and must say so."""
    desc = Simulator(seed=3).session("K", "keyboard").descriptor
    with Store(tmp_path / "t.db") as store:
        rec = TraceRecorder(store, max_events=100, evict_every=10_000)
        rec.on_attach("1-4", 0.0, desc)
        rec.record(_burst(4, 0.0, 150))
        rec.on_detach("1-4", 2.0)
        rec.on_attach("1-4", 2.0, desc)                 # same address, new device
        rec.record(_burst(4, 2.5, 40))
        rec.close(4.0)
        store.commit()
        first, second = trace_sessions(store)
        assert refeaturise(store, first["session_id"]).complete is False
        assert refeaturise(store, second["session_id"]).complete is True


# --------------------------------------------------------------------------
# 5. throughput, and the stored run as a capture source
# --------------------------------------------------------------------------

def test_recording_clears_the_capture_throughput_floor(tmp_path):
    """Plan §4.1: >= 20k events/s. Recording must not be the bottleneck."""
    events = _burst(4, 0.0, 60_000, step=0.0001)
    with Store(tmp_path / "t.db") as store:
        rec = TraceRecorder(store)
        t0 = time.perf_counter()
        for i in range(0, len(events), 2000):
            rec.record(events[i:i + 2000])
        store.commit()
        rate = len(events) / (time.perf_counter() - t0)
    assert rate >= 20_000, f"recorded {rate:,.0f} events/s"


def test_a_stored_run_replays_through_the_monitor_like_a_capture(
        tmp_path, detector_factory):
    """StoreSource is a capture backend: a pseudonymised trace, replayed through
    a fresh monitor, is scored identically to the original traffic."""
    events, descs = simulated(["keyboard", "mimicry", "mass_storage"])
    with Store(tmp_path / "t.db") as store:
        rec, _, original = monitor_run(store, events, descs.get, detector_factory())

    class Threaded(GatedSource):
        """Iterates the StoreSource INSIDE the monitor's reader thread, which is
        where a real --replay-db run reads it."""
        def __init__(self, inner):
            super().__init__([])
            self.inner = inner

        def events(self):
            yield from self.inner.events()
            self.done.set()

    with Store(tmp_path / "t.db") as store, Store(tmp_path / "t2.db") as store2:
        src = StoreSource(store, rec.run_id)
        gate = Threaded(src)
        rec2 = TraceRecorder(store2, source="replay")
        mon = LiveMonitor(detector_factory(), gate, src.descriptor,
                          clock=gated_clock(gate), recorder=rec2)
        again = {(v.device_id, v.t_start): v.features.copy()
                 for b in mon.run() for v in b}
    assert again.keys() == original.keys()
    for k in original:
        assert np.array_equal(original[k], again[k]), f"window {k} scored differently"


def test_descriptor_json_round_trip_keeps_late_interfaces():
    d = Simulator(seed=5).session("X", "identity_switch").descriptor
    assert d.has_late_interface()
    assert descriptor_from_json(descriptor_to_json(d)) == d
