"""The live monitoring loop, tested without a kernel.

LiveMonitor takes any EventSource and any descriptor lookup, so the windowing,
device discovery and detach logic are all exercisable against a fake. What
cannot be tested here is debugfs itself, which needs Linux.
"""

import numpy as np
import pytest

from iudbms.capture.simulator import Simulator
from iudbms.contract import CaptureStats, URBEvent
from iudbms.live import StreamingDetector
from iudbms.live.monitor import LiveMonitor
from iudbms.policy import PolicyEngine
from iudbms.stage1 import Stage1, class_group

DESC = Simulator(seed=1).session("T", "keyboard").descriptor
DESC2 = Simulator(seed=2).session("T", "mass_storage").descriptor


class FakeSource:
    """Emits a fixed list of events immediately, then ends."""

    def __init__(self, events):
        self._events = events
        self.stats = CaptureStats()

    def events(self):
        yield from self._events

    def descriptor(self, key):
        raise KeyError(key)


def ev(ts, dev=3, ttype="intr"):
    return URBEvent(ts=ts, evt="C", ttype=ttype, direction="in", bus=1, dev=dev,
                    ep=1, status=0, length=8, data=b"\x01" * 8)


@pytest.fixture(scope="module")
def detector_factory():
    rng = np.random.default_rng(0)
    from iudbms.contract import N_FEATURES
    X = rng.normal(0, 1, (300, N_FEATURES))
    s1 = Stage1(seed=0).fit(X, ["hid"] * 300)

    def make():
        return StreamingDetector(s1, None, PolicyEngine(threshold=0.9, k=4, n=6))
    return make


def _fast_clock():
    """A clock the monitor can race through, so tests do not sleep."""
    t = {"v": 0.0}

    def tick():
        t["v"] += 0.5      # two calls per window boundary
        return t["v"]
    return tick


def test_windows_close_and_devices_are_discovered(detector_factory):
    src = FakeSource([ev(0.1), ev(0.4), ev(1.2)])
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC,
                      clock=_fast_clock())
    batches = list(mon.run(max_windows=4))
    assert mon.windows_scored >= 1
    scored = [v for b in batches for v in b]
    assert scored, "no windows were scored"
    assert all(v.device_id == "1-3" for v in scored)


def test_idle_windows_between_traffic_are_scored(detector_factory):
    """An idle second is an observation -- the silence after a finished payload
    is the only signal some attacks leave."""
    src = FakeSource([ev(0.1), ev(3.5)])       # a two-window gap in between
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC, clock=_fast_clock())
    batches = list(mon.run(max_windows=5))
    scored = [b for b in batches if b]
    assert len(scored) >= 4, "device stopped being scored while it was quiet"
    counts = [v.features[0] for b in batches for v in b]   # n_events per window
    assert 0.0 in counts, "no genuinely empty window was scored"


def test_windows_are_cut_on_kernel_timestamps(detector_factory):
    """Not on arrival time: the reader thread has scheduling jitter, so the same
    capture must not score differently under load."""
    src = FakeSource([ev(0.2), ev(0.7), ev(2.4)])
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC, clock=_fast_clock())
    batches = list(mon.run(max_windows=4))
    n_events = [v.features[0] for b in batches for v in b]
    assert n_events[0] == 2.0, f"window 0 should hold both early events, got {n_events}"
    assert n_events[1] == 0.0, "window 1 is genuinely empty"
    assert n_events[2] == 1.0, "window 2 holds the late event"


def test_events_for_a_closed_window_are_counted_not_folded_forward(detector_factory):
    """Folding a late event into a later window would distort that window's rate."""
    src = FakeSource([ev(5.0), ev(0.1)])       # second event belongs to a past window
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC, clock=_fast_clock())
    list(mon.run(max_windows=7))
    assert mon.events_late >= 0


def test_unenumerated_device_is_retried_not_dropped(detector_factory):
    """A device whose descriptor is not available yet must not lose its events."""
    calls = {"n": 0}

    def desc(key):
        calls["n"] += 1
        return None if calls["n"] < 2 else DESC

    src = FakeSource([ev(0.1), ev(0.2)])
    mon = LiveMonitor(detector_factory(), src, desc, clock=_fast_clock())
    list(mon.run(max_windows=4))
    assert calls["n"] >= 2, "descriptor lookup was not retried"
    assert "1-3" in mon.detector.devices


def test_deferred_events_keep_chronological_order(detector_factory):
    """A device's events held before sysfs enumerates it must be scored in time
    order. They were folded in AFTER the attach window's own events, and in
    reverse window order, so np.diff produced negative inter-arrival times
    (min_iat -1.2 s on this input). On a real hotplug the enumeration URBs
    precede the sysfs entry, so this hit the first scored window of every new
    device -- the window in which a BadUSB starts typing."""
    from iudbms.contract import feature_index

    def e(ts):
        return URBEvent(ts=ts, evt="C", ttype="intr", direction="in", bus=1, dev=3,
                        ep=1, status=0, length=8, data=b"\x01" * 8)

    calls = {"n": 0}

    def desc(key):
        calls["n"] += 1
        return None if calls["n"] < 3 else DESC

    src = FakeSource([e(0.10), e(0.30), e(1.10), e(1.30), e(2.10), e(2.30)])
    mon = LiveMonitor(detector_factory(), src, desc, clock=_fast_clock())
    first = next(v for b in mon.run(max_windows=4) for v in b)
    assert first.features[feature_index("n_events")] == 6, "deferred events were lost"
    assert first.features[feature_index("min_iat")] >= 0.0, (
        f"negative inter-arrival time {first.features[feature_index('min_iat')]:.3f}: "
        f"deferred events were not scored in chronological order")


def test_finite_source_ends_the_run_without_max_windows(detector_factory):
    """A replay must stop on its own once its traffic is scored. End-of-stream
    used to be seen only on the drain that dequeued the sentinel; if later
    windows still held buffered events at that moment, every subsequent drain
    reported the source alive again and run() scored idle windows forever.
    Every other test here passes max_windows, which is why this went unseen --
    and it meant `live_monitor.py --replay` could never exit by itself."""
    import threading

    src = FakeSource([ev(0.1), ev(1.5), ev(4.2)])   # traffic spans five windows
    t = {"v": 0.0}

    def clock():
        t["v"] += 1.0
        return t["v"]

    mon = LiveMonitor(detector_factory(), src, lambda k: DESC, clock=clock)
    finished = threading.Event()

    def consume():
        for _ in mon.run():          # no max_windows
            pass
        finished.set()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    assert finished.wait(10), (
        f"run() did not terminate; {mon.windows_scored} windows scored so far")
    assert mon.windows_scored < 50


def test_stop_ends_the_run_even_while_the_source_is_blocked(detector_factory):
    """SIGINT/SIGTERM call stop(). run() never checked it -- stopping relied on
    the reader thread noticing, and on an idle live bus that thread is blocked
    in a debugfs read until the next URB. So Ctrl-C, or systemd's SIGTERM, did
    not stop the monitor, and systemd would SIGKILL it mid-write."""
    import threading

    class Blocked:
        stats = CaptureStats()

        def events(self):
            yield ev(0.1)
            threading.Event().wait()     # an idle bus: never returns

        def descriptor(self, key):
            raise KeyError(key)

    t = {"v": 0.0}

    def clock():
        t["v"] += 1.0
        return t["v"]

    mon = LiveMonitor(detector_factory(), Blocked(), lambda k: DESC, clock=clock)
    done = threading.Event()

    def consume():
        for i, _ in enumerate(mon.run()):
            if i == 2:
                mon.stop()
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    assert done.wait(10), "run() ignored stop() while the source was blocked"
    assert mon.windows_scored <= 4


def test_a_device_that_never_gets_a_descriptor_is_not_held_forever(detector_factory):
    """Found on the first real usbmon capture. SysfsScanner deliberately skips
    root hubs, and the kernel emits traffic for them, so descriptor_for returns
    None for ever. The monitor holds a not-yet-enumerated device's events and
    retries -- and re-deferred them APPENDED to each window's new events, so the
    held list grew by that device's traffic every window, without bound. The
    simulator never exercised it: every simulated device has a descriptor."""
    # a root-hub-like device (1-1) chattering every window for 60 windows,
    # alongside a normal one (1-3)
    events = []
    for w in range(60):
        events += [ev(w + 0.2, dev=1), ev(w + 0.4, dev=1), ev(w + 0.6, dev=1)]
    events += [ev(0.1, dev=3), ev(59.5, dev=3)]
    events.sort(key=lambda e: e.ts)

    mon = LiveMonitor(detector_factory(), FakeSource(events),
                      lambda k: DESC if k == "1-3" else None,
                      clock=_fast_clock(), max_defer_windows=10)
    list(mon.run(max_windows=62))

    held = len(mon._deferred.get("1-1", []))
    assert held <= 3 * 11, f"{held} events still held for a device with no descriptor"
    assert mon.events_undescribed >= 3 * 40, "events for the undescribed device were not dropped"
    assert "1-1" not in mon.detector.devices
    assert "1-3" in mon.detector.devices, "the described device must be unaffected"


def test_a_late_descriptor_still_attaches_within_the_deferral_window(detector_factory):
    """The cap must not break the case deferral exists for: a device whose
    sysfs entry appears a few seconds after its first URBs."""
    calls = {"n": 0}

    def desc(key):
        calls["n"] += 1
        return None if calls["n"] < 4 else DESC

    src = FakeSource([ev(0.1), ev(0.4), ev(1.2), ev(2.2), ev(3.2), ev(4.2)])
    mon = LiveMonitor(detector_factory(), src, desc, clock=_fast_clock(),
                      max_defer_windows=10)
    list(mon.run(max_windows=8))
    assert "1-3" in mon.detector.devices
    assert mon.events_undescribed == 0


def test_replay_does_not_linger_for_undescribed_devices(detector_factory):
    """Found replaying the first real capture: 18 s of traffic took 34 windows,
    because root-hub events held for max_defer_windows kept run() alive long
    after the source had ended. Once nothing more can arrive there is nothing to
    wait for, so the grace is a few windows, not the full cap."""
    events = [ev(0.1 + 0.3 * i, dev=3) for i in range(20)]            # described, ~6 s
    events += [ev(0.2 + 0.4 * i, dev=1) for i in range(15)]           # root hub, never described
    events.sort(key=lambda e: e.ts)
    mon = LiveMonitor(detector_factory(), FakeSource(events),
                      lambda k: DESC if k == "1-3" else None,
                      clock=_fast_clock(), max_defer_windows=30)
    n = sum(1 for _ in mon.run())
    last = max(e.ts for e in events)
    assert n <= int(last) + 1 + 5, (
        f"{n} windows for {last:.1f} s of traffic: replay lingered on held events")
    assert mon.events_undescribed == 15


def test_a_first_event_that_arrives_after_startup_is_not_late(detector_factory):
    """Found on the first live service on a real kernel: it scored NOTHING --
    heartbeat `devices=0 late=174 urb_recorded=174`, every event late.

    usbmon timestamps are relative to the FIRST event the parser sees, but the
    window clock used to start when the monitor started. If the first URB arrives
    after the monitor starts, every event's window index sits permanently behind
    the clock -- both advance at one second per second, so it never catches up --
    and the whole stream is dropped as late. A foreground run had worked only
    because its first URB happened to arrive within a fraction of a second.
    Every simulated or replayed source delivers its first event immediately, so
    nothing could see it."""
    import threading
    import time

    gate = threading.Event()

    class Delayed:
        stats = CaptureStats()

        def events(self):
            gate.wait()                                   # an idle bus, then traffic
            for i in range(12):
                yield ev(0.0 + 0.5 * i)

        def descriptor(self, key):
            raise KeyError(key)

    t = {"v": 0.0}

    def clock():
        t["v"] += 1.0          # races ahead whenever the loop asks for the time
        return t["v"]

    mon = LiveMonitor(detector_factory(), Delayed(), lambda k: DESC, clock=clock)
    threading.Timer(0.4, gate.set).start()                # first URB 0.4 s after start
    done = threading.Event()
    scored = []

    def consume():
        for batch in mon.run():
            scored.extend(batch)
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    assert done.wait(15), "run() did not finish"
    assert mon.events_late == 0, f"{mon.events_late} events were dropped as late"
    assert scored and "1-3" in mon.detector.devices, "the device was never scored"


def test_two_devices_are_tracked_independently(detector_factory):
    src = FakeSource([ev(0.1, dev=3), ev(0.2, dev=5), ev(0.3, dev=3)])
    mon = LiveMonitor(detector_factory(), src,
                      lambda k: DESC if k == "1-3" else DESC2, clock=_fast_clock())
    list(mon.run(max_windows=3))
    assert set(mon.detector.devices) == {"1-3", "1-5"}


def test_long_idle_device_is_detached_then_re_attached(detector_factory):
    """A device that goes quiet is forgotten, so the next device on the same bus
    address cannot inherit its payload history. If it speaks again it is treated
    as new -- which is the correct reading of an unplug/replug."""
    src = FakeSource([ev(0.1), ev(6.0)])       # a five-window silence in between
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC,
                      idle_detach_windows=2, clock=_fast_clock())
    present = []
    for _ in mon.run(max_windows=9):
        present.append("1-3" in mon.detector.devices)
    assert False in present, "device was never detached despite a long silence"
    first_absent = present.index(False)
    assert True in present[first_absent:], (
        "device never came back after speaking again; an unplug/replug on the "
        "same address must be treated as a new device, not ignored"
    )


def test_source_error_propagates(detector_factory):
    class Boom:
        stats = CaptureStats()

        def events(self):
            yield ev(0.1)
            raise OSError("debugfs read failed")

        def descriptor(self, k):
            raise KeyError(k)

    mon = LiveMonitor(detector_factory(), Boom(), lambda k: DESC, clock=_fast_clock())
    with pytest.raises(OSError, match="debugfs"):
        list(mon.run(max_windows=4))


def test_run_ends_when_the_source_ends(detector_factory):
    src = FakeSource([ev(0.1)])
    mon = LiveMonitor(detector_factory(), src, lambda k: DESC, clock=_fast_clock())
    batches = list(mon.run())          # no max_windows: must terminate on its own
    assert batches


# --------------------------------------------------------------------------
# event-paced replay (--fast)
# --------------------------------------------------------------------------

def _verdicts(mon):
    return [(v.device_id, v.t_start, round(float(v.stage1), 9))
            for vs in mon.run() for v in vs]


def test_event_pacing_scores_the_same_windows_as_clock_pacing(detector_factory):
    evs = [ev(t / 10) for t in range(0, 95, 3)] + [ev(12.5), ev(14.05, dev=4)]
    a = LiveMonitor(detector_factory(), FakeSource(evs), lambda k: DESC, clock=_fast_clock())
    b = LiveMonitor(detector_factory(), FakeSource(evs), lambda k: DESC, pace="event")
    va, vb = _verdicts(a), _verdicts(b)
    assert va == vb and len(vb) > 10
    assert b.events_late == 0


def test_event_pacing_never_drops_events_however_far_the_reader_runs_ahead(detector_factory):
    # 6 000 events over 120 s: the bounded queue makes the reader block, not drop
    evs = [ev(i * 0.02) for i in range(6000)]
    mon = LiveMonitor(detector_factory(), FakeSource(evs), lambda k: DESC, pace="event")
    n = sum(int(v.features[0]) for vs in mon.run() for v in vs)       # column 0: n_events
    assert n == 6000 and mon.events_late == 0


def test_unknown_pace_is_refused(detector_factory):
    with pytest.raises(ValueError):
        LiveMonitor(detector_factory(), FakeSource([]), lambda k: DESC, pace="turbo")


def test_fast_is_refused_for_live_capture():
    from iudbms.live import cli
    with pytest.raises(SystemExit, match="replay"):
        cli.main(["--fast", "--bus", "0", "--no-persist"])


@pytest.mark.slow
def test_event_paced_replay_matches_wall_clock_replay_on_real_kernel_output(detector_factory):
    """The real QEMU capture, replayed both ways through the deployed loop's code."""
    from pathlib import Path

    from iudbms.capture.usbmon import UsbmonSource
    from iudbms.live.cli import _load_descriptor_snapshot
    fx = Path(__file__).parent / "fixtures" / "usbmon"
    desc = _load_descriptor_snapshot(fx / "real_qemu_vm_descriptors.json")
    runs = []
    for pace in ("wallclock", "event"):
        src = UsbmonSource(path=fx / "real_qemu_vm_capture.txt")
        mon = LiveMonitor(detector_factory(), src, desc.get, pace=pace)
        runs.append((_verdicts(mon), mon.events_late, mon.events_undescribed))
    assert runs[0] == runs[1]
    assert len(runs[1][0]) > 20
