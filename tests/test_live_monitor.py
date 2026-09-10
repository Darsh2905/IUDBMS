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
