"""The first REAL kernel usbmon output this project has parsed.

Everything before this ran on text the simulator wrote in usbmon's grammar. These
338 lines were emitted by Linux 6.8 (usbmon, text interface) in a QEMU guest while
`lsusb -v` read every device's descriptors and the kernel polled an empty USB
CD-ROM. The devices are virtual, so this says nothing about real device
BEHAVIOUR -- but it is the first evidence about the parser, the loss accounting
and the monitor loop against genuine kernel output, and it has already found two
bugs (CLAUDE.md finding 26).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from iudbms.capture.usbmon import UsbmonSource
from iudbms.live import StreamingDetector
from iudbms.live.cli import _load_descriptor_snapshot
from iudbms.live.monitor import LiveMonitor
from iudbms.policy import PolicyEngine
from iudbms.stage1 import Stage1
from iudbms.contract import N_FEATURES

FIX = Path(__file__).parent / "fixtures" / "usbmon"
CAP = FIX / "real_qemu_vm_capture.txt"
DESC = FIX / "real_qemu_vm_descriptors.json"


@pytest.fixture(scope="module")
def parsed():
    src = UsbmonSource(path=CAP)
    return list(src.events()), src.stats


def test_every_real_line_parses(parsed):
    """Plan §4.1: < 0.1% of URBs lost. Never measurable before: the simulator's
    lines are the ones the parser was written against."""
    events, st = parsed
    assert st.lines_seen == 338 and st.events_emitted == 338
    assert st.lines_unparsed == 0 and st.loss_rate < 0.001


def test_every_real_event_satisfies_the_contract(parsed):
    events, _ = parsed
    for e in events:
        e.validate()


def test_real_control_submissions_carry_their_setup_packet(parsed):
    """The line form the Sprint-0 parser silently dropped."""
    ctrl_s = [e for e in parsed[0] if e.ttype == "control" and e.evt == "S"]
    assert len(ctrl_s) == 108 and all(len(e.setup) == 8 for e in ctrl_s)


def test_real_traffic_covers_every_transfer_type_the_parser_handles(parsed):
    kinds = Counter((e.ttype, e.direction) for e in parsed[0] if e.evt == "C")
    assert {k[0] for k in kinds} == {"control", "bulk", "intr"}
    assert ("bulk", "out") in kinds and ("bulk", "in") in kinds


def test_real_mass_storage_traffic_is_scsi_bulk_only_transport(parsed):
    """A bulk OUT of 31 bytes starting 'USBC' is a Command Block Wrapper; the
    matching IN of 13 bytes starting 'USBS' is its status."""
    cbw = [e for e in parsed[0] if e.evt == "S" and e.ttype == "bulk"
           and e.direction == "out" and e.length == 31]
    csw = [e for e in parsed[0] if e.evt == "C" and e.ttype == "bulk"
           and e.direction == "in" and e.length == 13]
    assert cbw and csw
    assert all(e.data[:4] == b"USBC" for e in cbw)
    assert all(e.data[:4] == b"USBS" for e in csw)


def test_polling_interval_is_not_read_as_length(parsed):
    """The Sprint-0 defect: usbmon prints the interval beside the status."""
    intr = [e for e in parsed[0] if e.ttype == "intr"]
    assert intr and all(e.interval is not None and e.length <= 8 for e in intr)


def test_timestamps_are_monotone(parsed):
    ts = [e.ts for e in parsed[0]]
    assert ts == sorted(ts)


def test_descriptor_snapshot_has_no_root_hubs():
    """vm_capture.sh skips usbN, as SysfsScanner does -- but the kernel still
    emits traffic for devnum 1 on every bus."""
    snap = _load_descriptor_snapshot(DESC)
    assert set(snap) == {"1-2", "1-3", "1-4", "1-5", "1-6"}
    keys = {e.device_key for e in UsbmonSource(path=CAP).events()}
    assert {"1-1", "2-1", "3-1", "4-1"} <= keys - set(snap)


class _Queued:
    """The parsed capture, yielded on the reader thread, with a flag for when it
    has all been queued. A fake clock that races ahead of a file parser would
    reach window 500 before the first event existed and mark every event late;
    gating the clock on this is the same device test_traces.py uses."""

    def __init__(self, events):
        import threading
        self._events = events
        self.done = threading.Event()
        from iudbms.contract import CaptureStats
        self.stats = CaptureStats()

    def events(self):
        yield from self._events
        self.done.set()

    def descriptor(self, key):
        raise KeyError(key)


def test_real_capture_through_the_monitor():
    """Real traffic through the real loop: described devices are scored, the four
    undescribed root hubs are counted and dropped rather than held, and an 18 s
    capture does not take 34 windows to replay."""
    X = np.random.default_rng(0).normal(0, 1, (300, N_FEATURES))
    s1 = Stage1(seed=0).fit(X, ["hid"] * 300)
    det = StreamingDetector(s1, None, PolicyEngine(threshold=0.9, k=4, n=6))
    snap = _load_descriptor_snapshot(DESC)
    src = _Queued(list(UsbmonSource(path=CAP).events()))
    t = {"v": 0.0}

    def clock():
        if not src.done.is_set():
            return 0.0
        t["v"] += 1.0
        return t["v"]

    mon = LiveMonitor(det, src, snap.get, clock=clock, idle_detach_windows=1000)
    scored = {v.device_id for batch in mon.run() for v in batch}
    assert mon.events_late == 0
    assert scored == set(snap), f"scored {sorted(scored)}"
    assert mon.events_undescribed == 158          # every root-hub event, counted
    assert mon.windows_scored <= 18 + 1 + 5, (
        f"{mon.windows_scored} windows for 18.1 s of traffic")
