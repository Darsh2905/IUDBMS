"""
Live monitoring — drive the streaming detector from a running capture.

Everything else in this package scores something already recorded. This is the
loop that watches USB traffic as it happens: read URBs continuously, cut them
into windows on the wall clock, discover devices as they enumerate, and score
each device's window the moment it closes.

Two things make it different from replay, and both are why it needs its own
class rather than a flag on the replay script:

* **Events arrive while you are working.** A blocking read on the debugfs node
  cannot also be a scoring loop, so a reader thread drains the source into a
  queue and the main loop wakes on window boundaries.
* **Windows are cut on the kernel's timestamps, not on arrival time.** The
  reader thread is subject to scheduling jitter and the queue adds latency, so
  bucketing by when we happened to dequeue an event would smear traffic across
  windows and make the same capture score differently under load. usbmon
  timestamps come from the kernel and are authoritative.
* **Devices come and go.** A device that was not present at start-up must be
  attached when its first URB arrives, and one that is unplugged must be
  detached so the next device on the same bus address does not inherit its
  payload history.

The monitor is deliberately source-agnostic: it takes any ``EventSource``, so
it is testable against a fake without a kernel, and takes any descriptor lookup,
so it works with sysfs on Linux and with a recorded snapshot elsewhere.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator

from ..contract import DeviceDescriptor, URBEvent
from .detector import StreamingDetector, WindowVerdict

__all__ = ["LiveMonitor"]


class LiveMonitor:
    """Cut a live event stream into windows and score each device's traffic.

    ``descriptor_for`` is called when an unknown device first appears; returning
    ``None`` means "not enumerated yet", and the monitor will ask again on the
    next window rather than dropping the device.
    """

    def __init__(
        self,
        detector: StreamingDetector,
        source,
        descriptor_for: Callable[[str], DeviceDescriptor | None],
        *,
        window: float = 1.0,
        idle_detach_windows: int = 30,
        clock: Callable[[], float] = time.monotonic,
        recorder=None,
        max_defer_windows: int = 30,
        end_grace_windows: int = 3,
    ):
        self.detector = detector
        self.source = source
        #: Optional ``storage.traces.TraceRecorder``. Receives every drained
        #: event (late ones flagged) and every attach/detach, which is what lets
        #: a stored run be re-featurised exactly as this loop scored it.
        self.recorder = recorder
        self._last_window: int | None = None
        #: Sticky once the source's end-of-stream sentinel is seen. It used to be
        #: reported only by the drain that dequeued it, so a finite source whose
        #: traffic still filled later windows at that moment was treated as live
        #: again on every following drain, and run() never returned.
        self._ended = False
        self.descriptor_for = descriptor_for
        self.window = window
        self.idle_detach_windows = idle_detach_windows
        self._clock = clock
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._idle: dict[str, int] = {}
        #: window index -> device key -> events, filled by timestamp
        self._buckets: dict[int, dict[str, list[URBEvent]]] = {}
        self._deferred: dict[str, list[URBEvent]] = {}
        self.windows_scored = 0
        self.events_late = 0
        #: How long a not-yet-enumerated device's events are held. A real device
        #: enumerates within a second or two; a device that has had no descriptor
        #: for this long never will -- root hubs, which SysfsScanner skips on
        #: purpose, are the standing example. Without a cap their events were held
        #: and RE-held every window, growing without bound.
        self.max_defer_windows = max_defer_windows
        self._defer_since: dict[str, int] = {}
        #: Once the source has ended nothing new can arrive to prompt an
        #: enumeration, so held events get this many more windows, not the full
        #: max_defer_windows. Without it an 18 s replay took 34 windows.
        self.end_grace_windows = end_grace_windows
        self._windows_since_end = 0
        #: Items pulled off the queue while waiting for the time origin; they are
        #: put back in front of the queue's contents for the first drain.
        self._pending: list = []
        #: Events dropped because their device never had a descriptor. Counted,
        #: not silent: they still reach the raw trace, which records at drain.
        self.events_undescribed = 0

    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for ev in self.source.events():
                if self._ended:
                    self._windows_since_end += 1
                    if self._windows_since_end > self.end_grace_windows and self._deferred:
                        self.events_undescribed += sum(
                            len(v) for v in self._deferred.values())
                        self._deferred = {}
                if self._stop.is_set():
                    break
                self._q.put(ev)
        except Exception as exc:                       # noqa: BLE001
            self._q.put(exc)
        finally:
            self._q.put(None)                          # end-of-stream sentinel

    def start(self) -> None:
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="iudbms-capture")
        self._reader.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _drain(self, current_window: int) -> bool:
        """Move everything queued into per-window, per-device buckets.

        Placement uses the event's own timestamp. Returns False once the source
        has ended.
        """
        on_time: list[URBEvent] = []
        late: list[URBEvent] = []
        while True:
            if self._pending:
                item = self._pending.pop(0)
            else:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
            if item is None:
                self._ended = True
                break
            if isinstance(item, Exception):
                raise item
            w = int(item.ts // self.window)
            if w < current_window:
                # its window already closed; count it rather than silently
                # folding it into a later one and distorting that window's rate
                self.events_late += 1
                late.append(item)
                continue
            on_time.append(item)
            self._buckets.setdefault(w, {}).setdefault(item.device_key, []).append(item)
        if self.recorder is not None:
            # Late events are still part of what the kernel emitted, so they are
            # stored -- flagged, because this loop never scored them.
            self.recorder.record(on_time)
            self.recorder.record(late, late=True)
        return not self._ended

    def _close_window(self, w: int) -> list[WindowVerdict]:
        t_start = w * self.window
        bucket = self._buckets.pop(w, {})
        for key, evs in self._deferred.items():
            # Held events are OLDER than this window's, so they go first. They
            # used to be appended after, and accumulated in reverse window order
            # across repeated deferrals; the extractor diffs timestamps as given,
            # so that produced negative inter-arrival times on the first scored
            # window of every device sysfs had not yet enumerated.
            bucket[key] = evs + bucket.get(key, [])
        self._deferred = {}

        verdicts: list[WindowVerdict] = []
        # Score every device we know about, not only those with traffic: an
        # idle second is an observation, and the silence after a finished
        # payload is the only signal some attacks leave.
        known = set(self.detector.devices) | set(bucket)
        for key in sorted(known):
            events = bucket.get(key, [])
            if key not in self.detector.devices:
                desc = self.descriptor_for(key)
                if desc is None:
                    # not enumerated yet; hold its events for the next window --
                    # but only for max_defer_windows, then count and drop them
                    if events:
                        since = self._defer_since.setdefault(key, w)
                        if w - since >= self.max_defer_windows:
                            self.events_undescribed += len(events)
                        else:
                            self._deferred[key] = events
                    continue
                self._defer_since.pop(key, None)
                self.detector.attach(desc, key)
                if self.recorder is not None:
                    self.recorder.on_attach(key, t_start, desc)
            verdicts.append(self.detector.observe_window(key, t_start, events))
            self._idle[key] = 0 if events else self._idle.get(key, 0) + 1

        for key, n in list(self._idle.items()):
            if n >= self.idle_detach_windows:
                self.detector.detach(key)
                self._idle.pop(key, None)
                if self.recorder is not None:
                    self.recorder.on_detach(key, t_start + self.window)
        self.windows_scored += 1
        self._last_window = w
        return verdicts

    def _await_origin(self) -> float | None:
        """Block until the first event arrives and return the window clock origin.

        Event timestamps are relative to the first event the parser sees, so the
        window clock has to start from that same instant. It used to start when
        the monitor did: a first URB arriving even a second later left every
        event's window permanently behind the clock (both advance at one second
        per second), and the whole stream was dropped as late. A real service
        scored nothing for exactly this reason. Anchoring on arrival makes the
        offset zero by construction, however long the bus was idle first.

        ``origin = now - first_event.ts`` so an event arriving "on time" lands in
        the window that is open when it arrives. Returns None if the source ends
        or the monitor is stopped before any event arrives.
        """
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                self._ended = True
                return None
            if isinstance(item, Exception):
                raise item
            self._pending.append(item)
            return self._clock() - item.ts
        return None

    def run(self, max_windows: int | None = None) -> Iterator[list[WindowVerdict]]:
        """Yield one list of verdicts per closed window, until the source ends."""
        self.start()
        w = 0
        try:
            t0 = self._await_origin()
            if t0 is None:
                return
            while max_windows is None or w < max_windows:
                deadline = t0 + (w + 1) * self.window
                while True:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.05))
                alive = self._drain(w)
                yield self._close_window(w)
                w += 1
                if self._ended:
                    self._windows_since_end += 1
                    if self._windows_since_end > self.end_grace_windows and self._deferred:
                        self.events_undescribed += sum(
                            len(v) for v in self._deferred.values())
                        self._deferred = {}
                if self._stop.is_set():
                    # Honour stop() here rather than waiting for the reader: on
                    # an idle live bus it is blocked in a debugfs read and would
                    # not notice until the next URB arrived.
                    break
                # Keep scoring while the source is live, or while buckets still
                # hold future traffic. A live debugfs read never ends, so this
                # only terminates for finite sources such as a replay file.
                if not alive and not self._buckets and not self._deferred:
                    break
        finally:
            self.stop()
            if self.recorder is not None:
                end = (None if self._last_window is None
                       else (self._last_window + 1) * self.window)
                self.recorder.close(end)
