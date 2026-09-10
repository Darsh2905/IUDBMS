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
    ):
        self.detector = detector
        self.source = source
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

    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for ev in self.source.events():
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
        alive = True
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                alive = False
                break
            if isinstance(item, Exception):
                raise item
            w = int(item.ts // self.window)
            if w < current_window:
                # its window already closed; count it rather than silently
                # folding it into a later one and distorting that window's rate
                self.events_late += 1
                continue
            self._buckets.setdefault(w, {}).setdefault(item.device_key, []).append(item)
        return alive

    def _close_window(self, w: int) -> list[WindowVerdict]:
        t_start = w * self.window
        bucket = self._buckets.pop(w, {})
        for key, evs in self._deferred.items():
            bucket.setdefault(key, []).extend(evs)
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
                    # not enumerated yet; hold its events for the next window
                    if events:
                        self._deferred[key] = events
                    continue
                self.detector.attach(desc, key)
            verdicts.append(self.detector.observe_window(key, t_start, events))
            self._idle[key] = 0 if events else self._idle.get(key, 0) + 1

        for key, n in list(self._idle.items()):
            if n >= self.idle_detach_windows:
                self.detector.detach(key)
                self._idle.pop(key, None)
        self.windows_scored += 1
        return verdicts

    def run(self, max_windows: int | None = None) -> Iterator[list[WindowVerdict]]:
        """Yield one list of verdicts per closed window, until the source ends."""
        self.start()
        t0 = self._clock()
        w = 0
        try:
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
                # Keep scoring while the source is live, or while buckets still
                # hold future traffic. A live debugfs read never ends, so this
                # only terminates for finite sources such as a replay file.
                if not alive and not self._buckets and not self._deferred:
                    break
        finally:
            self.stop()
