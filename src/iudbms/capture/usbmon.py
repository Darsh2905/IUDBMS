"""
L0/L1 — usbmon (debugfs) capture backend.

Reads the kernel's text-format URB log:

    /sys/kernel/debug/usb/usbmon/<bus>u      one bus
    /sys/kernel/debug/usb/usbmon/0u          all buses

and normalises it into :class:`~iudbms.contract.URBEvent`. This is the fallback
capture path from plan §2 and is built before eBPF deliberately: it works on any
Linux box with debugfs mounted, and both paths feed the identical event stream
so the choice never blocks L2-L5.

Line grammar (Documentation/usb/usbmon.rst, "Raw text format")
--------------------------------------------------------------

    <tag> <ts_us> <evt> <type><dir>:<bus>:<dev>:<ep> <status-word> \
        [<iso-descriptors>] <length> [<data-tag> <data-words>]

``<status-word>`` is the awkward part and is where the Sprint-0 parser went
wrong. It takes one of two shapes:

  ``s <bmRequestType> <bRequest> <wValue> <wIndex> <wLength>``
      A control *submission*. usbmon prints the 8-byte SETUP packet here
      instead of a status. There is no status number on such a line.

  ``<status>[:<interval>[:<start_frame>[:<error_count>]]]``
      ``interval`` appears for interrupt and isochronous transfers only;
      ``start_frame`` and ``error_count`` for isochronous only.

So in ``0:8 8``, the first ``8`` is the polling **interval** and the second is
the **data length**. The old parser read the interval as the length; the two
coincide for 8-byte HID reports, which is why simulator output never exposed
it. They are separate fields in the contract now.

``<data-tag>`` is a single character. Only ``=`` means the payload follows;
``-``, ``<`` and ``>`` all mean the kernel chose not to capture it, which is
recorded as ``data_truncated`` rather than silently becoming an empty payload.

Parsing is done by whitespace tokenisation rather than one large regex: the
optional-field combinatorics above make a single pattern both slower and much
harder to audit, and tokenisation tolerates unknown fields (iso frame
descriptors) by scanning forward for the data tag instead of failing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Iterable, Iterator

from ..contract import (
    USBMON_TTYPE,
    CaptureStats,
    DeviceDescriptor,
    URBEvent,
)

__all__ = ["parse_line", "UsbmonParser", "UsbmonSource", "DEFAULT_USBMON_DIR"]

DEFAULT_USBMON_DIR = Path("/sys/kernel/debug/usb/usbmon")

#: Only "=" means data words follow. "-", "<" and ">" mean not captured.
_DATA_TAGS = frozenset("=-<>")

#: usbmon text timestamps are CLOCK_MONOTONIC with the seconds truncated to 12 bits
#: (drivers/usb/mon/mon_text.c: ``stamp = (tv_sec & 0xFFF) * USEC_PER_SEC + usec``),
#: so they wrap every **4096 s**, not at 2^32 us. This constant was 2^32 until the
#: first hour-long sessions: every event after a wrap then landed 198.97 s in the
#: future. A wrap happens whenever the monotonic clock crosses a multiple of 4096 s,
#: so any capture can straddle one, and a running monitor meets one every 68 min.
#: A backward jump larger than half the range is a wrap, not reordering.
_TS_WRAP = 4096 * 1_000_000
_TS_WRAP_GUARD = _TS_WRAP // 2


def _parse_address(word: str) -> tuple[str, str, int, int, int] | None:
    """``Ii:1:003:1`` -> (ttype, direction, bus, dev, ep)."""
    head, _, rest = word.partition(":")
    if len(head) != 2:
        return None
    ttype = USBMON_TTYPE.get(head[0])
    if ttype is None or head[1] not in "io":
        return None
    bits = rest.split(":")
    if len(bits) != 3:
        return None
    try:
        bus, dev, ep = (int(b) for b in bits)
    except ValueError:
        return None
    return ttype, ("in" if head[1] == "i" else "out"), bus, dev, ep


def parse_line(line: str) -> URBEvent | None:
    """Parse one usbmon text line. Returns ``None`` if it is not a URB record.

    Timestamps come back in **seconds since the kernel epoch**, unadjusted;
    :class:`UsbmonParser` is what rebases them onto enumeration and undoes
    counter wrap.
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    # [0] urb tag (hex), [1] timestamp us, [2] event type, [3] address word
    try:
        ts_us = int(parts[1])
    except ValueError:
        return None
    evt = parts[2]
    if evt not in ("S", "C", "E"):
        return None
    addr = _parse_address(parts[3])
    if addr is None:
        return None
    ttype, direction, bus, dev, ep = addr

    # --- status word -------------------------------------------------------
    i = 4
    setup = b""
    interval: int | None = None
    if parts[i] == "s":
        # Control submission: the SETUP packet stands in for the status.
        #   s bmRequestType bRequest wValue wIndex wLength
        if len(parts) < i + 6:
            return None
        try:
            b_request_type, b_request = int(parts[i + 1], 16), int(parts[i + 2], 16)
            w_value, w_index, w_length = (int(parts[i + j], 16) for j in (3, 4, 5))
        except ValueError:
            return None
        setup = bytes((b_request_type, b_request)) + b"".join(
            w.to_bytes(2, "little") for w in (w_value, w_index, w_length)
        )
        status = -115  # -EINPROGRESS; a submission has no completion status yet
        i += 6
    else:
        nums = parts[i].split(":")
        try:
            status = int(nums[0])
            if len(nums) > 1 and ttype in ("intr", "iso"):
                interval = int(nums[1])
        except ValueError:
            return None
        i += 1

    # --- data length -------------------------------------------------------
    if i >= len(parts):
        return None
    try:
        length = int(parts[i])
    except ValueError:
        return None
    i += 1

    # --- data tag ----------------------------------------------------------
    # Scan forward: isochronous lines may carry frame descriptors between the
    # length and the tag. Data words are never single characters, so a
    # one-character token from the tag set is unambiguous.
    data = b""
    truncated = False
    tag_at = next(
        (j for j in range(i, len(parts)) if len(parts[j]) == 1 and parts[j] in _DATA_TAGS),
        None,
    )
    if tag_at is None:
        truncated = length > 0
    elif parts[tag_at] == "=":
        try:
            data = bytes.fromhex("".join(parts[tag_at + 1:]))
        except ValueError:
            return None
        truncated = len(data) < length
    else:
        truncated = length > 0

    return URBEvent(
        ts=ts_us / 1e6,
        evt=evt,  # type: ignore[arg-type]
        ttype=ttype,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        bus=bus,
        dev=dev,
        ep=ep,
        status=status,
        length=length,
        data=data,
        interval=interval,
        data_truncated=truncated,
        setup=setup,
    )


class UsbmonParser:
    """Stateful line -> event conversion.

    Owns the two things a pure function cannot: rebasing timestamps onto
    device enumeration, and undoing the 32-bit microsecond counter wrap.
    Populates :class:`CaptureStats` so loss is measurable (plan §4.1).
    """

    def __init__(self, *, origin_us: int | None = None, strict: bool = False):
        self.stats = CaptureStats()
        self.strict = strict
        self._origin_us = origin_us
        self._prev_raw_us: int | None = None
        self._wraps = 0

    def _rebase(self, ts_seconds: float) -> float:
        raw_us = int(round(ts_seconds * 1e6))
        if self._prev_raw_us is not None and raw_us < self._prev_raw_us - _TS_WRAP_GUARD:
            self._wraps += 1
        self._prev_raw_us = raw_us
        abs_us = raw_us + self._wraps * _TS_WRAP
        if self._origin_us is None:
            self._origin_us = abs_us
        return (abs_us - self._origin_us) / 1e6

    def feed(self, lines: Iterable[str]) -> Iterator[URBEvent]:
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            self.stats.lines_seen += 1
            ev = parse_line(line)
            if ev is None:
                self.stats.note_unparsed(line)
                continue
            ev = URBEvent(
                ts=self._rebase(ev.ts), evt=ev.evt, ttype=ev.ttype,
                direction=ev.direction, bus=ev.bus, dev=ev.dev, ep=ev.ep,
                status=ev.status, length=ev.length, data=ev.data,
                interval=ev.interval, data_truncated=ev.data_truncated,
                setup=ev.setup,
            )
            if self.strict:
                ev.validate()
            self.stats.note_event(ev)
            yield ev


class UsbmonSource:
    """An :class:`~iudbms.contract.EventSource` over usbmon text.

    Works identically against a live debugfs node and a recorded capture, which
    is what makes a VM-recorded trace replayable on any host through the same
    code path.

        UsbmonSource.live(bus=1)                      # /sys/kernel/debug/...
        UsbmonSource(path="capture.txt")              # recorded
        UsbmonSource(stream=io.StringIO(text))        # in memory
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        stream: IO[str] | None = None,
        descriptors: dict[str, DeviceDescriptor] | None = None,
        origin_us: int | None = None,
        strict: bool = False,
    ):
        if (path is None) == (stream is None):
            raise ValueError("pass exactly one of path= or stream=")
        self._path = Path(path) if path is not None else None
        self._stream = stream
        self._descriptors = dict(descriptors or {})
        self._parser = UsbmonParser(origin_us=origin_us, strict=strict)

    @classmethod
    def live(cls, bus: int = 0, *, usbmon_dir: Path = DEFAULT_USBMON_DIR, **kw) -> UsbmonSource:
        """Open a live bus. ``bus=0`` is usbmon's "all buses" node.

        Requires Linux with debugfs mounted, ``usbmon`` loaded, and read
        permission (usually root)::

            sudo modprobe usbmon
            sudo mount -t debugfs none /sys/kernel/debug   # if not mounted
        """
        node = Path(usbmon_dir) / f"{bus}u"
        if not node.exists():
            raise FileNotFoundError(
                f"{node} not found. Is this Linux with debugfs mounted and "
                f"usbmon loaded? (sudo modprobe usbmon)"
            )
        return cls(path=node, **kw)

    @property
    def stats(self) -> CaptureStats:
        return self._parser.stats

    def descriptor(self, device_key: str) -> DeviceDescriptor:
        return self._descriptors[device_key]

    def add_descriptor(self, desc: DeviceDescriptor) -> None:
        self._descriptors[desc.device_key] = desc

    def events(self) -> Iterator[URBEvent]:
        if self._stream is not None:
            yield from self._parser.feed(self._stream)
            return
        assert self._path is not None
        # errors="replace": a live debugfs read can tear mid-line under load;
        # a decode error must not abort a capture.
        with open(self._path, "r", errors="replace") as fh:
            yield from self._parser.feed(fh)
