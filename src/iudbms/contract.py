"""
IUDBMS — frozen data contract (milestone 1).

This module is the interface every layer agrees on and **is not renegotiated**
(implementation plan §1). It exists so that L2 can consume L1 output without
knowing whether it came from usbmon text, an eBPF ring buffer, a recorded
replay file, or the simulator — and so that all four can be swapped in an
evaluation without touching anything downstream.

Layer boundaries
----------------
    L0/L1  EventSource   -> Iterator[URBEvent] + DeviceDescriptor + CaptureStats
    L2     FeatureExtractor(events, t0, descriptor) -> float32[N_FEATURES]
    L3+    consume feature vectors only

Anything that imports from a concrete capture backend outside of
``iudbms.capture`` is a contract violation.

Design notes that are load-bearing (see CLAUDE.md "Contract rationale")
----------------------------------------------------------------------
``length`` is the *data length* — the requested length on a submission and the
actual transferred length on a completion. It is **never** the polling
interval. The Sprint-0 parser conflated the two because usbmon prints them
adjacently as ``status:interval`` followed by the length; ``interval`` now has
its own field so the two cannot be confused again.

``data_truncated`` distinguishes "this was a zero-length transfer" from
"usbmon chose not to capture the payload" (the ``-`` data tag). Without it the
payload-modality features silently treat an uncaptured bulk stream as a
perfectly repeating one.

``setup`` carries the 8-byte SETUP packet that usbmon prints in place of a
status word on control submissions (``s 80 06 0100 0000 0012``). It is kept out
of ``data`` because ``length`` is the *requested data* length, which is
routinely smaller than 8 — putting SETUP in ``data`` would make well-formed
events fail the ``len(data) <= length`` invariant. No current feature reads it;
it is captured because discarding protocol content at the capture layer is
irreversible.

``URBEvent`` is frozen as well as slotted: events are values, and freezing them
means one feature extractor cannot mutate the stream a later extractor sees.
Validation is deliberately *not* in ``__post_init__`` — the plan requires
≥20k events/s (§4.1) and per-event validation costs more than it is worth in
the hot path. Use :meth:`URBEvent.validate` in tests and at source boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "EventKind", "TransferType", "Direction", "Modality",
    "USBMON_TTYPE", "TTYPE_USBMON",
    "CLASS_AUDIO", "CLASS_CDC", "CLASS_HID", "CLASS_MASS_STORAGE", "CLASS_VIDEO",
    "Endpoint", "Interface", "DeviceDescriptor",
    "URBEvent", "CaptureStats", "EventSource",
    "FEATURE_SCHEMA", "FEATURE_NAMES", "N_FEATURES", "FEATURE_DTYPE",
    "MODALITIES", "modality_of", "modality_indices", "feature_index",
    "FeatureExtractor",
    "ContractError",
]

#: Bumped only when a field is added, removed or re-typed. Trained model
#: checkpoints record this and refuse to load against a different major.
#: 1.1.0 — added URBEvent.setup (append-only, defaulted; no checkpoint churn).
#: 2.0.0 — appended feature 29, payload_seq_novelty. MAJOR: the vector changed
#:         width, so every checkpoint trained against 1.x is invalid and
#:         LSTMDetector.load refuses them. Feature ORDER is still append-only.
CONTRACT_VERSION = "3.0.0"


class ContractError(ValueError):
    """Raised when a value violates the frozen data contract."""


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

EventKind = Literal["S", "C", "E"]           # submission, completion, error
TransferType = Literal["control", "iso", "intr", "bulk"]
Direction = Literal["in", "out"]
Modality = Literal["timing", "payload", "protocol", "identity", "host"]

#: usbmon's transfer-type letter -> canonical name (Documentation/usb/usbmon.rst)
USBMON_TTYPE: dict[str, TransferType] = {
    "C": "control", "Z": "iso", "I": "intr", "B": "bulk",
}
TTYPE_USBMON: dict[TransferType, str] = {v: k for k, v in USBMON_TTYPE.items()}

# USB interface class codes this system models explicitly.
CLASS_AUDIO = 0x01
CLASS_CDC = 0x02
CLASS_HID = 0x03
CLASS_MASS_STORAGE = 0x08
CLASS_VIDEO = 0x0E


# --------------------------------------------------------------------------
# L1 — device identity
# --------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Endpoint:
    """One endpoint descriptor as exposed under /sys/bus/usb/devices/."""

    addr: int                      # bEndpointAddress, e.g. 0x81
    type: TransferType
    interval: int | None = None    # bInterval; intr/iso only

    @property
    def number(self) -> int:
        """Endpoint number with the direction bit masked off."""
        return self.addr & 0x0F

    @property
    def direction(self) -> Direction:
        return "in" if self.addr & 0x80 else "out"


@dataclass(slots=True, frozen=True)
class Interface:
    """One USB interface.

    ``first_seen_ts`` is seconds since device enumeration, so a composite
    device that exposes an interface late (the ``identity_switch`` attack
    class) is represented by data rather than by a side-channel flag. The
    Sprint-0 code carried a single ``late_iface_ts`` on the descriptor and
    reconstructed visibility as ``interfaces[:-1]``, which is only correct
    when exactly one interface arrives late *and* it happens to sort last.
    """

    bInterfaceClass: int
    bInterfaceSubClass: int = 0
    bInterfaceProtocol: int = 0
    endpoints: tuple[Endpoint, ...] = ()
    first_seen_ts: float = 0.0


@dataclass(slots=True, frozen=True)
class DeviceDescriptor:
    """Enumeration-time identity — what the device *claims* to be.

    Deliberately separate from URBEvent: the entire thesis is that these two
    can disagree, so nothing may collapse them into one record.
    """

    bus: int
    devnum: int
    vid: int
    pid: int
    bDeviceClass: int = 0x00
    serial_present: bool = False
    manufacturer: str | None = None
    product: str | None = None
    interfaces: tuple[Interface, ...] = ()

    @property
    def device_key(self) -> str:
        return f"{self.bus}-{self.devnum}"

    def visible_at(self, t: float) -> tuple[Interface, ...]:
        """Interfaces the host had enumerated at time ``t`` (s since enum)."""
        return tuple(i for i in self.interfaces if i.first_seen_ts <= t)

    def classes_at(self, t: float) -> frozenset[int]:
        return frozenset(i.bInterfaceClass for i in self.visible_at(t))

    def has_late_interface(self) -> bool:
        return any(i.first_seen_ts > 0.0 for i in self.interfaces)


# --------------------------------------------------------------------------
# L1 — the event record
# --------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class URBEvent:
    """One USB Request Block observation.

    Field semantics are fixed by usbmon's on-wire format so that the eBPF path
    can emit records that are indistinguishable to every consumer.
    """

    ts: float                      # seconds, monotonic, relative to enumeration
    evt: EventKind
    ttype: TransferType
    direction: Direction
    bus: int
    dev: int
    ep: int                        # endpoint number 0-15, direction bit stripped
    status: int                    # 0 = OK, negative = -errno (e.g. -115 EINPROGRESS)
    length: int                    # DATA length. Requested on S, actual on C.
    data: bytes = b""              # captured payload prefix; may be shorter than length
    interval: int | None = None    # polling interval; intr/iso only
    data_truncated: bool = False   # payload absent or clipped by the capture layer
    setup: bytes = b""             # 8-byte SETUP packet; control submissions only

    @property
    def device_key(self) -> str:
        return f"{self.bus}-{self.dev}"

    @property
    def is_completion(self) -> bool:
        return self.evt == "C"

    def validate(self) -> None:
        """Raise :class:`ContractError` if this event is not well formed.

        Not called per-event in the capture hot path; used by tests and by
        ``--strict`` capture runs.
        """
        if not np.isfinite(self.ts):
            raise ContractError(f"ts must be finite, got {self.ts!r}")
        if self.evt not in ("S", "C", "E"):
            raise ContractError(f"evt must be S/C/E, got {self.evt!r}")
        if self.ttype not in USBMON_TTYPE.values():
            raise ContractError(f"unknown transfer type {self.ttype!r}")
        if self.direction not in ("in", "out"):
            raise ContractError(f"direction must be in/out, got {self.direction!r}")
        if not 0 <= self.ep <= 15:
            raise ContractError(
                f"ep must be an endpoint number 0-15 with the direction bit "
                f"stripped, got {self.ep!r}"
            )
        if self.length < 0:
            raise ContractError(f"length must be non-negative, got {self.length!r}")
        if len(self.data) > self.length:
            raise ContractError(
                f"captured {len(self.data)} bytes for a transfer of "
                f"length {self.length}"
            )
        if self.data and len(self.data) < self.length and not self.data_truncated:
            raise ContractError(
                f"payload is short ({len(self.data)} of {self.length} bytes) but "
                f"data_truncated is False"
            )
        if self.interval is not None and self.ttype not in ("intr", "iso"):
            raise ContractError(
                f"interval is only meaningful for intr/iso, got {self.ttype!r}"
            )
        if self.setup:
            if self.ttype != "control":
                raise ContractError(
                    f"setup packet is only meaningful for control, got {self.ttype!r}"
                )
            if len(self.setup) != 8:
                raise ContractError(
                    f"SETUP packet must be exactly 8 bytes, got {len(self.setup)}"
                )


@dataclass(slots=True)
class CaptureStats:
    """Loss accounting for one capture run.

    Plan §4.1 requires "< 0.1 % of URBs lost" and "usbmon and eBPF agree within
    1 %". Neither is measurable unless every source reports what it discarded,
    so this is part of the contract rather than a debugging aid. The Sprint-0
    parser dropped unparseable lines with no counter at all.
    """

    lines_seen: int = 0
    events_emitted: int = 0
    lines_unparsed: int = 0
    events_dropped_overflow: int = 0     # eBPF ring-buffer / usbmon buffer loss
    first_ts: float | None = None
    last_ts: float | None = None
    unparsed_samples: list[str] = field(default_factory=list)

    #: Keep at most this many rejected lines for post-hoc diagnosis.
    MAX_SAMPLES = 20

    def note_unparsed(self, line: str) -> None:
        self.lines_unparsed += 1
        if len(self.unparsed_samples) < self.MAX_SAMPLES:
            self.unparsed_samples.append(line.rstrip("\n"))

    def note_event(self, ev: URBEvent) -> None:
        self.events_emitted += 1
        if self.first_ts is None:
            self.first_ts = ev.ts
        self.last_ts = ev.ts

    @property
    def loss_rate(self) -> float:
        """Fraction of observed records that did not become events."""
        total = self.lines_seen + self.events_dropped_overflow
        if total == 0:
            return 0.0
        lost = self.lines_unparsed + self.events_dropped_overflow
        return lost / total

    @property
    def duration(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return self.last_ts - self.first_ts

    @property
    def event_rate(self) -> float:
        return self.events_emitted / self.duration if self.duration > 0 else 0.0


@runtime_checkable
class EventSource(Protocol):
    """L0/L1. usbmon, eBPF, replay and simulator are interchangeable here.

    Implementations must be iterable exactly once (they may wrap a live fd) and
    must populate :attr:`stats` as they go.
    """

    def events(self) -> Iterator[URBEvent]:
        """Yield events in non-decreasing timestamp order."""
        ...

    def descriptor(self, device_key: str) -> DeviceDescriptor:
        """Enumeration-time identity for one device, or raise KeyError."""
        ...

    @property
    def stats(self) -> CaptureStats:
        ...


# --------------------------------------------------------------------------
# L2 — the feature vector
# --------------------------------------------------------------------------

#: The multi-modal window vector (plan §4.2, novelty pillar N1). 28 features in
#: contract 1.x; 29 from 2.0.0.
#:
#: **Order is frozen.** It is baked into trained checkpoints, the SQLite
#: ``windows.features`` JSON, and every published figure. Append-only; never
#: reorder. The modality tag drives the §8.3 single-modality ablations, so
#: ablations select by tag rather than by hardcoded column slices.
FEATURE_SCHEMA: tuple[tuple[str, Modality], ...] = (
    # --- timing (9) ---
    ("n_events",                 "timing"),
    ("ev_rate",                  "timing"),
    ("mean_iat",                 "timing"),
    ("std_iat",                  "timing"),
    ("cv_iat",                   "timing"),
    ("min_iat",                  "timing"),
    ("p10_iat",                  "timing"),
    ("iat_entropy",              "timing"),
    ("burstiness_fano",          "timing"),
    # --- payload (5 of 7; the remaining 2 follow the protocol block) ---
    ("mean_len",                 "payload"),
    ("std_len",                  "payload"),
    ("max_len",                  "payload"),
    ("bytes_in_rate",            "payload"),
    ("bytes_out_rate",           "payload"),
    # --- protocol (6) ---
    ("n_endpoints",              "protocol"),
    ("frac_intr",                "protocol"),
    ("frac_bulk",                "protocol"),
    ("frac_iso",                 "protocol"),
    ("frac_ctrl",                "protocol"),
    ("frac_in",                  "protocol"),
    # --- payload, continued (2) ---
    ("payload_novelty",          "payload"),
    ("repeat_ratio",             "payload"),
    # --- payload, appended in contract 2.0.0 (see FeatureExtractor.reset) ---
    ("payload_seq_novelty",      "payload"),
    # --- identity (6) ---
    ("n_interfaces",             "identity"),
    ("has_hid",                  "identity"),
    ("has_storage",              "identity"),
    ("class_behaviour_mismatch", "identity"),
    ("late_iface_flag",          "identity"),
    ("log_time_since_enum",      "identity"),
    # --- host, appended in contract 3.0.0 ---
    # Out of modality order on purpose: the schema is APPEND-ONLY (the comment
    # above is not advisory -- a digest of this tuple is asserted by
    # tests/test_contract.py and the order is baked into checkpoints and the
    # SQLite windows.features JSON). Grouping it beside the other modalities
    # would shift every identity index by one.
    ("concurrent_device_rate",   "host"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _ in FEATURE_SCHEMA)
N_FEATURES: int = len(FEATURE_SCHEMA)
FEATURE_DTYPE = np.float32
MODALITIES: tuple[Modality, ...] = ("timing", "payload", "protocol", "identity",
                                    "host")

_NAME_TO_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_MODALITY_TO_INDICES: dict[str, tuple[int, ...]] = {
    m: tuple(i for i, (_, mm) in enumerate(FEATURE_SCHEMA) if mm == m)
    for m in MODALITIES
}


def feature_index(name: str) -> int:
    """Column index of a named feature. Raises KeyError if unknown."""
    return _NAME_TO_INDEX[name]


def modality_of(name: str) -> Modality:
    return FEATURE_SCHEMA[_NAME_TO_INDEX[name]][1]


def modality_indices(*modalities: Modality) -> tuple[int, ...]:
    """Column indices belonging to the given modalities, ascending.

    Used by the §8.3 ablation harness to build single-modality variants
    without hardcoding column numbers.
    """
    out: set[int] = set()
    for m in modalities:
        if m not in _MODALITY_TO_INDICES:
            raise ContractError(f"unknown modality {m!r}; expected one of {MODALITIES}")
        out.update(_MODALITY_TO_INDICES[m])
    return tuple(sorted(out))


class FeatureExtractor(ABC):
    """L2. Turns a window of events plus a descriptor into a fixed vector.

    Subclasses implement :meth:`extract`. Callers that want the contract
    enforced (tests, ``--strict`` runs) use :meth:`extract_checked`.

    **Extractors carry per-device state and are not reusable across devices.**
    Contract 2.0.0 added ``payload_seq_novelty``, which asks whether this
    window's payload transitions have been seen earlier *in this session* — a
    cross-window question a stateless extractor cannot answer. One extractor
    instance therefore belongs to one device session; call :meth:`reset`
    between sessions or the previous device's history leaks into the next one,
    which reads as suspiciously familiar traffic.

    In deployment this matches the natural arrangement: one extractor per
    monitored device, created at enumeration.
    """

    WINDOW: float = 1.0    # seconds; plan §4.2
    STRIDE: float = 1.0    # tumbling

    def reset(self) -> None:
        """Forget per-session state. Called between device sessions."""

    @abstractmethod
    def extract(
        self,
        events: Sequence[URBEvent],
        t0: float,
        descriptor: DeviceDescriptor,
    ) -> np.ndarray:
        """Return float32[N_FEATURES] for the window ``[t0, t0 + WINDOW)``.

        An empty ``events`` sequence is legitimate input, not an error: a
        zero-traffic second is a real observation and every feature must still
        be finite (plan §4.2 acceptance criterion).
        """

    def extract_checked(
        self,
        events: Sequence[URBEvent],
        t0: float,
        descriptor: DeviceDescriptor,
    ) -> np.ndarray:
        v = self.extract(events, t0, descriptor)
        self.validate_vector(v)
        return v

    @staticmethod
    def validate_vector(v: np.ndarray) -> None:
        if v.shape != (N_FEATURES,):
            raise ContractError(f"expected shape ({N_FEATURES},), got {v.shape}")
        if v.dtype != FEATURE_DTYPE:
            raise ContractError(f"expected dtype {FEATURE_DTYPE.__name__}, got {v.dtype}")
        if not np.all(np.isfinite(v)):
            bad = [FEATURE_NAMES[i] for i in np.flatnonzero(~np.isfinite(v))]
            raise ContractError(f"non-finite features: {bad}")
