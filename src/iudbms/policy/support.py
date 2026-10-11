"""
Model support: which declared device types a trained model may judge.

A supervised model trained on keyboards, mice, storage, webcams and audio has no
basis for a verdict about a USB network adapter. On a real kernel it gives one
anyway: a CDC-Ethernet adapter's link-status endpoint sends an identical 8-byte
interrupt report every 32 ms -- perfectly regular, perfectly repetitive -- which
is the keystroke-injection signature, and every one of its windows scored over
threshold (CLAUDE.md finding 37). That is extrapolation, not detection.

``ModelSupport`` records the interface triples (class, subclass, protocol) of the
benign devices a model was trained on, and judges coverage by **interface class**.
A device declaring a class outside that set is **unmodelled**: it is still scored
and stored, but the policy engine never escalates on it and reports its state as
``unmodelled`` so an operator can apply policy (allow or deny the device type).

**Abstention must never remove coverage of a threat the model handles.** Every
attack family it is trained on sends keystrokes, and keystrokes reach the host only
through a HID interface whose report descriptor declares the Keyboard/Keypad usage
page (consumer controls can launch applications, so they count too; capture.hid).
So a device that MAY send keystrokes is always judged, whatever else it declares --
and "may" is conservative: a HID interface whose report descriptor is unknown counts.
A BadUSB cannot hide behind an extra network interface. A pure pointer (mouse,
tablet, pen) whose descriptor declares no keyboard or consumer usage cannot inject
keystrokes and is outside this threat model; on a real kernel a VM tablet's 6-byte
absolute reports, a shape the training set never contained, alarmed in every session
(finding 37). A network-adapter attack with no HID interface was never in scope, and
is now reported as such instead of being "detected" by accident.
"""

from __future__ import annotations

from typing import Iterable

from ..capture.hid import KEYSTROKE_PAGES
from ..contract import CLASS_HID, DeviceDescriptor

__all__ = ["ModelSupport", "interface_triples", "may_send_keystrokes", "is_keyboard"]

Triple = tuple[int, int, int]


def interface_triples(descriptor: DeviceDescriptor) -> set[Triple]:
    """Every interface the device declares, at enumeration or later."""
    return {(i.bInterfaceClass, i.bInterfaceSubClass, i.bInterfaceProtocol)
            for i in descriptor.interfaces}


def may_send_keystrokes(descriptor: DeviceDescriptor) -> bool:
    """A HID interface that declares keyboard/consumer usages, or whose descriptor is unknown."""
    return any(i.bInterfaceClass == CLASS_HID and (
        i.hid_usage_pages is None or bool(set(i.hid_usage_pages) & KEYSTROKE_PAGES))
        for i in descriptor.interfaces)


def is_keyboard(descriptor: DeviceDescriptor) -> bool:
    """Declares the Keyboard/Keypad page, or (descriptor unknown) the boot-keyboard protocol.

    What the human-typing bound (live.detector, eval.analysis_hybrid) applies to. Using
    the report descriptor covers report-protocol keyboards, which the boot-protocol test
    alone missed.
    """
    for i in descriptor.interfaces:
        if i.bInterfaceClass != CLASS_HID:
            continue
        if i.hid_usage_pages is not None:
            if 0x07 in i.hid_usage_pages:
                return True
        elif i.bInterfaceProtocol == 1:
            return True
    return False


class ModelSupport:
    """Coverage is decided by interface CLASS, not by the exact triple.

    The exact triple is too strict for a simulated training set: real USB audio
    devices always declare an AudioControl interface (1,1,0) beside streaming, and
    the simulator's audio device declares streaming only, so triple matching would
    call every real sound card unmodelled because of a simulator omission (finding
    37). "Trained on audio-class devices" is the claim the training set supports.
    """

    #: Interface classes that are always scored, whatever the training set held.
    ALWAYS_SCORED = frozenset({CLASS_HID})

    def __init__(self, triples: Iterable[Iterable[int]]):
        self.triples: frozenset[Triple] = frozenset(tuple(int(v) for v in t) for t in triples)
        if any(len(t) != 3 for t in self.triples):
            raise ValueError("support entries are (class, subclass, protocol) triples")
        self.classes = frozenset(c for c, _, _ in self.triples)

    @classmethod
    def from_descriptors(cls, descriptors: Iterable[DeviceDescriptor]) -> "ModelSupport":
        out: set[Triple] = set()
        for d in descriptors:
            out |= interface_triples(d)
        return cls(out)

    def covers(self, descriptor: DeviceDescriptor) -> bool:
        if may_send_keystrokes(descriptor):
            return True                                     # never abstain on keystrokes
        other = {i.bInterfaceClass for i in descriptor.interfaces
                 if i.bInterfaceClass not in self.ALWAYS_SCORED}
        return bool(other) and other <= self.classes

    def uncovered(self, descriptor: DeviceDescriptor) -> list[Triple]:
        return sorted(t for t in interface_triples(descriptor)
                      if t[0] not in self.classes and t[0] not in self.ALWAYS_SCORED)

    def reason(self, descriptor: DeviceDescriptor) -> str:
        """Why a device is not judged (for the operator notice)."""
        bad = self.uncovered(descriptor)
        if bad:
            return f"declares interface types the model was not trained on {bad}"
        return ("a pointing device: its HID report descriptor declares no keyboard or "
                "consumer usages, so it cannot inject keystrokes")

    def to_json(self) -> list[list[int]]:
        return [list(t) for t in sorted(self.triples)]

    def __repr__(self) -> str:
        return f"ModelSupport(classes {sorted(self.classes)})"
