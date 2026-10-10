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
attack family it is trained on needs a HID interface -- keystroke injection is
impossible without one, and a HID report descriptor can declare a keyboard on any
HID interface, boot protocol or not. So a device with ANY HID interface is always
scored, whatever else it declares. A BadUSB cannot hide behind an extra network
interface; a network-adapter attack with no HID interface (e.g. traffic
hijacking) was never in the model's scope, and is now reported as such instead of
being "detected" by accident.
"""

from __future__ import annotations

from typing import Iterable

from ..contract import CLASS_HID, DeviceDescriptor

__all__ = ["ModelSupport", "interface_triples"]

Triple = tuple[int, int, int]


def interface_triples(descriptor: DeviceDescriptor) -> set[Triple]:
    """Every interface the device declares, at enumeration or later."""
    return {(i.bInterfaceClass, i.bInterfaceSubClass, i.bInterfaceProtocol)
            for i in descriptor.interfaces}


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
        classes = {c for c, _, _ in interface_triples(descriptor)}
        if classes & self.ALWAYS_SCORED:
            return True
        return bool(classes) and classes <= self.classes

    def uncovered(self, descriptor: DeviceDescriptor) -> list[Triple]:
        return sorted(t for t in interface_triples(descriptor) if t[0] not in self.classes)

    def to_json(self) -> list[list[int]]:
        return [list(t) for t in sorted(self.triples)]

    def __repr__(self) -> str:
        return f"ModelSupport(classes {sorted(self.classes)})"
