"""Model support: abstain on device types the model never saw, never on HID."""

import json
from pathlib import Path

import pytest

from iudbms.contract import DeviceDescriptor, Endpoint, Interface
from iudbms.policy import ModelSupport, PolicyEngine

ROOT = Path(__file__).resolve().parent.parent


def dev(*triples, late=None):
    ifs = [Interface(c, s, p, (Endpoint(0x81, "intr", 10),)) for c, s, p in triples]
    if late is not None:
        c, s, p = late
        ifs.append(Interface(c, s, p, (Endpoint(0x83, "intr", 10),), first_seen_ts=20.0))
    return DeviceDescriptor(bus=1, devnum=5, vid=0x1234, pid=0x5678, interfaces=tuple(ifs))


TRAINED = ModelSupport([(3, 1, 1), (3, 1, 2), (8, 6, 0x50), (1, 2, 0), (0x0E, 1, 0)])
KEYBOARD, NIC = dev((3, 1, 1)), dev((2, 6, 0), (10, 0, 0))


def test_trained_types_are_covered_and_others_are_not():
    assert TRAINED.covers(KEYBOARD)
    assert TRAINED.covers(dev((8, 6, 0x50)))
    assert not TRAINED.covers(NIC)
    assert TRAINED.covers(dev((8, 6, 0x62)))               # UAS: same class as trained storage
    assert TRAINED.covers(dev((1, 1, 0), (1, 2, 0)))       # real audio: control + streaming
    assert not TRAINED.covers(dev((255, 255, 255)))        # vendor-specific serial converter
    assert not TRAINED.covers(dev((11, 0, 0)))             # smart-card reader
    assert TRAINED.uncovered(NIC) == [(2, 6, 0), (10, 0, 0)]


@pytest.mark.parametrize("d", [
    dev((3, 0, 0)),                        # report-protocol HID can still declare a keyboard
    dev((2, 6, 0), (10, 0, 0), (3, 1, 1)),  # a NIC that also carries a keyboard
    dev((255, 255, 255), (3, 0, 0)),        # vendor-specific + HID
    dev((255, 255, 255), late=(3, 1, 1)),   # identity switch from an untrained class
])
def test_any_hid_interface_is_always_judged(d):
    """Abstention must never remove coverage of keystroke injection."""
    assert TRAINED.covers(d)


def test_an_unmodelled_device_never_escalates_however_high_it_scores():
    eng = PolicyEngine(threshold=0.5, k=4, n=6, support=TRAINED)
    ctx = eng.new_device("nic")
    assert all(eng.step(ctx, float(t), 1.0, 1.0, [], NIC) is None for t in range(30))
    assert ctx.state == "unmodelled"


def test_a_covered_device_escalates_exactly_as_before():
    with_support = PolicyEngine(threshold=0.5, k=4, n=6, support=TRAINED)
    without = PolicyEngine(threshold=0.5, k=4, n=6)
    a, b = with_support.new_device("k"), without.new_device("k")
    seq = [0.1, 0.9, 0.9, 0.2] + [0.9] * 14
    ra = [bool(with_support.step(a, t, 0, s, [], KEYBOARD)) for t, s in enumerate(seq)]
    rb = [bool(without.step(b, t, 0, s, [], KEYBOARD)) for t, s in enumerate(seq)]
    assert ra == rb and any(ra) and a.state == b.state == "revoked"


def test_no_support_record_judges_every_device():
    eng = PolicyEngine(threshold=0.5, k=4, n=6)
    ctx = eng.new_device("nic")
    fired = [eng.step(ctx, float(t), 1.0, 1.0, [], NIC) for t in range(6)]
    assert any(fired)


def test_round_trip():
    assert ModelSupport(TRAINED.to_json()).triples == TRAINED.triples


def test_the_deployed_set_records_its_training_support():
    cfg = json.loads((ROOT / "out" / "models" / "deploy.json").read_text())
    sup = ModelSupport(cfg["support"])
    assert sup.covers(KEYBOARD) and not sup.covers(NIC)
