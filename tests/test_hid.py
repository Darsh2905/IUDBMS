"""HID report descriptors: what a device can send, and keystroke capability."""

import importlib.util
from pathlib import Path

from iudbms.capture.hid import usage_pages
from iudbms.contract import DeviceDescriptor, Endpoint, Interface
from iudbms.policy import ModelSupport, is_keyboard, may_send_keystrokes

# Boot keyboard (HID 1.11 Appendix B.1) and a QEMU-style absolute pointer, as bytes.
KEYBOARD = bytes.fromhex(
    "05010906a101050719e029e71500250175019508810295017508810195057501050819012905"
    "910295017503910195067508150025650507190029658100c0")
TABLET = bytes.fromhex(
    "05010902a101090105091901290315002501950375018102950175058101050109300931"
    "150026ff7f350046ff7f750f950281020501093815812579350045007508950181 06c0c0".replace(" ", ""))
CONSUMER = bytes.fromhex("050c0901a1011500250109e909ea750195028102950675018103c0")
EXTENDED = bytes.fromhex("0b06000100a101c0")         # 32-bit usage: page 0x0001, usage 0x0006


def test_usage_pages():
    assert usage_pages(KEYBOARD) == (0x01, 0x07, 0x08)
    assert 0x07 not in usage_pages(TABLET) and 0x01 in usage_pages(TABLET)
    assert usage_pages(CONSUMER) == (0x0C,)
    assert usage_pages(EXTENDED) == (0x01,)
    assert usage_pages(b"") == () and usage_pages(b"\x05") == ()     # truncated: no guess


def _dev(*ifaces):
    return DeviceDescriptor(bus=1, devnum=4, vid=1, pid=1, interfaces=tuple(ifaces))


def _hid(proto, pages):
    return Interface(3, 1 if proto else 0, proto, (Endpoint(0x81, "intr", 10),),
                     hid_usage_pages=pages)


SUP = ModelSupport([(3, 1, 1), (3, 1, 2), (8, 6, 0x50), (1, 2, 0)])


def test_a_pointer_with_a_known_descriptor_cannot_send_keystrokes():
    tablet = _dev(_hid(0, usage_pages(TABLET)))
    assert not may_send_keystrokes(tablet) and not is_keyboard(tablet)
    assert not SUP.covers(tablet) and "pointing device" in SUP.reason(tablet)


def test_keyboards_and_consumer_controls_are_always_judged():
    assert SUP.covers(_dev(_hid(1, usage_pages(KEYBOARD))))
    assert SUP.covers(_dev(_hid(0, usage_pages(CONSUMER))))          # can launch apps
    assert SUP.covers(_dev(Interface(255, 255, 255), _hid(0, usage_pages(KEYBOARD))))


def test_report_protocol_keyboard_is_a_keyboard():
    """The boot-protocol test missed this case; the descriptor does not."""
    assert is_keyboard(_dev(_hid(0, usage_pages(KEYBOARD))))


def test_unknown_descriptor_is_conservative():
    mouse = _dev(_hid(2, None))
    assert may_send_keystrokes(mouse) and SUP.covers(mouse)           # judged, as before
    assert not is_keyboard(mouse) and is_keyboard(_dev(_hid(1, None)))


def test_guest_script_parser_matches_the_package():
    spec = importlib.util.spec_from_file_location(
        "gd", Path(__file__).resolve().parent.parent / "testbed" / "guest_descriptors.py")
    src = spec.loader.get_source("gd")
    ns: dict = {}
    exec(src.split("out = {}")[0], ns)                 # definitions only; do not walk sysfs

    class FakeDir:
        def __init__(self, data):
            self.data = data

        def glob(self, _):
            data = self.data

            class F:
                def read_bytes(self):
                    return data
            return [F()]
    for d in (KEYBOARD, TABLET, CONSUMER, EXTENDED):
        assert tuple(ns["hid_pages"](FakeDir(d))) == usage_pages(d)
