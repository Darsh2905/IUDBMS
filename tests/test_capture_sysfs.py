"""Milestone 2 — sysfs descriptor scanner.

Builds a fake /sys/bus/usb/devices tree so the scanner is testable off Linux.
The interesting case is a device that grows an interface between scans: that is
the identity_switch attack class, and it must surface as a later
``first_seen_ts`` rather than as a silently changed descriptor.
"""

from iudbms.capture.sysfs import SysfsScanner, scan_once


def _mkdev(root, name, *, busnum, devnum, vid, pid, serial=None,
           manufacturer=None, product=None, bDeviceClass=0):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "busnum").write_text(f"{busnum}\n")
    (d / "devnum").write_text(f"{devnum}\n")
    (d / "idVendor").write_text(f"{vid:04x}\n")
    (d / "idProduct").write_text(f"{pid:04x}\n")
    (d / "bDeviceClass").write_text(f"{bDeviceClass:02x}\n")
    if serial is not None:
        (d / "serial").write_text(serial + "\n")
    if manufacturer is not None:
        (d / "manufacturer").write_text(manufacturer + "\n")
    if product is not None:
        (d / "product").write_text(product + "\n")
    return d


def _mkiface(root, dev_name, cfg, num, *, cls, sub=0, proto=0, endpoints=()):
    d = root / f"{dev_name}:{cfg}.{num}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "bInterfaceClass").write_text(f"{cls:02x}\n")
    (d / "bInterfaceSubClass").write_text(f"{sub:02x}\n")
    (d / "bInterfaceProtocol").write_text(f"{proto:02x}\n")
    for addr, attrs, interval in endpoints:
        ep = d / f"ep_{addr:02x}"
        ep.mkdir(exist_ok=True)
        (ep / "bEndpointAddress").write_text(f"{addr:02x}\n")
        (ep / "bmAttributes").write_text(f"{attrs:02x}\n")
        (ep / "bInterval").write_text(f"{interval}\n")
    return d


def test_missing_sysfs_returns_empty(tmp_path):
    assert scan_once(tmp_path / "nope") == {}


def test_scans_a_keyboard(tmp_path):
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=0x046D, pid=0xC31C,
           serial="ABC123", manufacturer="Logitech", product="Keyboard")
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x03, sub=0x01, proto=0x01,
             endpoints=[(0x81, 0x03, 10)])          # bmAttributes 0x03 = interrupt

    devs = scan_once(tmp_path)
    assert set(devs) == {"1-4"}
    d = devs["1-4"]
    assert (d.vid, d.pid) == (0x046D, 0xC31C)
    assert d.serial_present and d.manufacturer == "Logitech"
    assert len(d.interfaces) == 1
    iface = d.interfaces[0]
    assert iface.bInterfaceClass == 0x03
    assert len(iface.endpoints) == 1
    ep = iface.endpoints[0]
    assert ep.type == "intr" and ep.number == 1 and ep.direction == "in" and ep.interval == 10


def test_endpoint_transfer_types_from_bmattributes(tmp_path):
    _mkdev(tmp_path, "1-3", busnum=1, devnum=5, vid=0x0781, pid=0x5583)
    _mkiface(tmp_path, "1-3", 1, 0, cls=0x08,
             endpoints=[(0x81, 0x02, 0), (0x02, 0x02, 0), (0x83, 0x01, 1), (0x00, 0x00, 0)])
    d = scan_once(tmp_path)["1-5"]
    by_addr = {e.addr: e.type for e in d.interfaces[0].endpoints}
    assert by_addr == {0x00: "control", 0x02: "bulk", 0x81: "bulk", 0x83: "iso"}


def test_missing_serial_reported_absent(tmp_path):
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x03)
    assert scan_once(tmp_path)["1-4"].serial_present is False


def test_root_hubs_and_interface_dirs_are_not_devices(tmp_path):
    _mkdev(tmp_path, "usb1", busnum=1, devnum=1, vid=0x1D6B, pid=0x0002)
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x03)
    assert set(scan_once(tmp_path)) == {"1-4"}, "root hub must be skipped"


def test_late_interface_gets_a_later_first_seen_ts(tmp_path):
    """identity_switch: a storage device that later exposes an HID interface."""
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=0x0781, pid=0x5583)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x08, endpoints=[(0x81, 0x02, 0)])

    scanner = SysfsScanner(root=tmp_path, origin=0.0)
    first = scanner.scan()["1-4"]
    assert first.classes_at(1e9) == frozenset({0x08})
    # The old assertion here was `not has_late_interface() or ts >= 0`, which is true
    # whatever the answer; it hid the defect tested properly below.
    assert not first.has_late_interface()

    _mkiface(tmp_path, "1-2", 1, 1, cls=0x03, endpoints=[(0x83, 0x03, 10)])
    second = scanner.scan()["1-4"]

    by_class = {i.bInterfaceClass: i for i in second.interfaces}
    assert set(by_class) == {0x08, 0x03}
    assert by_class[0x03].first_seen_ts > by_class[0x08].first_seen_ts, (
        "the HID interface arrived later and must be timestamped later"
    )
    assert second.has_late_interface()
    # and the descriptor must report it as invisible before it arrived
    assert second.classes_at(by_class[0x03].first_seen_ts - 1e-6) == frozenset({0x08})


def test_first_seen_is_stable_across_rescans(tmp_path):
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x03)
    scanner = SysfsScanner(root=tmp_path, origin=0.0)
    a = scanner.scan()["1-4"].interfaces[0].first_seen_ts
    b = scanner.scan()["1-4"].interfaces[0].first_seen_ts
    assert a == b, "re-scanning must not reset an interface's arrival time"


def test_unreadable_device_dir_is_skipped_not_fatal(tmp_path):
    d = _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    (d / "idVendor").unlink()
    _mkdev(tmp_path, "1-3", busnum=1, devnum=5, vid=3, pid=4)
    _mkiface(tmp_path, "1-3", 1, 0, cls=0x03)
    assert set(scan_once(tmp_path)) == {"1-5"}


def test_interfaces_present_at_first_sight_are_not_late(tmp_path):
    """A device already attached when the scanner starts did not grow an interface.

    first_seen_ts > 0 means "this interface appeared after the device did", and the
    model treats that as the strongest attack indicator it has (identity_switch). The
    scanner used to stamp every interface with seconds-since-scanner-start, so on a
    real host EVERY device carried late_iface_flag = 1 in every window and was
    classed 'other' (classes_at(0.0) excluded its interfaces). The simulator always
    emits first_seen_ts = 0.0 for enumeration-time interfaces, so nothing in the
    evaluation could see it; it surfaced as late_iface_flag = 1.0 on every window of
    every device on the VM.
    """
    import time
    from iudbms.stage1 import class_group
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x03, endpoints=[(0x81, 0x03, 10)])
    # a scanner created long before its first scan: now = monotonic() - origin >> 0
    scanner = SysfsScanner(root=tmp_path, origin=time.monotonic() - 500.0)
    d = scanner.scan()["1-4"]
    assert d.interfaces[0].first_seen_ts == 0.0
    assert not d.has_late_interface()
    assert class_group(d) == "hid", "an interface present at first sight is an enumeration-time one"


def test_a_device_plugged_in_later_is_not_a_late_interface(tmp_path):
    """Hot-plug: the device is new, so its interfaces are its enumeration, not additions."""
    import time
    scanner = SysfsScanner(root=tmp_path, origin=time.monotonic() - 100.0)
    assert scanner.scan() == {}
    _mkdev(tmp_path, "1-3", busnum=1, devnum=5, vid=1, pid=2)
    _mkiface(tmp_path, "1-3", 1, 0, cls=0x03)
    d = scanner.scan()["1-5"]
    assert d.interfaces[0].first_seen_ts == 0.0 and not d.has_late_interface()


def test_an_interface_added_to_a_known_device_is_late_on_the_shared_clock(tmp_path):
    """The one real late arrival must still be caught, and stamped on the monitor's clock."""
    import time
    _mkdev(tmp_path, "1-2", busnum=1, devnum=4, vid=1, pid=2)
    _mkiface(tmp_path, "1-2", 1, 0, cls=0x08)
    scanner = SysfsScanner(root=tmp_path, origin=time.monotonic() - 40.0)
    scanner.scan()
    _mkiface(tmp_path, "1-2", 1, 1, cls=0x03)
    d = scanner.scan()["1-4"]
    late = next(i for i in d.interfaces if i.bInterfaceClass == 0x03)
    assert 39.0 < late.first_seen_ts < 42.0       # seconds since `origin`, the window clock
    assert d.has_late_interface()
