"""
L1 — descriptor snapshot from sysfs.

usbmon tells you what a device *does*; ``/sys/bus/usb/devices/`` tells you what
it *claims to be*. The entire thesis is that these can disagree, so they are
collected separately and joined only at the feature layer.

Layout::

    /sys/bus/usb/devices/1-2/            device      idVendor, idProduct, ...
    /sys/bus/usb/devices/1-2:1.0/        interface   bInterfaceClass, ...
    /sys/bus/usb/devices/1-2:1.0/ep_81/  endpoint    bEndpointAddress, ...

Interface directories appear *as the host enumerates them*, so re-scanning
periodically is how the ``identity_switch`` attack class (a storage device that
later grows an HID interface) becomes visible. :func:`scan` records
``first_seen_ts`` for interfaces it has not seen before, which is what feeds
``DeviceDescriptor.visible_at``.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..contract import DeviceDescriptor, Endpoint, Interface, TransferType

__all__ = ["SYSFS_USB_DEVICES", "SysfsScanner", "scan_once"]

SYSFS_USB_DEVICES = Path("/sys/bus/usb/devices")

#: bmAttributes transfer-type bits (USB 2.0 §9.6.6, bits 1..0)
_XFER = {0: "control", 1: "iso", 2: "bulk", 3: "intr"}


def _read(base: Path, name: str) -> str | None:
    try:
        return (base / name).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(base: Path, name: str, radix: int = 16) -> int | None:
    raw = _read(base, name)
    if raw is None:
        return None
    try:
        return int(raw, radix)
    except ValueError:
        return None


def _endpoints(iface_dir: Path) -> tuple[Endpoint, ...]:
    eps: list[Endpoint] = []
    for ep_dir in sorted(iface_dir.glob("ep_*")):
        addr = _read_int(ep_dir, "bEndpointAddress")
        attrs = _read_int(ep_dir, "bmAttributes")
        if addr is None or attrs is None:
            continue
        xfer: TransferType = _XFER.get(attrs & 0x03, "control")  # type: ignore[assignment]
        eps.append(Endpoint(addr=addr, type=xfer, interval=_read_int(ep_dir, "bInterval", 10)))
    return tuple(eps)


class SysfsScanner:
    """Repeatedly snapshots sysfs, remembering when each interface first appeared.

    ``first_seen_ts`` is seconds relative to ``origin`` (default: scanner
    construction), matching the URBEvent timebase.
    """

    def __init__(self, root: Path = SYSFS_USB_DEVICES, origin: float | None = None):
        self.root = Path(root)
        self.origin = time.monotonic() if origin is None else origin
        self._first_seen: dict[tuple[str, str], float] = {}

    def _now(self) -> float:
        return time.monotonic() - self.origin

    def scan(self) -> dict[str, DeviceDescriptor]:
        """Snapshot every currently attached USB device, keyed by ``bus-devnum``."""
        out: dict[str, DeviceDescriptor] = {}
        if not self.root.exists():
            return out
        now = self._now()

        for dev_dir in sorted(self.root.iterdir()):
            name = dev_dir.name
            # Skip interface dirs ("1-2:1.0") and root hubs ("usb1").
            if ":" in name or name.startswith("usb") or not dev_dir.is_dir():
                continue
            busnum = _read_int(dev_dir, "busnum", 10)
            devnum = _read_int(dev_dir, "devnum", 10)
            vid = _read_int(dev_dir, "idVendor")
            pid = _read_int(dev_dir, "idProduct")
            if None in (busnum, devnum, vid, pid):
                continue

            # Interface dirs are reachable two ways and both occur in the wild:
            # as children of the resolved device dir (/sys/devices/.../1-2/1-2:1.0)
            # and as siblings under /sys/bus/usb/devices/ (1-2:1.0). Check both,
            # de-duplicated by name, so the scanner does not depend on whether
            # the entry it was handed is a symlink or a real directory.
            iface_dirs = {d.name: d for d in dev_dir.glob(f"{name}:*") if d.is_dir()}
            for d in self.root.glob(f"{name}:*"):
                if d.is_dir():
                    iface_dirs.setdefault(d.name, d)

            ifaces: list[Interface] = []
            for iface_dir in [iface_dirs[k] for k in sorted(iface_dirs)]:
                cls = _read_int(iface_dir, "bInterfaceClass")
                if cls is None:
                    continue
                key = (name, iface_dir.name)
                first = self._first_seen.setdefault(key, now)
                ifaces.append(Interface(
                    bInterfaceClass=cls,
                    bInterfaceSubClass=_read_int(iface_dir, "bInterfaceSubClass") or 0,
                    bInterfaceProtocol=_read_int(iface_dir, "bInterfaceProtocol") or 0,
                    endpoints=_endpoints(iface_dir),
                    first_seen_ts=first,
                ))

            desc = DeviceDescriptor(
                bus=busnum, devnum=devnum, vid=vid, pid=pid,          # type: ignore[arg-type]
                bDeviceClass=_read_int(dev_dir, "bDeviceClass") or 0,
                serial_present=_read(dev_dir, "serial") is not None,
                manufacturer=_read(dev_dir, "manufacturer"),
                product=_read(dev_dir, "product"),
                interfaces=tuple(ifaces),
            )
            out[desc.device_key] = desc
        return out


def scan_once(root: Path = SYSFS_USB_DEVICES) -> dict[str, DeviceDescriptor]:
    """One-shot scan; every interface gets ``first_seen_ts = 0.0``."""
    return SysfsScanner(root=root, origin=time.monotonic()).scan()
