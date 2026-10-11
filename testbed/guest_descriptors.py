"""Guest side: print a sysfs descriptor snapshot as JSON (same logic as scripts/vm_capture.sh).

Kept byte-compatible with the snapshot that iudbms.live.cli._load_descriptor_snapshot reads.
Runs under any python3 >= 3.6 on the guest; standard library only.
"""
import json, sys
from pathlib import Path

root = Path("/sys/bus/usb/devices")

def rd(p, n, radix=16):
    try:
        v = (p / n).read_text().strip()
    except OSError:
        return None
    try:
        return int(v, radix)
    except ValueError:
        return v

def hid_pages(iface_dir):
    """Usage pages of the interface's HID report descriptor (iudbms.capture.hid, inlined:
    this script runs standalone in guests). None if the interface exposes none."""
    for rd in sorted(iface_dir.glob("*:*:*.*/report_descriptor")):
        try:
            desc = rd.read_bytes()
        except OSError:
            return None
        out, i = [], 0
        while i < len(desc):
            b = desc[i]
            if b == 0xFE:
                if i + 2 >= len(desc):
                    break
                i += 3 + desc[i + 1]
                continue
            size = (0, 1, 2, 4)[b & 3]
            typ, tag = (b >> 2) & 3, (b >> 4) & 15
            data = desc[i + 1:i + 1 + size]
            if len(data) < size:
                break
            val = int.from_bytes(data, "little") if size else 0
            page = val & 0xFFFF if (typ == 1 and tag == 0) else (
                val >> 16 if (typ == 2 and tag == 0 and size == 4) else None)
            if page is not None and page not in out:
                out.append(page)
            i += 1 + size
        return out
    return None

out = {}
for dev in sorted(root.iterdir()):
    if ":" in dev.name or dev.name.startswith("usb") or not dev.is_dir():
        continue
    bus, num = rd(dev, "busnum", 10), rd(dev, "devnum", 10)
    vid, pid = rd(dev, "idVendor"), rd(dev, "idProduct")
    if None in (bus, num, vid, pid):
        continue
    ifaces = []
    for i in sorted(root.glob(f"{dev.name}:*")):
        cls = rd(i, "bInterfaceClass")
        if cls is None:
            continue
        eps = []
        for ep in sorted(i.glob("ep_*")):
            a, at = rd(ep, "bEndpointAddress"), rd(ep, "bmAttributes")
            if a is not None and at is not None:
                eps.append({"addr": a, "bmAttributes": at,
                            "bInterval": rd(ep, "bInterval", 10)})
        ifaces.append({"bInterfaceClass": cls,
                       "bInterfaceSubClass": rd(i, "bInterfaceSubClass") or 0,
                       "bInterfaceProtocol": rd(i, "bInterfaceProtocol") or 0,
                       "endpoints": eps,
                       "hid_usage_pages": hid_pages(i) if cls == 3 else None})
    out[f"{bus}-{num}"] = {
        "bus": bus, "devnum": num, "vid": vid, "pid": pid,
        "bDeviceClass": rd(dev, "bDeviceClass") or 0,
        "serial_present": (dev / "serial").exists(),
        "manufacturer": rd(dev, "manufacturer"), "product": rd(dev, "product"),
        "interfaces": ifaces,
    }

print(json.dumps(out))
