#!/usr/bin/env bash
# Record a real usbmon trace on a Linux box (the project VM) for replay
# elsewhere. Produces a text capture plus a sysfs descriptor snapshot, which
# together are everything UsbmonSource needs to replay the session on any host.
#
#   sudo ./scripts/vm_capture.sh [seconds] [bus] [outdir]
#
# Then copy the output back and commit it under tests/fixtures/usbmon/:
#   scp user@vm:/tmp/iudbms-capture/*.{txt,json} tests/fixtures/usbmon/
#
# Do something with the USB devices while it runs — type on the keyboard, copy
# a file to a USB stick. An idle capture is a valid but uninteresting trace.
set -euo pipefail

SECONDS_TO_CAPTURE="${1:-60}"
BUS="${2:-0}"                      # 0 = all buses
OUTDIR="${3:-/tmp/iudbms-capture}"
STAMP="$(date +%Y%m%d-%H%M%S)"

[[ "$(uname -s)" == "Linux" ]] || { echo "This must run on Linux (the VM), not $(uname -s)." >&2; exit 1; }
[[ "${EUID}" -eq 0 ]] || { echo "Needs root: sudo $0 $*" >&2; exit 1; }

mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
modprobe usbmon 2>/dev/null || true

NODE="/sys/kernel/debug/usb/usbmon/${BUS}u"
[[ -r "$NODE" ]] || { echo "Cannot read $NODE. Is usbmon loaded?" >&2; exit 1; }

mkdir -p "$OUTDIR"
CAP="$OUTDIR/capture-${STAMP}-bus${BUS}.txt"
DESC="$OUTDIR/descriptors-${STAMP}.json"

# --- descriptor snapshot (what each device CLAIMS to be) ---------------------
python3 - "$DESC" <<'PY'
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
                       "endpoints": eps})
    out[f"{bus}-{num}"] = {
        "bus": bus, "devnum": num, "vid": vid, "pid": pid,
        "bDeviceClass": rd(dev, "bDeviceClass") or 0,
        "serial_present": (dev / "serial").exists(),
        "manufacturer": rd(dev, "manufacturer"), "product": rd(dev, "product"),
        "interfaces": ifaces,
    }

Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print(f"  {len(out)} device(s) -> {sys.argv[1]}")
PY

# --- the capture -------------------------------------------------------------
echo "  capturing bus ${BUS} for ${SECONDS_TO_CAPTURE}s -> ${CAP}"
echo "  (use your USB devices now: type, move the mouse, copy a file)"
timeout "${SECONDS_TO_CAPTURE}" cat "$NODE" > "$CAP" || true

LINES=$(wc -l < "$CAP")
echo
echo "  captured ${LINES} lines"
[[ "$LINES" -gt 0 ]] || echo "  WARNING: empty capture — was any USB device active?"
echo "  copy back with:"
echo "    scp $(whoami)@<vm>:${OUTDIR}/'*'-${STAMP}*.{txt,json} tests/fixtures/usbmon/"
