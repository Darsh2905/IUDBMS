#!/usr/bin/env python3
"""
Record a real session on YOUR OWN Linux machine, with physical USB devices, for
the IUDBMS evaluation -- without your keystrokes ever touching disk.

    sudo python3 scripts/field_capture.py --minutes 60 --alias tester1
    # use the machine normally; Ctrl-C ends early. Produces iudbms-field-<alias>-<stamp>.tar.gz

What is recorded:
  * the kernel's usbmon text for every USB device, with every captured payload
    replaced AS IT IS READ by a keyed, length-preserving permutation
    (iudbms.storage.traces.PayloadPseudonymizer). Equal payloads stay equal and
    lengths are kept, so every feature the detector computes is exact; the key is
    drawn from os.urandom for this run and never written anywhere;
  * a sysfs descriptor snapshot at the start and every 30 s (so a device you plug in
    later is described), and labels.json marking every device benign;
  * kernel version, distribution, and the driver bound to each interface.

What is NOT protected: pseudonymous is not anonymous. The mapping is consistent for
the whole run, so a long typing session is a substitution cipher over keystrokes and
open to frequency analysis. Use --no-payload to drop payload bytes entirely (three
payload features then cannot be computed). Record only devices you own, on a machine
you control, and share the archive only with people you would trust with the
pseudonymised data.

Score it on any machine:  python3 -m testbed.score <unpacked directory>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

USBMON = Path("/sys/kernel/debug/usb/usbmon/0u")
_WORD = re.compile(r"^[0-9a-fA-F]+$")


def pseudonymise_line(line: str, pz) -> str:
    """Replace the data words after '=' with their keyed permutation, same grouping."""
    parts = line.rstrip("\n").split(" ")
    try:
        k = parts.index("=")
    except ValueError:
        return line if line.endswith("\n") else line + "\n"
    words = parts[k + 1:]
    if not words or not all(_WORD.match(w) and len(w) % 2 == 0 for w in words):
        return line if line.endswith("\n") else line + "\n"
    raw = bytes.fromhex("".join(words))
    mapped = pz(raw).hex() if pz is not None else ""
    if pz is None:                                  # --no-payload: keep the tag, drop the bytes
        return " ".join(parts[:k]) + " -\n"
    out, i = [], 0
    for w in words:
        out.append(mapped[i:i + len(w)])
        i += len(w)
    return " ".join(parts[:k + 1] + out) + "\n"


def snapshot() -> dict:
    # the same sysfs walk the testbed guests use (testbed/guest_descriptors.py)
    r = subprocess.run([sys.executable, str(ROOT / "testbed" / "guest_descriptors.py")],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def env_info() -> dict:
    ifaces = {}
    for d in Path("/sys/bus/usb/devices").glob("*:*"):
        try:
            cls = (d / "bInterfaceClass").read_text().strip()
        except OSError:
            continue
        drv = (d / "driver").resolve().name if (d / "driver").exists() else None
        ifaces[d.name] = {"class": cls, "driver": drv}
    osr = {}
    try:
        for ln in Path("/etc/os-release").read_text().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                osr[k] = v.strip('"')
    except OSError:
        pass
    return {"kernel": platform.release(), "os": osr.get("PRETTY_NAME"), "interfaces": ifaces,
            "python": platform.python_version()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--alias", required=True, help="a pseudonym for you, not your name")
    ap.add_argument("--no-payload", action="store_true", help="drop payload bytes entirely")
    ap.add_argument("--out", type=Path, default=Path.cwd())
    a = ap.parse_args()
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise SystemExit("run on Linux as root: sudo python3 scripts/field_capture.py ...")
    subprocess.run(["modprobe", "usbmon"], check=False)
    if not Path("/sys/kernel/debug/usb").exists():
        subprocess.run(["mount", "-t", "debugfs", "none", "/sys/kernel/debug"], check=False)
    if not USBMON.exists():
        raise SystemExit(f"{USBMON} missing: is usbmon available (CONFIG_USB_MON)?")

    from iudbms.storage.traces import PayloadPseudonymizer
    from testbed.campaign import device_role
    pz = None if a.no_payload else PayloadPseudonymizer(os.urandom(32))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rd = a.out / f"iudbms-field-{a.alias}-{stamp}"
    rd.mkdir(parents=True)
    print(f"recording {a.minutes:.0f} min to {rd}  (payloads "
          f"{'dropped' if pz is None else 'pseudonymised, key in memory only'}); Ctrl-C ends early")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    snaps = [snapshot()]

    def snapper():
        while not stop.wait(30.0):
            try:
                snaps.append(snapshot())
            except Exception as e:                     # a failed snapshot must not end the run
                print(f"  snapshot failed: {e}", file=sys.stderr)
    threading.Thread(target=snapper, daemon=True).start()

    deadline = time.time() + a.minutes * 60
    n = 0
    with open(USBMON, "r", errors="replace") as src, open(rd / "usbmon.txt", "w") as dst:
        reader_done = threading.Event()

        def pump():
            nonlocal n
            for line in src:
                dst.write(pseudonymise_line(line, pz))
                n += 1
                if stop.is_set():
                    break
            reader_done.set()
        threading.Thread(target=pump, daemon=True).start()
        while time.time() < deadline and not stop.is_set():
            time.sleep(1.0)
        stop.set()
        time.sleep(1.5)
    snaps.append(snapshot())
    merged: dict = {}
    for s in snaps:
        merged.update(s)
    (rd / "descriptors.json").write_text(json.dumps(merged, indent=1))
    (rd / "labels.json").write_text(json.dumps(
        {k: {"role": device_role(d), "label": 0, "product": d.get("product")}
         for k, d in merged.items()}, indent=1))
    (rd / "env.json").write_text(json.dumps(env_info(), indent=1))
    (rd / "workload.json").write_text(json.dumps(
        {"distro": env_info().get("os"), "kind": "field", "seed": None, "minutes": a.minutes,
         "keys_typed": None, "mouse_events": 0, "tablet_events": 0, "thread_errors": [],
         "alias": a.alias, "payload_mode": "none" if pz is None else "pseudonymous",
         "log": []}, indent=1))
    tar = a.out / f"{rd.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(rd, arcname=rd.name)
    print(f"{n:,} usbmon lines, {len(merged)} devices -> {tar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
