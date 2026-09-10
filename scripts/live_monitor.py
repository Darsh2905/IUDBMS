#!/usr/bin/env python3
"""
Live USB monitoring — read the kernel, score devices as they act.

    sudo modprobe usbmon
    sudo python3 scripts/live_monitor.py --bus 0

Requires **Linux** with debugfs mounted and read access to
``/sys/kernel/debug/usb/usbmon/<bus>u`` (root, normally). Bus 0 is usbmon's
"all buses" node.

This is the only entry point that touches live traffic. `live_replay.py` drives
the same detector from a recorded capture or a simulated session; this one opens
the kernel's log and never stops until interrupted.

Enforcement stays dry-run: actions are generated and printed, and nothing is
written to /sys/bus/usb/…/authorized. That gate is
`policy.engine.ENFORCEMENT_REVIEWED`, which is still False.
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iudbms.capture.sysfs import SysfsScanner  # noqa: E402
from iudbms.capture.usbmon import DEFAULT_USBMON_DIR, UsbmonSource  # noqa: E402
from iudbms.live import LiveMonitor, StreamingDetector  # noqa: E402
from iudbms.policy import ENFORCEMENT_REVIEWED, PolicyEngine  # noqa: E402
from iudbms.stage1 import Stage1  # noqa: E402
from iudbms.stage2 import LSTMDetector  # noqa: E402

MODELS = ROOT / "out" / "models"


def _load_descriptors(path):
    """Rebuild DeviceDescriptors from a vm_capture.sh snapshot."""
    if path is None or not Path(path).exists():
        return {}
    from iudbms.contract import DeviceDescriptor, Endpoint, Interface
    xfer = {0: "control", 1: "iso", 2: "bulk", 3: "intr"}
    out = {}
    for key, d in json.loads(Path(path).read_text()).items():
        ifaces = []
        for i in d.get("interfaces", []):
            eps = tuple(Endpoint(addr=e["addr"], type=xfer.get(e["bmAttributes"] & 3, "control"),
                                 interval=e.get("bInterval"))
                        for e in i.get("endpoints", []))
            ifaces.append(Interface(bInterfaceClass=i["bInterfaceClass"],
                                    bInterfaceSubClass=i.get("bInterfaceSubClass", 0),
                                    bInterfaceProtocol=i.get("bInterfaceProtocol", 0),
                                    endpoints=eps))
        out[key] = DeviceDescriptor(
            bus=d["bus"], devnum=d["devnum"], vid=d["vid"], pid=d["pid"],
            bDeviceClass=d.get("bDeviceClass", 0),
            serial_present=bool(d.get("serial_present")),
            manufacturer=d.get("manufacturer"), product=d.get("product"),
            interfaces=tuple(ifaces))
    return out
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, AMBER, GREEN, GREY = "\033[31m", "\033[33m", "\033[32m", "\033[90m"
STATE_COLOUR = {"observe": GREEN, "throttled": AMBER, "revoked": RED}


def bar(v, width=22, colour=""):
    n = int(round(max(0.0, min(1.0, v)) * width))
    return f"{colour}{'█' * n}{GREY}{'·' * (width - n)}{RESET}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", type=int, default=0, help="USB bus; 0 = all buses")
    ap.add_argument("--usbmon-dir", type=Path, default=DEFAULT_USBMON_DIR)
    ap.add_argument("--rescan-every", type=int, default=5,
                    help="windows between sysfs descriptor rescans")
    ap.add_argument("--quiet-devices", action="store_true",
                    help="only print a device's line when it has traffic or a state change")
    ap.add_argument("--replay", type=Path, metavar="FILE",
                    help="drive the SAME monitor loop from a recorded usbmon file "
                         "instead of debugfs. Use this to verify the monitor works "
                         "before trusting it on live traffic -- it exercises every "
                         "part except the kernel read, and runs on any OS.")
    ap.add_argument("--descriptors", type=Path, metavar="FILE",
                    help="descriptor snapshot JSON to accompany --replay "
                         "(from scripts/vm_capture.sh)")
    args = ap.parse_args()

    if args.replay is None and platform.system() != "Linux":
        sys.exit(f"Live capture needs Linux; this is {platform.system()}.\n"
                 f"Use scripts/live_replay.py --capture <file> to replay a trace here.")

    cfg_path = MODELS / "deploy.json"
    if not cfg_path.exists():
        sys.exit(f"No trained models at {MODELS}.\n"
                 f"Run:  python3 -m eval.run_eval --cache out/corpus_cache.npz "
                 f"--target-fpr 0.0001 --persist")
    cfg = json.loads(cfg_path.read_text())
    s1 = Stage1.load(MODELS / "stage1.pkl")
    s2 = LSTMDetector.load(MODELS / "stage2.pt") if (MODELS / "stage2.pt").exists() else None

    detector = StreamingDetector(
        s1, s2, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"]),
        feature_idx=np.array(cfg["feature_idx"]), seq_len=cfg["seq_len"])

    cache: dict = {}
    state = {"w": 0}
    if args.replay is not None:
        source = UsbmonSource(path=args.replay)
        cache.update(_load_descriptors(args.descriptors))

        def descriptor_for(key: str):
            return cache.get(key)
    else:
        source = UsbmonSource.live(bus=args.bus, usbmon_dir=args.usbmon_dir)
        scanner = SysfsScanner()

        def descriptor_for(key: str):
            # rescan periodically: a device that enumerates mid-session, or
            # grows an interface, only becomes visible on a later scan
            if key not in cache or state["w"] % args.rescan_every == 0:
                cache.update(scanner.scan())
            return cache.get(key)

    monitor = LiveMonitor(detector, source, descriptor_for, window=1.0)
    signal.signal(signal.SIGINT, lambda *_: monitor.stop())

    origin = (f"replay of {args.replay.name}" if args.replay
              else f"bus {args.bus or 'all'} (live)")
    print(f"\n{BOLD}IUDBMS live monitor{RESET}  {DIM}{origin}{RESET}")
    print(f"{DIM}threshold {cfg['threshold']:.4f}   gate {cfg['k']}-of-{cfg['n']}   "
          f"{len(cfg['feature_idx'])} features{RESET}")
    print(f"{DIM}enforcement is dry-run (ENFORCEMENT_REVIEWED={ENFORCEMENT_REVIEWED}); "
          f"nothing is written to sysfs{RESET}")
    if args.replay is None:
        print(f"{AMBER}  ! The deployed model was trained on SIMULATED traffic. Its "
              f"thresholds are\n    calibrated against synthetic benign windows, so "
              f"scores on a real host are\n    out of distribution and should not be "
              f"read as verdicts yet. What this\n    demonstrates is that the capture "
              f"and scoring path runs end to end on real\n    kernel output.{RESET}")
    print(f"{DIM}Ctrl-C to stop.{RESET}\n")

    last_state: dict[str, str] = {}
    try:
        for verdicts in monitor.run():
            state["w"] = monitor.windows_scored
            for v in verdicts:
                changed = last_state.get(v.device_id) != v.trust_state
                last_state[v.device_id] = v.trust_state
                if args.quiet_devices and v.features[0] == 0 and not changed and not v.action:
                    continue
                colour = RED if v.stage2 >= cfg["threshold"] else (
                    AMBER if v.stage2 >= cfg["threshold"] * 0.6 else "")
                sc = STATE_COLOUR.get(v.trust_state, "")
                print(f"{v.t_start:6.0f}s  {v.device_id:>7s}  {bar(v.stage2, 22, colour)} "
                      f"{v.stage2:6.3f}  s1={v.stage1:5.3f}  "
                      f"{int(v.features[0]):4d} urb  {v.gate_hits}/{v.gate_n}  "
                      f"{sc}{v.trust_state}{RESET}", flush=True)
                if v.action is not None:
                    sev = RED if v.action.action == "DEAUTHORIZE" else AMBER
                    print(f"{sev}{BOLD}  ⚑ {v.action.action}{RESET} on {v.device_id} "
                          f"(severity {v.action.severity})")
                    print(f"{DIM}    would run: {v.action.rule['kernel_cmd']}{RESET}")
                    print(f"{DIM}    evidence : "
                          f"{', '.join(f'{a} {b}' for a, b in v.evidence[:3])}{RESET}",
                          flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        st = source.stats
        print(f"\n{DIM}{'─' * 70}{RESET}")
        print(f"  windows scored : {monitor.windows_scored}")
        print(f"  devices seen   : {len(last_state)}")
        print(f"  URB lines      : {st.lines_seen:,}  unparsed {st.lines_unparsed:,} "
              f"({st.loss_rate:.3%})")
        if monitor.events_late:
            print(f"  late events    : {monitor.events_late:,} (arrived after their "
                  f"window closed)")
        if st.unparsed_samples:
            print(f"{DIM}  first unparsed: {st.unparsed_samples[0][:80]}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
