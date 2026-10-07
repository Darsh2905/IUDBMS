"""
Parser validation against real kernel usbmon output.

    python3 -m eval.real_capture_summary

Unlike eval/real_typing.py, this needs no external capture: it runs the
committed fixture (tests/fixtures/usbmon/real_qemu_vm_capture.txt — 338 lines
of real usbmon text from a UTM/QEMU Ubuntu 24.04.5 aarch64 guest, kernel 6.8,
recorded while `lsusb -v` read every descriptor and the kernel polled an idle
USB CD-ROM) through the real parser and reports what plan §4.1 asks for: loss
rate against the < 0.1% criterion, and that every emitted event satisfies the
contract. Writes out/real_capture_summary.json for scripts/build_paper.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIXTURE = ROOT / "tests" / "fixtures" / "usbmon" / "real_qemu_vm_capture.txt"
sys.path.insert(0, str(ROOT / "src"))

from iudbms.capture.usbmon import UsbmonSource  # noqa: E402


def main() -> int:
    src = UsbmonSource(path=FIXTURE)
    events = list(src.events())
    st = src.stats

    valid = 0
    for e in events:
        try:
            e.validate()
            valid += 1
        except Exception:
            pass

    kinds = Counter((e.ttype, e.direction) for e in events if e.evt == "C")
    ctrl_s = [e for e in events if e.ttype == "control" and e.evt == "S"]
    intr = [e for e in events if e.ttype == "intr"]
    devices = sorted({e.device_key for e in events})
    span = float(events[-1].ts - events[0].ts) if events else 0.0

    result = {
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "lines_seen": st.lines_seen,
        "events_emitted": st.events_emitted,
        "lines_unparsed": st.lines_unparsed,
        "loss_rate": st.loss_rate,
        "plan_criterion": 0.001,
        "events_contract_valid": valid,
        "events_total": len(events),
        "devices_seen": devices,
        "n_devices": len(devices),
        "span_s": span,
        "completions_by_type": {f"{t}_{d}": n for (t, d), n in kinds.items()},
        "control_submissions_with_setup": sum(1 for e in ctrl_s if e.setup),
        "control_submissions_total": len(ctrl_s),
        "interrupt_events": len(intr),
        "interrupt_interval_is_own_field": all(
            e.interval is not None and e.length <= 8 for e in intr) if intr else None,
        "timestamps_monotone": all(a.ts <= b.ts for a, b in zip(events, events[1:])),
    }

    print(f"fixture         : {result['fixture']}")
    print(f"lines / unparsed: {st.lines_seen} / {st.lines_unparsed} "
          f"({st.loss_rate:.4%}, plan §4.1 criterion < 0.1%)")
    print(f"contract-valid  : {valid} of {len(events)}")
    print(f"devices seen    : {len(devices)}  {devices}")
    print(f"span            : {span:.1f} s")
    print(f"completions     : {dict(kinds)}")
    print(f"control w/ SETUP: {result['control_submissions_with_setup']} of "
          f"{result['control_submissions_total']}")

    OUT.mkdir(exist_ok=True)
    (OUT / "real_capture_summary.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"\n  wrote out/real_capture_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
