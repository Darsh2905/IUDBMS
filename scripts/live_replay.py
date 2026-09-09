#!/usr/bin/env python3
"""
Live replay — watch the loop close in real time.

Streams a device session through the *deployed* pipeline at wall-clock speed and
renders what an operator would see: both stage scores, the persistence gate
filling, the trust state escalating, and the generated rule at the moment
enforcement fires.

This is not a mock. It drives ``iudbms.live.StreamingDetector`` with the exact
weights saved by the evaluation, and
``tests/test_live_detector.py::test_streaming_matches_offline_scoring`` asserts
the scores are identical to the offline ones. What you see here is what the
reported numbers describe.

    python3 -m eval.run_eval --cache out/corpus_cache.npz \
        --target-fpr 0.0001 --persist        # trains and saves out/models/
    python3 scripts/live_replay.py --profile mimicry --speed 8

    # or replay a real capture taken on the VM:
    python3 scripts/live_replay.py --capture tests/fixtures/usbmon/capture.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iudbms.capture.simulator import ALL_PROFILES, SimulatedSource, Simulator  # noqa: E402
from iudbms.capture.usbmon import UsbmonSource  # noqa: E402
from iudbms.features.extractor import window_events  # noqa: E402
from iudbms.live import StreamingDetector  # noqa: E402
from iudbms.policy import PolicyEngine  # noqa: E402
from iudbms.stage1 import Stage1  # noqa: E402
from iudbms.stage2 import LSTMDetector  # noqa: E402

MODELS = ROOT / "out" / "models"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, AMBER, GREEN, BLUE, GREY = (
    "\033[31m", "\033[33m", "\033[32m", "\033[34m", "\033[90m")
STATE_COLOUR = {"observe": GREEN, "throttled": AMBER, "revoked": RED}


def bar(value: float, width: int = 28, colour: str = "") -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return f"{colour}{'█' * filled}{GREY}{'·' * (width - filled)}{RESET}"


def load_deployment():
    cfg_path = MODELS / "deploy.json"
    if not cfg_path.exists():
        sys.exit(
            f"No trained models at {MODELS}.\n"
            f"Run:  python3 -m eval.run_eval --cache out/corpus_cache.npz "
            f"--target-fpr 0.0001 --persist")
    cfg = json.loads(cfg_path.read_text())
    s1 = Stage1.load(MODELS / "stage1.pkl")
    s2 = LSTMDetector.load(MODELS / "stage2.pt") if (MODELS / "stage2.pt").exists() else None
    return cfg, s1, s2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--profile", choices=ALL_PROFILES, default="staged_delay",
                     help="simulated device profile to replay")
    src.add_argument("--capture", type=Path,
                     help="a recorded usbmon text file (e.g. from the VM)")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="replay rate; 1.0 is real time, 6.0 is six times faster")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--follow-through", type=int, default=4, metavar="N",
                    help="windows to keep scoring after DEAUTHORIZE before stopping. "
                         "A live deployment would have detached the device; we keep "
                         "going briefly to show dry-run really is inert (0 = to the end)")
    args = ap.parse_args()

    cfg, s1, s2 = load_deployment()
    feature_idx = np.array(cfg["feature_idx"])
    policy = PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"])
    det = StreamingDetector(s1, s2, policy, feature_idx=feature_idx,
                            seq_len=cfg["seq_len"])

    # ---- source ----
    if args.capture:
        source = UsbmonSource(path=args.capture)
        events = list(source.events())
        if not events:
            sys.exit(f"No URB events parsed from {args.capture}")
        descriptors = list(source._descriptors.values())
        if not descriptors:
            sys.exit("Capture has no descriptor sidecar; pass one alongside it.")
        desc, duration, onset, label = descriptors[0], max(e.ts for e in events), None, None
        title = f"replay of {args.capture.name}"
    else:
        sess = Simulator(seed=args.seed).session("LIVE-01", args.profile)
        events = list(SimulatedSource(sess).events())
        desc, duration, onset, label = sess.descriptor, sess.duration, sess.onset, sess.label
        title = f"simulated {args.profile}"

    device_id = desc.device_key
    det.attach(desc, device_id)

    print(f"\n{BOLD}IUDBMS live replay{RESET}  {DIM}{title}{RESET}")
    print(f"{DIM}device {device_id}  vid:pid {desc.vid:04x}:{desc.pid:04x}  "
          f"declared class {[hex(i.bInterfaceClass) for i in desc.interfaces]}{RESET}")
    print(f"{DIM}threshold {cfg['threshold']:.4f}   gate {cfg['k']}-of-{cfg['n']}   "
          f"{len(feature_idx)} features   replay {args.speed}x{RESET}")
    if onset is not None:
        print(f"{DIM}ground truth: attack begins at t={onset:.0f}s{RESET}")
    print(f"{DIM}{'─' * 78}{RESET}")
    print(f"{DIM}  t     Stage 2                        score   S1     gate   state{RESET}")

    fired = None
    revoked_at = None
    t0_wall = time.monotonic()
    for t_start, bucket in window_events(events, duration=duration):
        # pace the replay so the loop is visibly closing in time
        target = t0_wall + (t_start + 1.0) / max(args.speed, 0.01)
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, 2.0))

        v = det.observe_window(device_id, t_start, bucket)
        colour = RED if v.stage2 >= cfg["threshold"] else (
            AMBER if v.stage2 >= cfg["threshold"] * 0.6 else BLUE)
        onset_mark = "▶" if onset is not None and abs(t_start - onset) < 0.5 else " "
        gate = f"{v.gate_hits}/{v.gate_n}"
        state_c = STATE_COLOUR.get(v.trust_state, "")
        print(f"{onset_mark} {t_start:4.0f}s  {bar(v.stage2, 28, colour)} "
              f"{v.stage2:6.3f}  {v.stage1:5.3f}  {gate:>5s}  "
              f"{state_c}{v.trust_state}{RESET}", flush=True)

        if v.action is not None:
            fired = fired or (t_start + 1.0)
            if v.action.action == "DEAUTHORIZE":
                revoked_at = st_idx = 0
            sev = RED if v.action.action == "DEAUTHORIZE" else AMBER
            print(f"{DIM}{'─' * 78}{RESET}")
            print(f"{sev}{BOLD}  ⚑ {v.action.action}{RESET}  "
                  f"{sev}severity={v.action.severity}{RESET}  at t={v.action.ts:.0f}s")
            print(f"{DIM}  generated rule:{RESET}")
            for line in json.dumps(v.action.rule, indent=2).splitlines():
                print(f"{DIM}    {line}{RESET}")
            print(f"{DIM}{'─' * 78}{RESET}")

    print(f"{DIM}{'─' * 78}{RESET}")
    st = det.devices[device_id]
    print(f"  windows scored : {st.n_windows}")
    print(f"  final state    : {STATE_COLOUR.get(st.ctx.state, '')}{st.ctx.state}{RESET}")
    if onset is not None and fired is not None:
        print(f"  detection latency: {BOLD}{fired - onset:.1f}s{RESET} "
              f"(onset {onset:.0f}s → action {fired:.0f}s)")
    elif label == 1:
        print(f"  {RED}not detected{RESET} — this is a miss, and it is reported as one")
    elif fired is None:
        print(f"  {GREEN}no action{RESET} — benign device, correctly left alone")
    print(f"{DIM}  enforcement is dry-run; no write to /sys/bus/usb/…/authorized "
          f"was executed{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
