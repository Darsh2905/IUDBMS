"""
iudbms-monitor -- read the kernel, score devices as they act, keep the evidence.

    sudo modprobe usbmon
    sudo iudbms-monitor --bus 0                       # live, all buses
    iudbms-monitor --replay capture.txt --descriptors desc.json
    iudbms-monitor --replay-db iudbms.db --run-id 3    # replay a stored run

Live capture needs **Linux** with debugfs mounted and read access to
``/sys/kernel/debug/usb/usbmon/<bus>u`` (root, normally). Bus 0 is usbmon's
"all buses" node. Both replay modes exercise every part of the loop except the
kernel read, and run on any OS.

What it writes, to ``--db`` (``$IUDBMS_DB``):
  * one ``windows`` row per scored window, ``alerts``/``actions`` on every
    escalation, and the device's trust state -- what the dashboard shows;
  * the raw URB trace (``--trace``), payloads pseudonymised by default, with each
    escalation's evidence pinned against eviction.

Enforcement stays dry-run: actions are generated, logged and persisted, and
nothing is written to /sys/bus/usb/.../authorized. That gate is
``policy.engine.ENFORCEMENT_REVIEWED``, which is still False.

Every option can be set from the environment, which is how the systemd unit in
``deploy/`` configures it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np

from ..capture.sysfs import SysfsScanner
from ..capture.usbmon import DEFAULT_USBMON_DIR, UsbmonSource
from ..contract import CONTRACT_VERSION, DeviceDescriptor, Endpoint, Interface
from ..policy import ENFORCEMENT_REVIEWED, ModelSupport, PolicyEngine
from ..stage1 import Stage1
from ..stage2 import LSTMDetector
from ..storage import PAYLOAD_MODES, Store, StoreSource, TraceRecorder
from .detector import StreamingDetector
from .monitor import LiveMonitor
from .persist import LivePersister

__all__ = ["main", "load_models"]

HEARTBEAT_WINDOWS = 60


def _env(name: str, default):
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def load_models(models: Path, *, allow_stage1_only: bool = False):
    """Load the deployed operating point and both stages.

    Refuses to run without Stage 2. The previous script loaded it only "if it
    exists", so a checkout missing ``stage2.pt`` -- which a ``*.pt`` gitignore
    rule guaranteed for every fresh clone -- silently ran Stage 1 alone: the
    configuration measured at 15.8% detection, presented as the deployed one.
    """
    cfg_path = models / "deploy.json"
    try:
        present = cfg_path.exists()
    except PermissionError:
        raise SystemExit(
            f"Cannot read {models}: permission denied. The deployed models are "
            f"root:iudbms 2750 -- run with sudo, or as a member of group iudbms.") from None
    if not present:
        raise SystemExit(
            f"No trained models at {models}.\n"
            f"Produce them with:  make eval   (writes out/models/)\n"
            f"or point --models / $IUDBMS_MODELS at an installed set.")
    cfg = json.loads(cfg_path.read_text())
    want, have = cfg.get("contract_version", "?"), CONTRACT_VERSION
    if want.split(".")[0] != have.split(".")[0]:
        raise SystemExit(f"Models were trained against contract {want}; this build is "
                         f"{have}. Retrain with `make eval` before deploying.")
    s1 = Stage1.load(models / "stage1.pkl")
    s2_path = models / "stage2.pt"
    if s2_path.exists():
        s2 = LSTMDetector.load(s2_path)
    elif allow_stage1_only:
        s2 = None
    else:
        raise SystemExit(
            f"{s2_path} is missing. Without Stage 2 the monitor runs the Stage-1-only "
            f"configuration (15.8% session detection on the evaluation corpus), not the "
            f"deployed system. Pass --allow-stage1-only if that is really intended.")
    verify_canary(cfg, s1, s2, models)
    return cfg, s1, s2


def verify_canary(cfg: dict, s1, s2, models: Path, tol: float = 1e-4) -> None:
    """Refuse weights that do not belong to the threshold they are deployed with.

    The first deployment shipped a Stage-2 checkpoint from a different run than
    the ``deploy.json`` beside it: at the recorded threshold it detected 0 of 120
    attacks, while the service reported itself healthy and scored traffic. Every
    other check passed, because each file was individually valid.
    """
    want = cfg.get("canary")
    if not want:
        print(f"warning: {models / 'deploy.json'} has no canary, so these weights "
              f"cannot be verified against the threshold. Retrain with `make eval`.",
              file=sys.stderr)
        return
    for name, model, ref in (("stage1", s1, want.get("stage1")),
                             ("stage2", s2, want.get("stage2"))):
        if model is None or ref is None:
            continue
        got = model.canary(cfg["seq_len"]) if name == "stage2" else model.canary()
        worst = max(abs(a - b) for a, b in zip(got, ref))
        if worst > tol:
            raise SystemExit(
                f"{models / ('stage1.pkl' if name == 'stage1' else 'stage2.pt')} does not "
                f"match deploy.json (canary differs by {worst:.4g}). These weights are not "
                f"the ones the threshold {cfg['threshold']:.4f} was calibrated for, so the "
                f"monitor would score traffic but could not be trusted to alert. "
                f"Regenerate the set together with `make eval`.")


def _load_descriptor_snapshot(path):
    """Rebuild DeviceDescriptors from a scripts/vm_capture.sh snapshot."""
    if path is None or not Path(path).exists():
        return {}
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


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

class Out:
    """Terminal or journal. Colour only on a TTY: under systemd stdout is the
    journal, where ANSI escapes are noise in every line."""

    def __init__(self, colour: bool):
        c = (lambda code: code) if colour else (lambda code: "")
        self.RESET, self.BOLD, self.DIM = c("\033[0m"), c("\033[1m"), c("\033[2m")
        self.RED, self.AMBER, self.GREEN, self.GREY = (c("\033[31m"), c("\033[33m"),
                                                       c("\033[32m"), c("\033[90m"))
        self.colour = colour

    def bar(self, v, width=22, colour=""):
        n = int(round(max(0.0, min(1.0, v)) * width))
        fill, empty = ("█", "·") if self.colour else ("#", ".")
        return f"{colour}{fill * n}{self.GREY}{empty * (width - n)}{self.RESET}"

    def line(self, s=""):
        print(s, flush=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="iudbms-monitor", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--bus", type=int, default=int(_env("IUDBMS_BUS", 0)),
                     help="USB bus; 0 = all buses ($IUDBMS_BUS)")
    src.add_argument("--usbmon-dir", type=Path, default=DEFAULT_USBMON_DIR)
    src.add_argument("--rescan-every", type=int, default=5,
                     help="windows between sysfs descriptor rescans")
    src.add_argument("--replay", type=Path, metavar="FILE",
                     help="drive the same loop from a recorded usbmon text file")
    src.add_argument("--descriptors", type=Path, metavar="FILE",
                     help="descriptor snapshot JSON to accompany --replay "
                          "(from scripts/vm_capture.sh)")
    src.add_argument("--replay-db", type=Path, metavar="DB",
                     help="drive the loop from a run stored in a database's trace")
    src.add_argument("--run-id", type=int, help="the stored run to replay (--replay-db)")
    src.add_argument("--fast", action="store_true",
                     help="replay only: close each window as soon as the file has been read "
                          "past it, instead of at wall-clock pace. Same windows, same scores.")

    mdl = ap.add_argument_group("models")
    mdl.add_argument("--models", type=Path,
                     default=Path(_env("IUDBMS_MODELS", "out/models")),
                     help="directory with deploy.json, stage1.pkl, stage2.pt "
                          "($IUDBMS_MODELS, default out/models)")
    mdl.add_argument("--allow-stage1-only", action="store_true",
                     help="run without stage2.pt (NOT the deployed configuration)")

    st = ap.add_argument_group("storage")
    # Not out/iudbms.db: that is the evaluation's database, and `make eval`
    # deletes it before rewriting it -- it would wipe live data on a dev machine
    # and mix live devices with evaluation sessions on the dashboard. Deployed,
    # both services point at one file through $IUDBMS_DB.
    st.add_argument("--db", type=Path, default=Path(_env("IUDBMS_DB", "out/live.db")),
                    help="SQLite store ($IUDBMS_DB, default out/live.db)")
    st.add_argument("--no-persist", action="store_true",
                    help="write nothing; print only (the old behaviour)")
    st.add_argument("--trace", choices=PAYLOAD_MODES + ("off",),
                    default=_env("IUDBMS_TRACE", "pseudonymous"),
                    help="raw URB trace payload mode ($IUDBMS_TRACE, default "
                         "pseudonymous). 'full' stores keystrokes in clear.")
    st.add_argument("--trace-max-events", type=int,
                    default=int(_env("IUDBMS_TRACE_MAX_EVENTS", 1_000_000)),
                    help="retention cap on stored URBs; alert evidence is exempt")

    o = ap.add_argument_group("output")
    o.add_argument("--log", choices=("verbose", "actions"),
                   default=_env("IUDBMS_LOG", "verbose" if sys.stdout.isatty() else "actions"),
                   help="verbose: every device every window. actions: escalations, "
                        "plus a heartbeat every 60 windows (the service default)")
    o.add_argument("--quiet-devices", action="store_true",
                   help="verbose mode: skip idle devices with no state change")
    o.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    o.add_argument("--max-windows", type=int, help="stop after N windows")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replaying = args.replay is not None or args.replay_db is not None
    if args.fast and not replaying:
        raise SystemExit("--fast applies to --replay / --replay-db only: live capture is "
                         "paced by the kernel's clock, and an idle bus would stall it")
    if not replaying and platform.system() != "Linux":
        raise SystemExit(f"Live capture needs Linux; this is {platform.system()}.\n"
                         f"Use --replay FILE or --replay-db DB to run the loop here.")
    if args.replay_db is not None and args.run_id is None:
        raise SystemExit("--replay-db needs --run-id (list runs with the API's "
                         "/traces/runs, or the trace_runs table)")

    out = Out(args.color == "always" or (args.color == "auto" and sys.stdout.isatty()))
    cfg, s1, s2 = load_models(args.models, allow_stage1_only=args.allow_stage1_only)
    # Device types the model was trained on (policy.support). A set saved before this
    # existed has no record and judges every device, as it always did.
    support = ModelSupport(cfg["support"]) if cfg.get("support") else None
    detector = StreamingDetector(
        s1, s2, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"], support=support),
        feature_idx=np.array(cfg["feature_idx"]), seq_len=cfg["seq_len"])

    # ---- source ------------------------------------------------------------
    cache: dict = {}
    state = {"w": 0}
    replay_store = None
    if args.replay is not None:
        source = UsbmonSource(path=args.replay)
        cache.update(_load_descriptor_snapshot(args.descriptors))
        origin = f"replay of {args.replay.name}"

        def descriptor_for(key):
            return cache.get(key)
    elif args.replay_db is not None:
        replay_store = Store(args.replay_db)
        source = StoreSource(replay_store, args.run_id)
        origin = f"replay of stored run {args.run_id} from {args.replay_db.name}"

        def descriptor_for(key):
            try:
                return source.descriptor(key)
            except KeyError:
                return None
    else:
        source = UsbmonSource.live(bus=args.bus, usbmon_dir=args.usbmon_dir)
        scanner = SysfsScanner()
        origin = f"bus {args.bus or 'all'} (live)"

        def descriptor_for(key):
            # rescan periodically: a device that enumerates mid-session, or
            # grows an interface, only becomes visible on a later scan
            if key not in cache or state["w"] % args.rescan_every == 0:
                if monitor.time_origin is not None:
                    scanner.origin = monitor.time_origin
                cache.update(scanner.scan())
            return cache.get(key)

    # ---- storage -----------------------------------------------------------
    store = recorder = persister = None
    if not args.no_persist:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        store = Store(args.db)
        if args.trace != "off":
            recorder = TraceRecorder(store, payload_mode=args.trace,
                                     max_events=args.trace_max_events,
                                     source=origin, window=1.0)
        persister = LivePersister(store, recorder=recorder, seq_len=cfg["seq_len"],
                                  window=1.0)

    monitor = LiveMonitor(detector, source, descriptor_for, window=1.0, recorder=recorder,
                          pace="event" if args.fast else "wallclock")

    def _stop(*_):
        monitor.stop()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)     # systemd stop: shut down cleanly

    # ---- banner ------------------------------------------------------------
    out.line(f"\n{out.BOLD}IUDBMS monitor{out.RESET}  {out.DIM}{origin}{out.RESET}")
    out.line(f"{out.DIM}threshold {cfg['threshold']:.4f}   gate {cfg['k']}-of-{cfg['n']}   "
             f"{len(cfg['feature_idx'])} features   stage 2 "
             f"{'loaded' if s2 is not None else 'ABSENT (stage-1-only)'}{out.RESET}")
    out.line(f"{out.DIM}model support: "
             + (f"interface classes {sorted(support.classes)}; devices declaring others are reported "
                f"'unmodelled' and never escalated (any HID interface is always judged)"
                if support is not None else "not recorded in deploy.json; every device is judged")
             + out.RESET)
    out.line(f"{out.DIM}enforcement is dry-run (ENFORCEMENT_REVIEWED={ENFORCEMENT_REVIEWED}); "
             f"nothing is written to sysfs{out.RESET}")
    if store is None:
        out.line(f"{out.DIM}persistence off: nothing is written{out.RESET}")
    else:
        trace = ("off" if recorder is None else
                 f"{args.trace} payloads, cap {args.trace_max_events:,} URBs")
        out.line(f"{out.DIM}store {args.db}   trace: {trace}{out.RESET}")
    if args.trace == "full" and store is not None:
        out.line(f"{out.RED}{out.BOLD}  ! --trace full stores raw payloads. Every keystroke "
                 f"typed on a monitored keyboard is\n    written to {args.db} in clear. Use "
                 f"only on devices you own, with consent.{out.RESET}")
    if not replaying:
        out.line(f"{out.AMBER}  ! The deployed model was trained on SIMULATED traffic. Its "
                 f"thresholds are\n    calibrated against synthetic benign windows, so "
                 f"scores on a real host are\n    out of distribution and should not be "
                 f"read as verdicts yet.{out.RESET}")
    out.line(f"{out.DIM}Ctrl-C (or SIGTERM) to stop.{out.RESET}\n")

    # ---- loop --------------------------------------------------------------
    STATE_COLOUR = {"observe": out.GREEN, "throttled": out.AMBER, "revoked": out.RED,
                    "unmodelled": out.GREY}
    last_state: dict[str, str] = {}
    t_wall = time.monotonic()
    try:
        for verdicts in monitor.run(max_windows=args.max_windows):
            state["w"] = monitor.windows_scored
            if persister is not None:
                persister.observe(verdicts, lambda k: (detector.devices[k].descriptor
                                                       if k in detector.devices else None))
            for v in verdicts:
                changed = last_state.get(v.device_id) != v.trust_state
                last_state[v.device_id] = v.trust_state
                if changed and v.trust_state == "unmodelled":
                    d = detector.devices[v.device_id].descriptor
                    out.line(f"{out.GREY}NOTICE device={v.device_id} declares interface types the "
                             f"model was not trained on {support.uncovered(d)}: scored and stored, "
                             f"never escalated. Decide its policy by type.{out.RESET}")
                if args.log == "verbose" and not (
                        args.quiet_devices and v.features[0] == 0 and not changed
                        and not v.action):
                    colour = out.RED if v.stage2 >= cfg["threshold"] else (
                        out.AMBER if v.stage2 >= cfg["threshold"] * 0.6 else "")
                    sc = STATE_COLOUR.get(v.trust_state, "")
                    out.line(f"{v.t_start:6.0f}s  {v.device_id:>7s}  "
                             f"{out.bar(v.stage2, 22, colour)} {v.stage2:6.3f}  "
                             f"s1={v.stage1:5.3f}  {int(v.features[0]):4d} urb  "
                             f"{v.gate_hits}/{v.gate_n}  {sc}{v.trust_state}{out.RESET}")
                if v.action is not None:
                    sev = out.RED if v.action.action == "DEAUTHORIZE" else out.AMBER
                    out.line(f"{sev}{out.BOLD}ACTION {v.action.action}{out.RESET} "
                             f"device={v.device_id} t={v.t_start:.0f}s "
                             f"severity={v.action.severity} stage2={v.stage2:.3f}")
                    out.line(f"{out.DIM}    would run: {v.action.rule['kernel_cmd']}{out.RESET}")
                    out.line(f"{out.DIM}    evidence : "
                             f"{', '.join(f'{a} {b}' for a, b in v.evidence[:3])}{out.RESET}")
            if (args.log == "actions" and monitor.windows_scored
                    and monitor.windows_scored % HEARTBEAT_WINDOWS == 0):
                extra = (f" urb_recorded={recorder.n_recorded} evicted={recorder.n_evicted}"
                         if recorder else "")
                out.line(f"heartbeat windows={monitor.windows_scored} "
                         f"devices={len(detector.devices)} late={monitor.events_late}"
                         f"{extra}")
    except KeyboardInterrupt:
        pass
    finally:
        if persister is not None:
            persister.close()
        st = source.stats
        out.line(f"\n{out.DIM}{'─' * 70}{out.RESET}")
        out.line(f"  windows scored : {monitor.windows_scored}"
                 f"   ({time.monotonic() - t_wall:.1f} s wall)")
        out.line(f"  devices seen   : {len(last_state)}")
        out.line(f"  URB lines      : {st.lines_seen:,}  unparsed {st.lines_unparsed:,} "
                 f"({st.loss_rate:.3%})")
        if monitor.events_late:
            out.line(f"  late events    : {monitor.events_late:,} (arrived after their "
                     f"window closed; stored, not scored)")
        if monitor.events_undescribed:
            out.line(f"  undescribed    : {monitor.events_undescribed:,} events from devices "
                     f"with no sysfs entry (root hubs); stored, not scored")
        if persister is not None:
            out.line(f"  persisted      : {persister.n_windows:,} windows, "
                     f"{persister.n_actions} actions -> {args.db}")
        if recorder is not None:
            s = recorder.stats
            out.line(f"  trace run {s['run_id']}    : {s['recorded']:,} URBs "
                     f"({s['payload_mode']}), {s['pinned']:,} pinned, "
                     f"{s['evicted']:,} evicted")
        if st.unparsed_samples:
            out.line(f"{out.DIM}  first unparsed: {st.unparsed_samples[0][:80]}{out.RESET}")
        if store is not None:
            store.close()
        if replay_store is not None:
            replay_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
