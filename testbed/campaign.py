"""
Run testbed sessions: boot a distribution, drive its USB devices, record usbmon.

    python3 -m testbed.campaign benign  --distro debian-13 --minutes 75 --seed 1
    python3 -m testbed.campaign attack  --distro debian-13 --seed 1
    python3 -m testbed.campaign all     --distros debian-13 fedora-44 alma-9 ubuntu-24.04

Each session writes out/testbed/raw/<distro>-<kind>-s<seed>/ with:

  usbmon.txt          the guest's /sys/kernel/debug/usb/usbmon/0u, whole session
  descriptors.json    union of sysfs snapshots taken at start, after every hotplug, end
  labels.json         device key -> role (benign class, or the attack it carried)
  env.json            kernel, distribution, driver bound to every interface
  workload.json       host-side log of what was driven when, plus the guest schedule

Raw directories are git-ignored (they contain replayed keystrokes); the scorer
(testbed.score) writes the committed summaries.

**Sessions are long on purpose.** A benign session defaults to 75 minutes so that
it crosses the usbmon timestamp wrap (4096 s of guest uptime) at least once on a
real kernel -- the bug fixed in commit c4cde70 was invisible to every shorter
capture.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import human_input as hi
from .vm import DEVICES, DISTROS, ROOT, Guest
from .workloads import (INJECT_TEXT, LOOP_TEXT, PANGRAM, HumanKeyboard, HumanPointer, Log,
                        guest_schedule, human_timing, replay_strokes, type_text)

RAW = ROOT / "out" / "testbed" / "raw"
GUEST_DIR = "/var/tmp/tb"
EPISODES = ("hid_injection", "benign_hotplug", "mimicry", "mimicry_adaptive", "staged_delay",
            "identity_switch")
BENIGN_DEVICES = ("kbd", "mouse", "tablet", "wacom", "storage", "uas", "audio", "net",
                  "serial", "ccid", "hub")


# ----------------------------------------------------------------- helpers ---
def snapshot(g: Guest) -> dict:
    r = g.ssh(f"sudo python3 {GUEST_DIR}/guest_descriptors.py", timeout=60)
    return json.loads(r.stdout)


def env_info(g: Guest) -> dict:
    r = g.ssh("uname -r; . /etc/os-release; echo \"$PRETTY_NAME\"; "
              "for d in /sys/bus/usb/devices/*:*; do "
              "echo \"$(basename $d) $(cat $d/bInterfaceClass) "
              "$(basename $(readlink $d/driver 2>/dev/null) 2>/dev/null)\"; done; "
              "python3 --version", sudo=True, check=False)
    out = r.stdout.strip().splitlines()
    ifaces = {}
    for ln in out[2:]:
        bits = ln.split()
        if len(bits) >= 2 and ":" in bits[0]:
            ifaces[bits[0]] = {"class": bits[1], "driver": bits[2] if len(bits) > 2 else None}
    return {"kernel": out[0] if out else None, "os": out[1] if len(out) > 1 else None,
            "interfaces": ifaces, "python": out[-1] if out else None}


def prepare_guest(g: Guest, log: Log) -> None:
    g.ssh(f"mkdir -p {GUEST_DIR} && sudo chmod 777 {GUEST_DIR}")
    g.scp_to(ROOT / "testbed" / "guest_descriptors.py", f"{GUEST_DIR}/guest_descriptors.py")
    # Audio playback needs alsa-utils, which cloud images do not ship. Best effort with a
    # time limit; if it fails the audio device is present and idle, and env.json says so.
    r = g.ssh("(command -v apt-get >/dev/null && DEBIAN_FRONTEND=noninteractive "
              "timeout 400 apt-get -qq update && DEBIAN_FRONTEND=noninteractive timeout 400 "
              "apt-get -qq install -y alsa-utils) || (command -v dnf >/dev/null && "
              "timeout 600 dnf -q -y install alsa-utils) ; command -v aplay",
              sudo=True, check=False, timeout=1200)
    log.add("alsa_utils", ok=r.returncode == 0 and "aplay" in r.stdout)
    # A desktop session holds every input device open, which is what makes usbhid
    # poll it. A headless guest has nobody reading the mouse, so hold them open.
    g.ssh("for e in /dev/input/event*; do setsid nohup cat $e >/dev/null 2>&1 </dev/null & done; true",
          sudo=True, check=False)


def start_capture(g: Guest) -> None:
    g.ssh(f"modprobe usbmon; mountpoint -q /sys/kernel/debug || mount -t debugfs none "
          f"/sys/kernel/debug; setsid nohup cat /sys/kernel/debug/usb/usbmon/0u "
          f"> {GUEST_DIR}/usbmon.txt 2>/dev/null </dev/null & sleep 1; cat /proc/uptime",
          sudo=True)


def stop_capture(g: Guest) -> None:
    g.ssh("pkill -f 'cat /sys/kernel/debug/usb/usbmon/0u'; sleep 1; sync", sudo=True, check=False)


def human_material(seed: int, n_typists: int = 400):
    rng = random.Random(seed)
    typing = hi.aalto_sessions()
    if not typing:
        raise SystemExit("no Aalto sessions: run `python3 -m testbed.human_input fetch`")
    rng.shuffle(typing)
    mice = hi.balabit_sessions()
    rng.shuffle(mice)
    return typing[:n_typists], mice


def device_role(d: dict) -> str:
    """Role from what the device declares (interface triples, vendor), not its name."""
    t = {(i["bInterfaceClass"], i["bInterfaceSubClass"], i["bInterfaceProtocol"])
         for i in d.get("interfaces", [])}
    cls = {c for c, _, _ in t}
    if (3, 1, 1) in t:
        return "keyboard"
    if (3, 1, 2) in t:
        return "pen-tablet" if d.get("vid") == 0x056A else "mouse"   # Wacom declares boot mouse
    if 3 in cls:
        return "tablet"
    for c, role in ((1, "audio"), (2, "network"), (11, "smartcard"), (9, "hub"), (255, "serial")):
        if c in cls:
            return role
    if (8, 6, 98) in t:
        return "mass_storage_uas"
    if 8 in cls:
        return "mass_storage"
    return "unknown"


def _new_key(before: dict, after: dict) -> str | None:
    new = sorted(set(after) - set(before))
    return new[0] if new else None


# ---------------------------------------------------------------- sessions ---
def run_session(distro: str, kind: str, seed: int, minutes: float,
                episodes: tuple[str, ...] = EPISODES) -> Path:
    rd = RAW / f"{distro}-{kind}-s{seed}"
    rd.mkdir(parents=True, exist_ok=True)
    typing, mice = human_material(seed)
    rng = random.Random(seed * 7919 + zlib.crc32(distro.encode()) % 1000)
    g = Guest(distro, BENIGN_DEVICES, rd)
    log = Log()
    duration = minutes * 60.0
    labels: dict[str, dict] = {}
    snaps: list[dict] = []
    threads: list = []
    try:
        g.start()
        log.add("booted")
        prepare_guest(g, log)
        env = env_info(g)
        (rd / "env.json").write_text(json.dumps(env, indent=1))
        snaps.append(snapshot(g))
        for key, d in snaps[0].items():
            labels[key] = {"role": device_role(d), "label": 0, "product": d.get("product")}
        sched = guest_schedule(seed * 31 + 1, duration)
        (rd / "guest_schedule.sh").write_text(sched)
        g.scp_to(rd / "guest_schedule.sh", f"{GUEST_DIR}/schedule.sh")

        # ---- go ------------------------------------------------------------
        log.t0 = time.perf_counter()
        start_capture(g)
        log.add("capture_start")
        g.ssh(f"setsid nohup bash {GUEST_DIR}/schedule.sh > {GUEST_DIR}/schedule.log 2>&1 </dev/null &",
              sudo=True)
        kb = HumanKeyboard(g.qmp, "gpu0", typing, log, rng, until=duration)
        ms = HumanPointer(g.qmp, None, mice[0::2] or mice, log, until=duration, mode="rel")
        tb = HumanPointer(g.qmp, "gpu0", mice[1::2] or mice, log, until=duration, mode="abs")
        threads = [kb, ms, tb]
        for t in threads:
            t.start()

        if kind == "attack":
            run_attacks(g, log, typing, labels, snaps, rng, duration, episodes)
        else:
            while log.now() < duration:
                if any(t.error for t in threads):
                    raise RuntimeError(f"workload thread failed: {[t.error for t in threads]}")
                time.sleep(5)

        for t in threads:
            t.stop.set()
        for t in threads:
            t.join(timeout=30)
        stop_capture(g)
        log.add("capture_stop")
        snaps.append(snapshot(g))
        g.scp_from(f"{GUEST_DIR}/usbmon.txt", rd / "usbmon.txt")
        g.scp_from(f"{GUEST_DIR}/schedule.log", rd / "guest_schedule.log")
        errs = [repr(t.error) for t in threads if t.error]
        merged: dict = {}
        for s in snaps:
            merged.update(s)
        (rd / "descriptors.json").write_text(json.dumps(merged, indent=1))
        (rd / "labels.json").write_text(json.dumps(labels, indent=1))
        (rd / "workload.json").write_text(json.dumps({
            "distro": distro, "kind": kind, "seed": seed, "minutes": minutes,
            "keys_typed": kb.keys, "mouse_events": ms.events, "tablet_events": tb.events,
            "thread_errors": errs, "log": log.rows}, indent=1))
        print(f"[{distro} {kind} s{seed}] done: {kb.keys} keys, {ms.events}+{tb.events} pointer "
              f"events, {sum(1 for _ in open(rd / 'usbmon.txt'))} usbmon lines"
              + (f"  ERRORS {errs}" if errs else ""), flush=True)
    finally:
        for t in threads:
            t.stop.set()
        g.stop()
    return rd


def _plug(g: Guest, log: Log, snaps: list, driver: str, dev_id: str, **props) -> str | None:
    before = snapshot(g)
    g.qmp.device_add(driver, id=dev_id, bus="xhci.0", **props)
    log.add("plug", dev=dev_id, driver=driver)
    for _ in range(20):                       # wait for the kernel to enumerate it
        time.sleep(0.5)
        after = snapshot(g)
        key = _new_key(before, after)
        if key:
            snaps.append(after)
            return key
    return None


def _unplug(g: Guest, log: Log, dev_id: str) -> None:
    g.qmp.device_del(dev_id)
    log.add("unplug", dev=dev_id)


def _wait_until(log: Log, t: float) -> None:
    while log.now() < t:
        time.sleep(0.2)


def run_attacks(g: Guest, log: Log, typing, labels, snaps, rng, duration,
                episodes: tuple[str, ...] = EPISODES) -> None:
    """Six episodes on hot-plugged devices, each on its own device number.

    The benign background (keyboard, mice, storage, network, serial, audio) runs the
    whole time, so the session also measures false alarms on the devices that stay.
    """
    stop = threading.Event()
    donor = rng.choice(typing)                       # a real participant's timing for mimicry
    timing = human_timing(donor)
    # Sequential, 60 s of background-only traffic between episodes, so an episode that
    # overruns can never overlap the next one.
    t_next = 60.0
    for i, what in enumerate(episodes):
        if t_next > duration - 60:
            log.add("episode_skipped", episode=what, reason="session too short")
            continue
        _wait_until(log, t_next)
        try:
            _episode(g, log, typing, labels, snaps, rng, timing, stop, i, what)
        except Exception as e:                   # one failed episode must not void the session
            log.add("episode_error", episode=what, error=repr(e))
            print(f"  episode {what} failed: {e!r}", file=sys.stderr)
        t_next = log.now() + 60.0
    _wait_until(log, min(duration, log.now() + 30))


def _episode(g, log, typing, labels, snaps, rng, timing, stop, i, what) -> None:
    if True:
        dev = f"ep{i}"
        if what == "identity_switch":
            # a storage device works normally, disappears, and comes back as a keyboard
            key = _plug(g, log, snaps, "usb-storage", dev + "s", drive="stick1", removable=True) \
                if _has_drive(g) else None
            if key:
                labels[key] = {"role": "identity_switch_storage_phase", "label": 0,
                               "episode": what}
                g.ssh(f"d=$(ls -t /dev/sd? | head -1); for i in 1 2 3; do dd if=$d of=/dev/null "
                      f"bs=1M count=16 iflag=direct status=none; sleep 8; done",
                      sudo=True, check=False, timeout=120)
                _unplug(g, log, dev + "s")
                time.sleep(1.0)
        key = _plug(g, log, snaps, "usb-kbd", dev, display="gpu1", usb_version=1)
        if key is None:
            log.add("plug_failed", dev=dev, episode=what)
            return
        labels[key] = {"role": what, "label": 0 if what == "benign_hotplug" else 1,
                       "episode": what, "qmp_id": dev}
        time.sleep(2.0)
        log.add("episode_begin", dev=dev, key=key, episode=what)
        if what in ("hid_injection", "identity_switch"):
            n = type_text(g.qmp, "gpu1", INJECT_TEXT, stop)
            time.sleep(5)
        elif what == "benign_hotplug":
            n = replay_strokes(g.qmp, "gpu1", rng.choice(typing).strokes, stop)
            time.sleep(10)
        elif what == "mimicry":
            n, t_end = 0, log.now() + 180
            while log.now() < t_end:
                n += type_text(g.qmp, "gpu1", LOOP_TEXT, stop, timing=timing)
        elif what == "mimicry_adaptive":
            n = type_text(g.qmp, "gpu1", (PANGRAM * 4)[:160], stop, timing=timing)
            time.sleep(120)                      # typed once; then nothing, for good
        elif what == "staged_delay":
            time.sleep(120)                      # dwell: plugged in, silent
            n = type_text(g.qmp, "gpu1", INJECT_TEXT, stop)
            time.sleep(5)
        log.add("episode_end", dev=dev, key=key, episode=what, keys=n)
        _unplug(g, log, dev)


def _has_drive(g: Guest) -> bool:
    """Attach a second backing drive for the identity-switch storage phase."""
    p = g.run_dir / "stick1.img"
    if not p.exists():
        with open(p, "wb") as f:
            f.truncate(64 * 1024 * 1024)
    try:
        g.qmp.cmd("blockdev-add", driver="raw", **{"node-name": "stick1",
                  "file": {"driver": "file", "filename": str(p)}})
        return True
    except Exception as e:
        print(f"  blockdev-add failed: {e}", file=sys.stderr)
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["benign", "attack", "all"])
    ap.add_argument("--distro", choices=sorted(DISTROS))
    ap.add_argument("--distros", nargs="*", choices=sorted(DISTROS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=75.0, help="benign session length")
    ap.add_argument("--attack-minutes", type=float, default=30.0)
    ap.add_argument("--episodes", default=",".join(EPISODES),
                    help="comma-separated subset of attack episodes, in order")
    a = ap.parse_args(argv)
    if a.kind != "all":
        if not a.distro:
            ap.error("--distro is required")
        run_session(a.distro, a.kind, a.seed, a.minutes if a.kind == "benign" else a.attack_minutes,
                    tuple(a.episodes.split(",")))
        return 0
    distros = a.distros or sorted(DISTROS)

    def one(d):
        out = []
        for kind, mins in (("benign", a.minutes), ("attack", a.attack_minutes)):
            try:
                out.append(str(run_session(d, kind, a.seed, mins)))
            except Exception as e:
                print(f"[{d} {kind}] FAILED: {e!r}", flush=True)
        return out
    with ThreadPoolExecutor(max_workers=len(distros)) as ex:
        for r in ex.map(one, distros):
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
