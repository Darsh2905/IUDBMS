"""
Live lab: a real Linux kernel, real usbmon, the deployed detector, one page.

    python3 -m iudbms.api.lab [--port 8001] [--distro debian-13]
    make lab                                   # then open http://localhost:8001

Boots a headless Linux guest under QEMU (the testbed's), streams ITS usbmon over SSH
into the deployed ``LiveMonitor`` / ``StreamingDetector`` -- the code the systemd monitor
runs -- and serves the verdicts as server-sent events. Buttons on the page drive the
guest's emulated USB keyboards and mouse through QMP: a person typing (replayed
keystroke timing from 1,972 real typists), a hot-plugged keystroke injector, and the
two cases the system does not catch.

What is real: the kernel, its xHCI and HID drivers, the usbmon traffic, the models, the
threshold, the 4-of-6 gate, the speed rule. What is not: the devices (QEMU's emulation,
driven by recorded input, not hardware and not a person at a keyboard). Enforcement is
dry-run. Nothing in this module scores anything itself.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..capture.usbmon import UsbmonSource
from ..live import LiveMonitor
from ..live.cli import _load_descriptor_snapshot
from .demo import _detector

REPO = Path(__file__).resolve().parents[3]
PAGE = Path(str(files("iudbms.api").joinpath("static", "lab.html")))
GUEST_DIR = "/var/tmp/tb"
DEVICES = ("kbd", "mouse", "storage", "net")
HISTORY = 240


def _testbed():
    """The testbed is a repository package, not part of the installed one."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from testbed import human_input as hi
    from testbed import workloads as wl
    from testbed.vm import KEY, Guest
    return hi, wl, Guest, KEY


class Hub:
    """Fan-out of events to every connected page, plus enough history to redraw one."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.subs: list[queue.Queue] = []
        self.devices: dict[str, dict] = {}
        self.hist: dict[str, deque] = {}
        self.log: deque = deque(maxlen=60)
        self.status = {"state": "starting", "message": "Starting"}
        self.marks: list[dict] = []

    def emit(self, kind: str, data: dict) -> None:
        with self.lock:
            if kind == "window":
                self.hist.setdefault(data["device"], deque(maxlen=HISTORY)).append(data)
            elif kind == "log":
                self.log.append(data)
            elif kind == "mark":
                self.marks.append(data)
            elif kind == "status":
                self.status = data
            for q in list(self.subs):
                q.put((kind, data))

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.status, "devices": self.devices, "log": list(self.log),
                    "marks": self.marks,
                    "history": {k: list(v) for k, v in self.hist.items()}}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


class Lab:
    def __init__(self, distro: str) -> None:
        self.distro = distro
        self.hub = Hub()
        self.guest = None
        self.monitor: LiveMonitor | None = None
        self.cache: dict = {}
        self.cfg: dict = {}
        self.stop = threading.Event()
        self.atk_lock = threading.Lock()        # one hot-plugged attacker at a time (shared display)
        self.actions: dict[str, threading.Event] = {}
        self._cap: subprocess.Popen | None = None
        self._typing = None
        self._seq = 0

    # ------------------------------------------------------------------ boot
    def say(self, msg: str, level: str = "info") -> None:
        self.hub.emit("log", {"t": time.time(), "msg": msg, "level": level})

    def status(self, state: str, message: str) -> None:
        self.hub.emit("status", {"state": state, "message": message})

    def start(self) -> None:
        threading.Thread(target=self._boot, name="lab-boot", daemon=True).start()

    def _boot(self) -> None:
        try:
            hi, wl, Guest, KEY = _testbed()
            det, self.cfg = _detector()
            run = REPO / "out" / "lab" / self.distro
            for stale in ("disk.qcow2", "seed.iso"):          # every boot starts from the clean image
                (run / stale).unlink(missing_ok=True)
            self.status("booting", f"Booting {self.distro} under QEMU. About a minute.")
            g = self.guest = Guest(self.distro, DEVICES, run)
            g.start()
            self.status("booting", "Kernel is up. Starting usbmon.")
            g.ssh(f"mkdir -p {GUEST_DIR} && sudo chmod 777 {GUEST_DIR}")
            g.scp_to(REPO / "testbed" / "guest_descriptors.py", f"{GUEST_DIR}/guest_descriptors.py")
            g.ssh("modprobe usbmon; mountpoint -q /sys/kernel/debug || mount -t debugfs none "
                  "/sys/kernel/debug; for e in /dev/input/event*; do setsid nohup cat $e "
                  ">/dev/null 2>&1 </dev/null & done; true", sudo=True)
            self._refresh()
            self._cap = subprocess.Popen(
                ["ssh", "-i", str(KEY), "-p", str(g.ssh_port), "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                 "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=5", "tb@127.0.0.1",
                 "sudo cat /sys/kernel/debug/usb/usbmon/0u"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            source = UsbmonSource(stream=self._cap.stdout)
            self.monitor = LiveMonitor(det, source, lambda k: self.cache.get(k), window=1.0)
            threading.Thread(target=self._descriptors, name="lab-desc", daemon=True).start()
            threading.Thread(target=self._monitor, name="lab-monitor", daemon=True).start()
            self._typing = None
            self.status("ready", f"Live. Linux {self._uname()} · usbmon streaming")
            self.say("Lab is live. The guest's USB bus is being scored once a second.")
        except BaseException as e:               # surfaced on the page, never swallowed
            self.status("error", f"{type(e).__name__}: {e}")
            self.say(f"Lab failed to start: {e}", "error")

    def _uname(self) -> str:
        try:
            return self.guest.ssh("uname -r", timeout=15).stdout.strip()
        except Exception:
            return "?"

    # ----------------------------------------------------------- descriptors
    def _refresh(self) -> None:
        r = self.guest.ssh(f"sudo python3 {GUEST_DIR}/guest_descriptors.py", timeout=60)
        p = self.guest.run_dir / "descriptors.json"
        p.write_text(r.stdout)
        self.cache.update(_load_descriptor_snapshot(p))

    def _descriptors(self) -> None:
        while not self.stop.wait(2.0):
            try:
                self._refresh()
            except Exception:
                pass

    # --------------------------------------------------------------- monitor
    def _info(self, key: str) -> dict:
        d = self.monitor.detector.devices[key].descriptor
        return {"id": key, "product": d.product, "vid": f"{d.vid:04x}", "pid": f"{d.pid:04x}",
                "interfaces": [{"cls": i.bInterfaceClass, "sub": i.bInterfaceSubClass,
                                "proto": i.bInterfaceProtocol,
                                "pages": list(i.hid_usage_pages) if i.hid_usage_pages is not None else None}
                               for i in d.interfaces]}

    def _monitor(self) -> None:
        cfg = self.cfg
        try:
            for verdicts in self.monitor.run():
                for v in verdicts:
                    if v.device_id not in self.hub.devices:
                        info = self._info(v.device_id)
                        with self.hub.lock:
                            self.hub.devices[v.device_id] = info
                        self.hub.emit("device", info)
                        self.say(f"Kernel enumerated {info['product'] or 'a device'} "
                                 f"({info['vid']}:{info['pid']}) as {v.device_id}")
                    a = v.action
                    w = {"device": v.device_id, "t": float(v.t_start), "s1": float(v.stage1),
                         "s2": float(v.stage2), "hits": v.gate_hits, "n": v.gate_n,
                         "state": v.trust_state, "events": int(v.features[0]),
                         "over": bool(v.stage2 >= cfg["threshold"]),
                         "evidence": [[k, float(x)] for k, x in
                                      sorted(v.evidence, key=lambda kv: -abs(kv[1]))[:3]],
                         "action": None if a is None else
                         {"name": a.action, "severity": a.severity,
                          "fired_by": (a.rule or {}).get("fired_by"), "rule": a.rule}}
                    self.hub.emit("window", w)
                    if a is not None:
                        self.say(f"{a.action} {v.device_id}: raised by the "
                                 f"{(a.rule or {}).get('fired_by') or 'model'} (dry run, nothing blocked)",
                                 "alarm")
        except Exception as e:
            self.status("error", f"monitor stopped: {e!r}")

    def stream_t(self) -> float | None:
        m = self.monitor
        return None if m is None or m.time_origin is None else time.monotonic() - m.time_origin

    # --------------------------------------------------------------- actions
    def _mark(self, device: str | None, what: str) -> None:
        t = self.stream_t()
        if t is not None:
            self.hub.emit("mark", {"device": device, "t": t, "what": what})

    def _typists(self):
        if self._typing is None:
            hi, *_ = _testbed()
            self._typing = hi.aalto_sessions(limit=40)
        return self._typing

    def _plug(self, stop: threading.Event) -> str | None:
        g = self.guest
        self._seq += 1
        dev = f"atk{self._seq}"
        before = set(self.cache)
        g.qmp.device_add("usb-kbd", id=dev, bus="xhci.0", display="gpu1", usb_version=1)
        self.say("A new keyboard was plugged in.")
        for _ in range(30):
            if stop.wait(0.5):
                break
            self._refresh()
            new = set(self.cache) - before
            if new:
                key = sorted(new)[0]
                time.sleep(1.5)
                return key, dev
        return None, dev

    def _unplug(self, dev: str) -> None:
        try:
            self.guest.qmp.device_del(dev)
            self.say("The keyboard was unplugged.")
        except Exception:
            pass

    def run_action(self, name: str) -> None:
        hi, wl, *_ = _testbed()
        g, stop = self.guest, threading.Event()
        self.actions[name] = stop
        qmp = g.qmp

        def human():
            s = random.choice(self._typists())
            strokes = [k for k in s.strokes if k.down < 45.0]
            self.say(f"A person types for about 45 s (replayed timing of typist {s.user}).")
            self._mark("1-2", "typing")
            wl.replay_strokes(qmp, "gpu0", strokes, stop)

        def mouse():
            self.say("The mouse moves for 25 s.")
            end, x = time.monotonic() + 25, random.Random()
            while time.monotonic() < end and not stop.is_set():
                qmp.rel(None, x.randint(-12, 12), x.randint(-12, 12))
                if x.random() < 0.02:
                    qmp.button(None, "left", True)
                    qmp.button(None, "left", False)
                stop.wait(0.02)

        def storage():
            self.say("The USB drive is read hard for 15 s.")
            g.ssh("d=$(for b in /sys/block/sd*; do readlink -f $b | grep -q usb && echo /dev/$(basename $b) "
                  "&& break; done); end=$(( $(date +%s) + 15 )); while [ $(date +%s) -lt $end ]; do "
                  "dd if=$d of=/dev/null bs=64k count=64 iflag=direct status=none; done",
                  sudo=True, check=False, timeout=60)

        def attacker(kind: str):
            if not self.atk_lock.acquire(blocking=False):
                self.say("One attacker at a time. Wait for the last one to finish.", "warn")
                return
            try:
                key, dev = self._plug(stop)
                if key is None:
                    self.say("The new keyboard never showed up in sysfs.", "error")
                    self._unplug(dev)
                    return
                timing = wl.human_timing(random.choice(self._typists()))
                if kind == "inject":
                    self.say("Keystroke injection: the payload is typed at machine speed.", "alarm")
                    self._mark(key, "attack starts")
                    wl.type_text(qmp, "gpu1", wl.INJECT_TEXT, stop)
                    stop.wait(6)
                elif kind == "staged":
                    self.say("Staged injector: plugged in and silent for 25 s, then it types.")
                    stop.wait(25)
                    self._mark(key, "attack starts")
                    wl.type_text(qmp, "gpu1", wl.INJECT_TEXT, stop)
                    stop.wait(6)
                elif kind == "slow":
                    self.say("Payload typed once at human speed, then silence. Not detected by anything here.", "warn")
                    self._mark(key, "attack starts")
                    wl.type_text(qmp, "gpu1", (wl.PANGRAM * 4)[:160], stop, timing=timing)
                    stop.wait(45)
                elif kind == "loop":
                    self.say("Human speed replay loop for 60 s. Missed on real kernels.", "warn")
                    self._mark(key, "attack starts")
                    end = time.monotonic() + 60
                    while time.monotonic() < end and not stop.is_set():
                        wl.type_text(qmp, "gpu1", wl.LOOP_TEXT, stop, timing=timing)
                self._unplug(dev)
            finally:
                self.atk_lock.release()

        try:
            {"human": human, "mouse": mouse, "storage": storage,
             "inject": lambda: attacker("inject"), "staged": lambda: attacker("staged"),
             "slow": lambda: attacker("slow"), "loop": lambda: attacker("loop")}[name]()
        except Exception as e:
            self.say(f"Action {name} failed: {e!r}", "error")
        finally:
            self.actions.pop(name, None)

    def shutdown(self) -> None:
        self.stop.set()
        for e in list(self.actions.values()):
            e.set()
        if self.monitor is not None:
            self.monitor.stop()
        if self._cap is not None:
            self._cap.kill()
        if self.guest is not None:
            self.guest.stop()


# ---------------------------------------------------------------------------

LAB: Lab | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global LAB
    LAB = Lab(DISTRO)
    LAB.start()
    yield
    LAB.shutdown()


DISTRO = "debian-13"
app = FastAPI(title="IUDBMS live lab", lifespan=lifespan)


class Action(BaseModel):
    name: str


ACTIONS = {"human", "mouse", "storage", "inject", "staged", "slow", "loop"}


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@app.get("/lab/state")
def state() -> dict:
    return LAB.hub.snapshot() | {"threshold": LAB.cfg.get("threshold"), "k": LAB.cfg.get("k"),
                                 "n": LAB.cfg.get("n"),
                                 "human_max_rps": (LAB.cfg.get("capability_rule") or {}).get("human_max_rps")}


@app.post("/lab/action")
def action(a: Action) -> dict:
    if a.name not in ACTIONS:
        raise HTTPException(404, f"unknown action {a.name!r}")
    if LAB.hub.status["state"] != "ready":
        raise HTTPException(409, "the lab is not ready yet")
    if a.name in LAB.actions:
        raise HTTPException(409, f"{a.name} is already running")
    threading.Thread(target=LAB.run_action, args=(a.name,), daemon=True).start()
    return {"started": a.name}


@app.post("/lab/stop")
def stop_all() -> dict:
    for e in list(LAB.actions.values()):
        e.set()
    return {"stopped": True}


@app.get("/lab/stream")
def stream():
    q = LAB.hub.subscribe()

    def gen():
        try:
            yield f"event: state\ndata: {json.dumps(state())}\n\n"
            while True:
                try:
                    kind, data = q.get(timeout=10)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        finally:
            LAB.hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def main(argv=None) -> int:
    global DISTRO
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--distro", default="debian-13")
    a = ap.parse_args(argv)
    DISTRO = a.distro
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
