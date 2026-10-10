"""
Minimal QEMU Machine Protocol client: the hypervisor's own control socket.

Used for three things only: injecting input into an emulated USB HID device
(``input-send-event`` with an explicit ``device``, the same mechanism VM
automation tools use to type into a guest), hot-plugging devices
(``device_add`` / ``device_del``), and asking what is attached.
"""

from __future__ import annotations

import json
import socket
import threading
import time


class QMPError(RuntimeError):
    pass


class QMP:
    def __init__(self, path: str, timeout: float = 30.0):
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except OSError:
                self.sock.close()
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        self.sock.settimeout(timeout)
        self._buf = b""
        self._lock = threading.Lock()                  # several input threads share one socket
        self.events: list[dict] = []
        self._recv()                                   # greeting
        self.cmd("qmp_capabilities")

    def _recv(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise QMPError("QMP socket closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def cmd(self, name: str, **args) -> dict:
        msg = {"execute": name}
        if args:
            msg["arguments"] = args
        with self._lock:
            self.sock.sendall(json.dumps(msg).encode() + b"\n")
            while True:
                r = self._recv()
                if "event" in r:                       # asynchronous; keep, keep reading
                    self.events.append(r)
                    continue
                if "error" in r:
                    raise QMPError(f"{name}: {r['error'].get('desc', r['error'])}")
                return r.get("return", {})

    # ------------------------------------------------------------ input ---
    def key(self, device: str, qcode: str, down: bool) -> None:
        self.cmd("input-send-event", device=device, events=[
            {"type": "key", "data": {"down": down, "key": {"type": "qcode", "data": qcode}}}])

    def _send(self, display: str | None, events: list) -> None:
        """Send input events to the devices bound to ``display`` (a display device id).

        input-send-event's ``device`` names the DISPLAY whose console is targeted, not
        the input device; events go to the input device bound to that console whose
        type matches. ``None`` routes to QEMU's active handler for the event type (used
        for usb-mouse, which cannot be bound to a display).
        """
        if display is None:
            self.cmd("input-send-event", events=events)
        else:
            self.cmd("input-send-event", device=display, events=events)

    def key(self, display: str | None, qcode: str, down: bool) -> None:
        self._send(display, [{"type": "key", "data": {"down": down,
                                                      "key": {"type": "qcode", "data": qcode}}}])

    def rel(self, display: str | None, dx: int, dy: int) -> None:
        ev = []
        if dx:
            ev.append({"type": "rel", "data": {"axis": "x", "value": int(dx)}})
        if dy:
            ev.append({"type": "rel", "data": {"axis": "y", "value": int(dy)}})
        if ev:
            self._send(display, ev)

    def abs(self, display: str | None, x: int, y: int) -> None:
        # absolute axes are scaled 0..0x7fff by QEMU
        self._send(display, [{"type": "abs", "data": {"axis": "x", "value": int(x)}},
                             {"type": "abs", "data": {"axis": "y", "value": int(y)}}])

    def button(self, display: str | None, button: str, down: bool) -> None:
        self._send(display, [{"type": "btn", "data": {"down": down, "button": button}}])

    # ---------------------------------------------------------- hotplug ---
    def device_add(self, driver: str, **props) -> None:
        self.cmd("device_add", driver=driver, **props)

    def device_del(self, dev_id: str, timeout: float = 10.0) -> None:
        self.cmd("device_del", id=dev_id)
        deadline = time.time() + timeout
        while time.time() < deadline:                  # wait for DEVICE_DELETED
            if any(e.get("event") == "DEVICE_DELETED"
                   and e.get("data", {}).get("device") == dev_id for e in self.events):
                return
            with self._lock:
                try:
                    self.sock.settimeout(0.5)
                    self.events.append(self._recv())
                except (socket.timeout, TimeoutError):
                    pass
                finally:
                    self.sock.settimeout(30.0)
        raise QMPError(f"device_del {dev_id}: no DEVICE_DELETED within {timeout}s")

    def usb_devices(self) -> list[str]:
        r = self.cmd("human-monitor-command", **{"command-line": "info usb"})
        return [ln.strip() for ln in r.splitlines() if ln.strip()]

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
