"""
IUDBMS - Device traffic simulator.

Produces two artefacts per session, mirroring exactly what the real capture
layer collects on Linux:

  1. usbmon-format text lines  -> what /sys/kernel/debug/usb/usbmon/<bus>u emits
  2. a descriptor record       -> what /sys/bus/usb/devices/<dev>/ exposes at
                                  enumeration (VID/PID/class/interfaces/endpoints)

Only the simulator is synthetic. Everything downstream (parser, features,
scoring, enforcement) consumes the real on-wire formats, so swapping the
simulator for a live capture is a one-line change in run_demo.py.
"""

import random
import numpy as np

# transfer type letters as used by usbmon: C=control Z=iso I=interrupt B=bulk
CTRL, ISO, INTR, BULK = "C", "Z", "I", "B"

BENIGN_PROFILES = ["keyboard", "mouse", "mass_storage", "webcam", "audio"]
ATTACK_PROFILES = ["hid_injection", "identity_switch", "staged_delay", "class_exfil",
                   "mimicry"]

VENDORS = {
    "keyboard": [(0x046D, 0xC31C), (0x413C, 0x2113), (0x1A2C, 0x2D23)],
    "mouse": [(0x046D, 0xC077), (0x093A, 0x2510)],
    "mass_storage": [(0x0781, 0x5583), (0x0951, 0x1666), (0x090C, 0x1000)],
    "webcam": [(0x046D, 0x0825), (0x1BCF, 0x2C99)],
    "audio": [(0x0B05, 0x1A00), (0x046D, 0x0A38)],
}


def _hid_report(rng, code=None):
    """8-byte boot-protocol keyboard report."""
    mod = rng.choice([0, 0, 0, 2, 32])
    key = code if code is not None else rng.randint(4, 40)
    return bytes([mod, 0, key, 0, 0, 0, 0, 0])


def _mouse_report(rng):
    return bytes([rng.randint(0, 3), rng.randint(0, 255), rng.randint(0, 255), 0])


class Session:
    def __init__(self, sid, profile, label, descriptor, lines, onset=None, duration=0.0):
        self.sid = sid
        self.profile = profile
        self.label = label              # 0 benign, 1 malicious
        self.descriptor = descriptor
        self.lines = lines              # list[str] usbmon text
        self.onset = onset              # seconds from enumeration, or None
        self.duration = duration


def _descriptor(profile_name, bus, devnum, rng, interfaces):
    vid, pid = rng.choice(VENDORS[profile_name])
    return {
        "bus": bus,
        "devnum": devnum,
        "vid": vid,
        "pid": pid,
        "bDeviceClass": 0x00,
        "serial_present": int(rng.random() < 0.7),
        "manufacturer": profile_name,
        "interfaces": interfaces,
    }


def _iface(cls, sub, proto, eps):
    return {"bInterfaceClass": cls, "bInterfaceSubClass": sub,
            "bInterfaceProtocol": proto, "endpoints": eps}


def _fmt(tag, ts_us, evt, ttype, direction, bus, dev, ep, status, payload):
    data = payload.hex() if payload else ""
    return (f"{tag:016x} {ts_us} {evt} {ttype}{direction}:{bus:03d}:{dev:03d}:{ep} "
            f"{status} {len(payload) if payload else 0}"
            + (f" = {data}" if data else ""))


class Simulator:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._tag = 0xFFFF880100000000
        self._devnum = 2

    def _next_tag(self):
        self._tag += 0x40
        return self._tag

    # ---------------- benign generators ----------------

    def _human_typing(self, t, dur, bus, dev, lines, ep=1):
        """Human keystrokes: log-normal IATs, pauses, bursts, high jitter."""
        end = t + dur
        while t < end:
            # a burst of a word, then a think-pause
            for _ in range(self.rng.randint(3, 9)):
                iat = float(self.np_rng.lognormal(mean=-1.9, sigma=0.55))  # ~150 ms, heavy tail
                t += min(iat, 1.2)
                if t >= end:
                    break
                lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", INTR, "i",
                                  bus, dev, ep, "0:8", _hid_report(self.rng)))
            t += self.rng.uniform(0.4, 3.0)  # human pause
        return t

    def _mouse_motion(self, t, dur, bus, dev, lines, ep=1):
        end = t + dur
        while t < end:
            for _ in range(self.rng.randint(10, 60)):        # movement burst
                t += max(0.002, self.np_rng.normal(0.010, 0.004))
                if t >= end:
                    break
                lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", INTR, "i",
                                  bus, dev, ep, "0:4", _mouse_report(self.rng)))
            t += self.rng.uniform(0.3, 2.5)
        return t

    def _bulk_io(self, t, dur, bus, dev, lines, ep_in=1, ep_out=2, rate=(10, 60)):
        end = t + dur
        while t < end:
            n = self.rng.randint(*rate)
            for _ in range(n):
                t += max(0.0002, self.np_rng.normal(0.0016, 0.0006))
                if t >= end:
                    break
                out = self.rng.random() < 0.35
                size = self.rng.choice([512, 1024, 2048, 4096])
                lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", BULK,
                                  "o" if out else "i", bus, dev,
                                  ep_out if out else ep_in,
                                  f"0:{size}", bytes(self.np_rng.integers(0, 255, 16, dtype=np.uint8))))
            t += self.rng.uniform(0.5, 4.0)   # idle between file operations
        return t

    def _iso_stream(self, t, dur, bus, dev, lines, ep=3, period=0.008, size=3072):
        end = t + dur
        while t < end:
            t += period * self.rng.uniform(0.95, 1.05)
            lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", ISO, "i", bus, dev, ep,
                              f"0:{size}", bytes(self.np_rng.integers(0, 255, 12, dtype=np.uint8))))
        return t

    # ---------------- attack generators ----------------

    def _injected_typing(self, t, dur, bus, dev, lines, ep=1, mean_iat=0.0045, jitter=0.14):
        """Programmatic HID injection: fast, near-constant IAT, no pauses,
        script-like scancode sequence (low novelty, repeating structure)."""
        end = t + dur
        script = [self.rng.randint(4, 40) for _ in range(self.rng.randint(20, 40))]
        i = 0
        while t < end:
            t += max(0.0008, self.np_rng.normal(mean_iat, mean_iat * jitter))
            lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", INTR, "i",
                              bus, dev, ep, "0:8", _hid_report(self.rng, script[i % len(script)])))
            i += 1
            if i % 200 == 0:
                t += self.rng.uniform(0.05, 0.20)   # GUI-launch wait, still far shorter than human
        return t

    def _mimicry_typing(self, t, dur, bus, dev, lines, ep=1):
        """Adaptive adversary: the payload throttles itself to the SAME lognormal
        inter-arrival distribution as a human operator, so every timing feature
        in the window is statistically indistinguishable from benign typing.
        This is the case identity- and timing-only defences both miss.

        What it cannot cheaply fake is structure: the keystrokes come from a
        fixed script (low payload novelty, high repeat ratio) and the cadence
        never stops for the multi-second idles a real user produces. Both tells
        are weak per window and only separate over a sequence of windows, which
        is the burden Stage 2 is designed to carry."""
        end = t + dur
        script = [self.rng.randint(4, 40) for _ in range(self.rng.randint(24, 48))]
        i = 0
        while t < end:
            for _ in range(self.rng.randint(3, 9)):
                iat = float(self.np_rng.lognormal(mean=-1.9, sigma=0.55))
                t += min(iat, 1.2)
                if t >= end:
                    break
                lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", INTR, "i",
                                  bus, dev, ep, "0:8",
                                  _hid_report(self.rng, script[i % len(script)])))
                i += 1
            t += self.rng.uniform(0.35, 1.1)       # human pauses are 0.4-3.0 s
        return t

    def _covert_bulk_out(self, t, dur, bus, dev, lines, ep=2):
        """HID-declared device pushing bulk OUT payloads: class/behaviour mismatch."""
        end = t + dur
        while t < end:
            t += max(0.0008, self.np_rng.normal(0.0035, 0.0006))
            lines.append(_fmt(self._next_tag(), int(t * 1e6), "C", BULK, "o", bus, dev, ep,
                              "0:2048", bytes(self.np_rng.integers(0, 255, 16, dtype=np.uint8))))
        return t

    # ---------------- session assembly ----------------

    def session(self, sid, profile, duration=None):
        rng = self.rng
        bus = rng.randint(1, 3)
        dev = self._devnum
        self._devnum = 2 + (self._devnum - 1) % 120
        lines, t, onset = [], 0.0, None
        # One duration distribution for every profile. If malicious sessions
        # ran longer than benign ones the models could separate them on session
        # length alone, which would be leakage rather than detection.
        dur = duration or rng.uniform(40, 85)

        if profile == "keyboard":
            d = _descriptor("keyboard", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x01, [{"addr": 0x81, "type": "intr", "interval": 10}])])
            self._human_typing(t, dur, bus, dev, lines)
            label = 0

        elif profile == "mouse":
            d = _descriptor("mouse", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x02, [{"addr": 0x81, "type": "intr", "interval": 8}])])
            self._mouse_motion(t, dur, bus, dev, lines)
            label = 0

        elif profile == "mass_storage":
            d = _descriptor("mass_storage", bus, dev, rng,
                            [_iface(0x08, 0x06, 0x50, [{"addr": 0x81, "type": "bulk"},
                                                       {"addr": 0x02, "type": "bulk"}])])
            self._bulk_io(t, dur, bus, dev, lines)
            label = 0

        elif profile == "webcam":
            d = _descriptor("webcam", bus, dev, rng,
                            [_iface(0x0E, 0x01, 0x00, [{"addr": 0x83, "type": "iso"}])])
            self._iso_stream(t, dur, bus, dev, lines)
            label = 0

        elif profile == "audio":
            d = _descriptor("audio", bus, dev, rng,
                            [_iface(0x01, 0x02, 0x00, [{"addr": 0x83, "type": "iso"}])])
            self._iso_stream(t, dur, bus, dev, lines, period=0.005, size=192)
            label = 0

        elif profile == "hid_injection":
            # Descriptor is INDISTINGUISHABLE from a real keyboard - defeats
            # whitelisting (USBGuard) and identity-based defences entirely.
            d = _descriptor("keyboard", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x01, [{"addr": 0x81, "type": "intr", "interval": 10}])])
            onset = rng.uniform(1.5, 4.0)
            self._human_typing(0.0, onset, bus, dev, lines)
            self._injected_typing(onset, dur - onset, bus, dev, lines)
            label = 1

        elif profile == "identity_switch":
            # Enumerates as storage, later exposes an HID interface and injects.
            d = _descriptor("mass_storage", bus, dev, rng,
                            [_iface(0x08, 0x06, 0x50, [{"addr": 0x81, "type": "bulk"},
                                                       {"addr": 0x02, "type": "bulk"}])])
            onset = rng.uniform(15, 30)
            self._bulk_io(0.0, onset, bus, dev, lines)
            d["interfaces"].append(_iface(0x03, 0x01, 0x01,
                                          [{"addr": 0x83, "type": "intr", "interval": 10}]))
            d["late_iface_ts"] = onset
            self._injected_typing(onset, dur - onset, bus, dev, lines, ep=3)
            label = 1

        elif profile == "staged_delay":
            # Passes as a human-driven keyboard for a long dwell, then fires.
            d = _descriptor("keyboard", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x01, [{"addr": 0x81, "type": "intr", "interval": 10}])])
            dur = rng.uniform(65, 85)
            onset = rng.uniform(30, 45)
            self._human_typing(0.0, onset, bus, dev, lines)
            self._injected_typing(onset, dur - onset, bus, dev, lines, mean_iat=0.004, jitter=0.25)
            label = 1

        elif profile == "mimicry":
            # Hardest case in the corpus: correct keyboard descriptor AND
            # human-calibrated inter-arrival timing. Only structural and
            # long-horizon cues remain.
            d = _descriptor("keyboard", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x01, [{"addr": 0x81, "type": "intr", "interval": 10}])])
            onset = rng.uniform(4, 12)
            self._human_typing(0.0, onset, bus, dev, lines)
            self._mimicry_typing(onset, dur - onset, bus, dev, lines)
            label = 1

        else:  # class_exfil
            d = _descriptor("keyboard", bus, dev, rng,
                            [_iface(0x03, 0x01, 0x01, [{"addr": 0x81, "type": "intr", "interval": 10}])])
            onset = rng.uniform(5, 12)
            self._human_typing(0.0, onset, bus, dev, lines)
            self._covert_bulk_out(onset, dur - onset, bus, dev, lines)
            label = 1

        lines.sort(key=lambda ln: int(ln.split()[1]))
        return Session(sid, profile, label, d, lines, onset, dur)

    def plan(self, n_benign=120, n_attack=80):
        """Session (id, profile) schedule, so the caller can generate, parse and
        discard one session at a time rather than holding every URB in memory."""
        items = [(f"S{i:04d}", self.rng.choice(BENIGN_PROFILES)) for i in range(n_benign)]
        items += [(f"A{j:04d}", self.rng.choice(ATTACK_PROFILES)) for j in range(n_attack)]
        self.rng.shuffle(items)
        return items

    def corpus(self, n_benign=120, n_attack=80):
        sessions = []
        for i in range(n_benign):
            sessions.append(self.session(f"S{i:04d}", self.rng.choice(BENIGN_PROFILES)))
        for j in range(n_attack):
            sessions.append(self.session(f"A{j:04d}", self.rng.choice(ATTACK_PROFILES)))
        self.rng.shuffle(sessions)
        return sessions
