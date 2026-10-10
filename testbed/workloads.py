"""
What drives each device during a testbed session.

Human input (keyboard, mouse, tablet) is REPLAYED from public recordings
(testbed.human_input) through QMP: every key-down and key-up lands at the recorded
offset, so the kernel sees real human press/release timing, rollover included.
Other devices are driven from inside the guest by a seeded schedule generated here
(storage reads/writes, network packets, serial output, audio playback).

Attack-shaped episodes run only inside these throw-away guests and type innocuous
text -- a pangram, never a command, and never the Enter key. What makes them
attacks is the traffic shape the kernel sees (machine-speed typing, a replayed
loop, typing then permanent silence, a storage device that comes back as a
keyboard), which is what the detector is being tested on.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

from .human_input import KeyStroke, MouseSession, TypingSession
from .qmp import QMP

#: Innocuous payloads. Lower-case letters, digits and spaces only.
PANGRAM = "the quick brown fox jumps over the lazy dog "
INJECT_TEXT = (PANGRAM + "0123456789 ") * 8
LOOP_TEXT = "pack my box with five dozen liquor jugs "

_CH = {" ": "spc", ".": "dot", ",": "comma", "-": "minus", "/": "slash", "'": "apostrophe"}


def char_qcode(c: str) -> str:
    if c.isalpha() or c.isdigit():
        return c.lower()
    return _CH[c]


@dataclass
class Log:
    """Host-side record of what was driven when (seconds since session start)."""
    t0: float = field(default_factory=time.perf_counter)
    rows: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, what: str, **kw) -> None:
        with self.lock:
            self.rows.append({"t": round(self.now(), 3), "what": what, **kw})


def _sleep_until(t_abs: float, stop: threading.Event) -> bool:
    """Sleep until perf_counter() == t_abs. False if stopped first."""
    while True:
        d = t_abs - time.perf_counter()
        if d <= 0:
            return not stop.is_set()
        if stop.wait(min(d, 0.05)):
            return False


def replay_strokes(qmp: QMP, dev: str, strokes: list[KeyStroke], stop: threading.Event,
                   speed: float = 1.0) -> int:
    """Press and release each key at its recorded offset. Returns keys sent."""
    evs = sorted([(k.down / speed, 0, k.key) for k in strokes] +
                 [(k.up / speed, 1, k.key) for k in strokes])
    held: set[str] = set()
    start, sent = time.perf_counter(), 0
    try:
        for t, kind, q in evs:
            if not _sleep_until(start + t, stop):
                break
            if kind == 0 and q not in held:
                qmp.key(dev, q, True)
                held.add(q)
                sent += 1
            elif kind == 1 and q in held:
                qmp.key(dev, q, False)
                held.discard(q)
    finally:
        for q in list(held):                   # never leave a key stuck down
            qmp.key(dev, q, False)
    return sent


def type_text(qmp: QMP, dev: str, text: str, stop: threading.Event, *, period: float = 0.008,
              timing: list[tuple[float, float]] | None = None) -> int:
    """Type ``text``. Machine speed (``period`` per edge) or a human timing template.

    ``timing`` is a list of (hold, gap-to-next-press) pairs taken from a real
    participant, cycled; it gives a scripted device human-shaped timing, which is
    exactly what the mimicry adversary does.
    """
    n = 0
    for i, c in enumerate(text):
        if stop.is_set():
            break
        q = char_qcode(c)
        hold, gap = (period, period) if timing is None else timing[i % len(timing)]
        qmp.key(dev, q, True)
        if stop.wait(hold):
            qmp.key(dev, q, False)
            break
        qmp.key(dev, q, False)
        n += 1
        if stop.wait(max(0.0, gap - hold) if timing is not None else gap):
            break
    return n


def human_timing(session: TypingSession, n: int = 400) -> list[tuple[float, float]]:
    """(hold, press-to-next-press) pairs from one real participant, gaps over 2 s dropped."""
    ks = session.strokes
    out = []
    for a, b in zip(ks, ks[1:]):
        gap = b.down - a.down
        if 0.02 < gap < 2.0:
            out.append((min(a.up - a.down, gap * 0.9), gap))
        if len(out) >= n:
            break
    return out


class HumanKeyboard(threading.Thread):
    """Replays participants back to back with idle gaps between them.

    Within a participant every interval is the recorded one, including the pauses
    between sentences. The gap BETWEEN participants is ours: uniform(gap_lo, gap_hi)
    seconds, standing in for the user doing something else.
    """

    def __init__(self, qmp: QMP, dev: str, sessions: list[TypingSession], log: Log,
                 rng: random.Random, until: float, gap: tuple[float, float] = (5.0, 90.0)):
        super().__init__(daemon=True)
        self.qmp, self.dev, self.sessions, self.log = qmp, dev, list(sessions), log
        self.rng, self.until, self.gap = rng, until, gap
        self.stop = threading.Event()
        self.keys = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            i = 0
            while not self.stop.is_set() and self.log.now() < self.until and self.sessions:
                left = self.until - self.log.now()
                fits = [x for x in self.sessions[i:] + self.sessions[:i] if x.duration < left]
                if not fits:
                    break                       # nobody's recording fits in what remains
                s = fits[0]
                i = (self.sessions.index(s) + 1) % len(self.sessions)
                self.log.add("type_begin", dev=self.dev, user=s.user, keys=len(s.strokes))
                self.keys += replay_strokes(self.qmp, self.dev, s.strokes, self.stop)
                self.log.add("type_end", dev=self.dev, user=s.user)
                if self.stop.wait(self.rng.uniform(*self.gap)):
                    break
        except Exception as e:                  # surfaced by the campaign, never swallowed
            self.error = e


class HumanPointer(threading.Thread):
    """Replays Balabit mouse sessions: relative motion (mouse) or absolute (tablet)."""

    def __init__(self, qmp: QMP, dev: str, sessions: list[MouseSession], log: Log,
                 until: float, mode: str = "rel", max_gap: float = 120.0):
        super().__init__(daemon=True)
        self.qmp, self.dev, self.sessions, self.log = qmp, dev, list(sessions), log
        self.until, self.mode, self.max_gap = until, mode, max_gap
        self.stop = threading.Event()
        self.events = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for s in self.sessions:
                if self.stop.is_set() or self.log.now() >= self.until:
                    break
                self.log.add("point_begin", dev=self.dev, user=s.user, mode=self.mode)
                self._one(s)
                self.log.add("point_end", dev=self.dev, user=s.user)
        except Exception as e:
            self.error = e

    def _one(self, s: MouseSession) -> None:
        sx = 0x7FFF / max(1, max(s.x))
        sy = 0x7FFF / max(1, max(s.y))
        start, shift, prev = time.perf_counter(), 0.0, None
        down = set()
        for t, x, y, b in zip(s.t, s.x, s.y, s.buttons):
            if prev is not None and t - prev[0] > self.max_gap:   # cap very long idle
                shift += (t - prev[0]) - self.max_gap
            if not _sleep_until(start + t - shift, self.stop) or self.log.now() >= self.until:
                break
            if self.mode == "rel" and prev is not None:
                dx, dy = x - prev[1], y - prev[2]
                while dx or dy:                                  # HID reports carry +/-127
                    sx_, sy_ = max(-127, min(127, dx)), max(-127, min(127, dy))
                    self.qmp.rel(self.dev, sx_, sy_)
                    dx, dy = dx - sx_, dy - sy_
            elif self.mode == "abs":
                self.qmp.abs(self.dev, int(x * sx), int(y * sy))
            btn, state = (b.split(":") + [""])[:2]
            name = {"Left": "left", "Right": "right"}.get(btn)
            if name and state == "Pressed" and name not in down:
                self.qmp.button(self.dev, name, True)
                down.add(name)
            elif name and state == "Released" and name in down:
                self.qmp.button(self.dev, name, False)
                down.discard(name)
            prev = (t, x, y)
            self.events += 1
        for name in list(down):
            self.qmp.button(self.dev, name, False)


# ----------------------------------------------------------------- guest side ---
GUEST_FIND = r'''
dev_for_driver() {  # first /dev/sdX whose USB interface is bound to driver $1
  for b in /sys/block/sd*; do
    p=$(readlink -f "$b/device")
    i=$(echo "$p" | grep -oE '/[0-9]+-[0-9.]+:[0-9]+\.[0-9]+/' | head -1 | tr -d /)
    [ -n "$i" ] || continue
    d=$(basename "$(readlink /sys/bus/usb/devices/$i/driver 2>/dev/null)" 2>/dev/null)
    [ "$d" = "$1" ] && { echo /dev/$(basename "$b"); return; }
  done
}
NETIF=""
for n in /sys/class/net/*; do
  d=$(basename "$(readlink "$n/device/driver" 2>/dev/null)" 2>/dev/null)
  case "$d" in cdc_*|rndis_host) NETIF=$(basename "$n");; esac
done
MSD=$(dev_for_driver usb-storage); UAS=$(dev_for_driver uas)
TTY=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
CARD=$(for c in /proc/asound/card[0-9]*; do [ -e "$c/usbid" ] && basename "$c" | tr -dc 0-9 && break; done)
echo "found msd=$MSD uas=$UAS net=$NETIF tty=$TTY card=$CARD aplay=$(command -v aplay)"
if [ -n "$NETIF" ]; then ip link set "$NETIF" up; ip addr add 10.0.9.15/24 dev "$NETIF" 2>/dev/null; fi
[ -n "$TTY" ] && stty -F "$TTY" 115200 raw 2>/dev/null
'''


def guest_schedule(seed: int, duration: float) -> str:
    """A bash script of storage / network / serial / audio activity with idle gaps.

    Every interval comes from this seeded generator, so a session is reproducible
    and the same schedule can be replayed on every distribution.
    """
    rng = random.Random(seed)
    lines = ["#!/bin/bash", "# generated by testbed.workloads.guest_schedule "
             f"(seed {seed}, {duration:.0f} s)", GUEST_FIND, "T0=$(date +%s)"]

    def at(t: float, cmd: str) -> None:
        lines.append(f"while [ $(( $(date +%s) - T0 )) -lt {int(t)} ]; do sleep 0.5; done; {cmd}")

    events = []
    t = rng.uniform(10, 40)
    while t < duration - 30:                        # removable storage: copy a file, read it back
        mb, off = rng.randint(1, 24), rng.randint(0, 200)
        dev = rng.choice(["$MSD", "$UAS"])
        events.append((t, f'[ -n "{dev}" ] && dd if=/dev/urandom of={dev} bs=1M count={mb} '
                          f'seek={off} oflag=direct conv=notrunc status=none; '
                          f'[ -n "{dev}" ] && dd if={dev} of=/dev/null bs=1M count={mb} skip={off} '
                          f'iflag=direct status=none'))
        t += rng.expovariate(1 / 90.0) + 5
    t = rng.uniform(5, 30)
    while t < duration - 30:                        # network: a burst of packets
        n = rng.randint(20, 400)
        events.append((t, f'[ -n "$NETIF" ] && for i in $(seq {n}); do '
                          f'echo "iudbms-testbed $i" > /dev/udp/10.0.9.2/9 2>/dev/null; '
                          f'sleep 0.0{rng.randint(1, 9)}; done'))
        t += rng.expovariate(1 / 60.0) + 3
    t = rng.uniform(5, 20)
    while t < duration - 10:                        # serial console: periodic log lines
        events.append((t, f'[ -n "$TTY" ] && echo "testbed log line t={int(t)} '
                          f'value={rng.randint(0, 9999)}" > "$TTY"'))
        t += rng.expovariate(1 / 20.0) + 1
    t = rng.uniform(30, 120)
    while t < duration - 60:                        # audio: play a clip (silence) if possible
        secs = rng.randint(5, 40)
        events.append((t, f'[ -n "$CARD" ] && command -v aplay >/dev/null && '
                          f'aplay -q -D plughw:$CARD -f S16_LE -r 48000 -c 2 -d {secs} /dev/zero'))
        t += secs + rng.expovariate(1 / 150.0) + 10
    for t, cmd in sorted(events):
        at(t, f"( {cmd} ) &")
    lines.append("wait")
    return "\n".join(lines) + "\n"
