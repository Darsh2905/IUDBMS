"""
IUDBMS - Layer 1 parser.

Converts raw usbmon text lines into normalised URB event records. This is the
same code path used for a live capture:

    cat /sys/kernel/debug/usb/usbmon/0u | python3 -m iudbms.usbmon_parser

usbmon line grammar (Documentation/usb/usbmon.rst):

  <urb-tag> <timestamp-us> <event-type> <type><dir>:<bus>:<dev>:<ep> <status> <len> [= data]

  event-type : S submission, C callback/completion, E error
  type       : C control, Z isochronous, I interrupt, B bulk
  dir        : i device-to-host, o host-to-device
"""

import re
from dataclasses import dataclass

LINE_RE = re.compile(
    r"^(?P<tag>[0-9a-fA-F]+)\s+"
    r"(?P<ts>\d+)\s+"
    r"(?P<evt>[SCE])\s+"
    r"(?P<ttype>[CZIB])(?P<dir>[io]):(?P<bus>\d+):(?P<dev>\d+):(?P<ep>\d+)\s+"
    r"(?P<status>-?\d+)(?::(?P<blen>\d+))?\s+"
    r"(?P<dlen>\d+)"
    r"(?:\s+=\s+(?P<data>[0-9a-fA-F ]+))?\s*$"
)

TTYPE = {"C": "control", "Z": "iso", "I": "intr", "B": "bulk"}


@dataclass(slots=True)
class URBEvent:
    ts: float           # seconds
    evt: str            # S / C / E
    ttype: str          # control / iso / intr / bulk
    direction: str      # in / out
    bus: int
    dev: int
    ep: int
    status: int
    length: int
    data: bytes

    @property
    def device_key(self):
        return f"{self.bus}-{self.dev}"


def parse_line(line):
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    length = int(g["blen"]) if g["blen"] is not None else int(g["dlen"])
    data = bytes.fromhex(g["data"].replace(" ", "")) if g.get("data") else b""
    return URBEvent(
        ts=int(g["ts"]) / 1e6,
        evt=g["evt"],
        ttype=TTYPE[g["ttype"]],
        direction="in" if g["dir"] == "i" else "out",
        bus=int(g["bus"]),
        dev=int(g["dev"]),
        ep=int(g["ep"]),
        status=int(g["status"]),
        length=length,
        data=data,
    )


def parse(lines):
    out = []
    for ln in lines:
        e = parse_line(ln)
        if e is not None:
            out.append(e)
    return out
