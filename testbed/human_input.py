"""
Recorded human input from public datasets, fetched and normalised.

    python3 -m testbed.human_input fetch            # ~40 MB, cached under data/human/
    python3 -m testbed.human_input stats            # timing statistics, real vs simulator

Three sources, each used for what it actually contains:

* **Aalto 136M Keystrokes** (Dhakal et al., CHI 2018) -- 168k participants typing
  15 English sentences each in a browser, key press *and* release times at 1 ms.
  The only large free-text set with release times, which is what decides how many
  HID reports a keystroke produces (key-down, key-up, and rollover overlap).
  Research and non-commercial use with attribution. We fetch a seeded sample of
  participants by HTTP range requests (the archive is 1.5 GB) and never commit it.
* **CMU keystroke benchmark** (Killourhy & Maxion, DSN 2009) -- 51 subjects typing
  one password 400 times: hold and latency times. Narrow (one string) but a
  second, independent population for the timing distributions.
* **Balabit Mouse Dynamics Challenge** (2016) -- 10 users doing real remote-desktop
  work, mouse position and buttons over long sessions, idle gaps included.

What this module does NOT do: claim the replayed input is a live user. A
replay reproduces when keys went down and up; it does not reproduce what a person
would do next in response to the screen.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "human"

AALTO_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
BALABIT_URL = "https://codeload.github.com/balabit/Mouse-Dynamics-Challenge/zip/refs/heads/master"

#: Sampling plan for Aalto: RUNS random start points, RUN_LEN consecutive files each.
#: Participant ids are assigned in arrival order and carry no meaning, so a run of
#: neighbours is as representative as scattered picks and costs one request.
AALTO_SEED, AALTO_RUNS, AALTO_RUN_LEN = 2018, 40, 50


# ---------------------------------------------------------------- HTTP ranges ---
def _curl_range(url: str, a: int, b: int, attempts: int = 6) -> bytes:
    # curl, not urllib: python.org builds on macOS ship without a CA bundle, and the
    # same command works unchanged inside every guest distribution. Retries are done
    # here, not with curl --retry: retrying to stdout APPENDS the second transfer to
    # the partial first one (seen: 24.5 MB returned for a 20.2 MB range).
    want = b - a + 1
    for i in range(attempts):
        r = subprocess.run(["curl", "-sSfL", "--speed-limit", "5000", "--speed-time", "30",
                            "-r", f"{a}-{b}", url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == want:
            return r.stdout
        time.sleep(2 * (i + 1))
    raise IOError(f"range {a}-{b}: {attempts} attempts failed (last: rc {r.returncode}, "
                  f"{len(r.stdout)} of {want} bytes)")


def _content_length(url: str) -> int:
    r = subprocess.run(["curl", "-sSfLI", url], capture_output=True, text=True, check=True)
    lens = [int(l.split(":", 1)[1]) for l in r.stdout.splitlines()
            if l.lower().startswith("content-length:")]
    return lens[-1]


@dataclass
class ZipMember:
    name: str
    method: int
    csize: int
    usize: int
    offset: int


def remote_zip_index(url: str) -> list[ZipMember]:
    """Central directory of a remote ZIP/ZIP64 archive, read with two range requests."""
    size = _content_length(url)
    tail = _curl_range(url, max(0, size - 65536), size - 1)
    base = max(0, size - 65536)
    loc = tail.rfind(b"PK\x06\x07")
    if loc >= 0:  # ZIP64: the classic record's counts have overflowed
        (z64_off,) = struct.unpack("<Q", tail[loc + 8:loc + 16])
        rec = tail[z64_off - base:] if z64_off >= base else _curl_range(url, z64_off, z64_off + 55)
        n, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    else:
        i = tail.rfind(b"PK\x05\x06")
        n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    cd = _curl_range(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (meth, csize, usize, nl, el, cl, lho) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0],
            *struct.unpack("<II", cd[p + 20:p + 28]),
            *struct.unpack("<HHH", cd[p + 28:p + 34]),
            struct.unpack("<I", cd[p + 42:p + 46])[0])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra = cd[p + 46 + nl:p + 46 + nl + el]
        if 0xFFFFFFFF in (csize, usize, lho):  # ZIP64 extended information
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + hlen - hlen % 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if lho == 0xFFFFFFFF:
                        lho = vals.pop(0)
                q += 4 + hlen
        out.append(ZipMember(name, meth, csize, usize, lho))
        p += 46 + nl + el + cl
    if len(out) != n:
        raise IOError(f"central directory: parsed {len(out)} entries, header says {n}")
    return out


def _inflate(member: ZipMember, blob: bytes, at: int) -> bytes:
    """Decompress one member from a buffer holding its local header at offset ``at``."""
    if blob[at:at + 4] != b"PK\x03\x04":
        raise IOError(f"{member.name}: no local header at {member.offset}")
    nl, el = struct.unpack("<HH", blob[at + 26:at + 30])
    start = at + 30 + nl + el
    raw = blob[start:start + member.csize]
    data = zlib.decompress(raw, -15) if member.method == 8 else raw
    if len(data) != member.usize:
        raise IOError(f"{member.name}: inflated {len(data)} bytes, expected {member.usize}")
    return data


def fetch_aalto(dest: Path, seed: int = AALTO_SEED, runs: int = AALTO_RUNS,
                run_len: int = AALTO_RUN_LEN) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    print("aalto: reading the archive's central directory (~20 MB) ...", flush=True)
    idx = remote_zip_index(AALTO_URL)
    files = sorted((m for m in idx if m.name.endswith("_keystrokes.txt")), key=lambda m: m.offset)
    meta = [m for m in idx if "metadata" in m.name.lower()]
    print(f"aalto: {len(files)} participant files; {len(meta)} metadata file(s)")
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(len(files) - run_len), runs))
    got = []
    for r, s in enumerate(starts):
        run = files[s:s + run_len]
        a = run[0].offset
        b = run[-1].offset + 30 + 512 + run[-1].csize  # header + generous name/extra slack
        blob = _curl_range(AALTO_URL, a, b)
        for m in run:
            p = dest / Path(m.name).name
            if not p.exists():
                p.write_bytes(_inflate(m, blob, m.offset - a))
            got.append(p)
        print(f"  run {r + 1}/{runs}: {len(run)} participants", flush=True)
    for m in meta:  # participant metadata (keyboard type, WPM, ...)
        p = dest / Path(m.name).name
        if not p.exists():
            blob = _curl_range(AALTO_URL, m.offset, m.offset + 30 + 512 + m.csize)
            p.write_bytes(_inflate(m, blob, 0))
    (dest / "SAMPLE.json").write_text(json.dumps(
        {"source": AALTO_URL, "seed": seed, "runs": runs, "run_len": run_len,
         "participants": [p.name for p in got],
         "citation": "Dhakal, Feit, Kristensson, Oulasvirta. Observations on Typing "
                     "from 136 Million Keystrokes. CHI 2018.",
         "terms": "free for research and non-commercial use with attribution"}, indent=1))
    return got


def fetch_simple(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # write-then-rename: an interrupted download must not look like a finished one
        part = dest.with_name(dest.name + ".part")
        subprocess.run(["curl", "-sSfL", "--retry", "5", "--speed-limit", "2000", "--speed-time", "120", "-o", str(part), url], check=True)
        part.rename(dest)
    return dest


# -------------------------------------------------------------------- parsing ---
@dataclass
class KeyStroke:
    """One physical key: when it went down and up, in seconds from session start."""
    down: float
    up: float
    key: str            # QEMU qcode, e.g. "a", "shift", "spc", "ret"


@dataclass
class TypingSession:
    source: str
    user: str
    strokes: list[KeyStroke] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(k.up for k in self.strokes) - min(k.down for k in self.strokes)


# Browser KeyboardEvent.keyCode -> QEMU qcode. Letters/digits map directly; the rest
# is the set that actually occurs in transcription typing. Anything unmapped is
# dropped and counted, never guessed.
_KC = {8: "backspace", 9: "tab", 13: "ret", 16: "shift", 17: "ctrl", 18: "alt",
       20: "caps_lock", 32: "spc", 186: "semicolon", 187: "equal", 188: "comma",
       189: "minus", 190: "dot", 191: "slash", 192: "grave_accent", 219: "bracket_left",
       220: "backslash", 221: "bracket_right", 222: "apostrophe", 37: "left", 38: "up",
       39: "right", 40: "down", 46: "delete"}
_KC.update({c: chr(c).lower() for c in range(65, 91)})
_KC.update({c: str(c - 48) for c in range(48, 58)})


def parse_aalto(path: Path) -> list[TypingSession]:
    """One participant file -> one TypingSession per test section (sentence).

    Columns: PARTICIPANT_ID TEST_SECTION_ID SENTENCE USER_INPUT KEYSTROKE_ID
    PRESS_TIME RELEASE_TIME LETTER KEYCODE (tab-separated, epoch ms).
    """
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="latin-1")),
                               delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows:
        return []
    # One session per participant: the 15 sentences in order, with the real gaps
    # between them (reading the next prompt). That is the longest contiguous typing
    # a participant produced, and the gaps are human, not ours.
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    presses = [x for x in (_num(r.get("PRESS_TIME")) for r in rows) if x is not None]
    if not presses:
        return []
    t0 = min(presses)
    strokes, dropped = [], 0
    for r in rows:
        try:
            kc, dn, up = int(r["KEYCODE"]), float(r["PRESS_TIME"]), float(r["RELEASE_TIME"])
        except (TypeError, ValueError):
            dropped += 1
            continue
        q = _KC.get(kc)
        if q is None or up < dn or up - dn > 5000:
            dropped += 1
            continue
        strokes.append(KeyStroke((dn - t0) / 1000.0, (up - t0) / 1000.0, q))
    strokes.sort(key=lambda k: k.down)
    pid = rows[0]["PARTICIPANT_ID"]
    return [TypingSession("aalto", pid, strokes,
                          {"dropped": dropped, "sections": len({r["TEST_SECTION_ID"] for r in rows})})]


def aalto_sessions(limit: int | None = None) -> list[TypingSession]:
    d = DATA / "aalto"
    files = sorted(d.glob("*_keystrokes.txt"))[:limit]
    out = []
    for f in files:
        out.extend(s for s in parse_aalto(f) if len(s.strokes) >= 50)
    return out


def cmu_sessions() -> list[TypingSession]:
    """CMU: hold (H.k) and keydown-keydown (DD.a.b) times; rebuild absolute times.

    Each row is one repetition of ".tie5Roanl" + Return. Repetitions of one sessionIndex
    are concatenated with the gap left as unknown -> a fixed 1.0 s, stated, not measured.
    """
    p = DATA / "cmu.csv"
    rows = list(csv.DictReader(p.open()))
    keys = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]
    qmap = {"period": "dot", "five": "5", "Shift.r": "shift", "Return": "ret"}
    by: dict[tuple, list] = {}
    for r in rows:
        by.setdefault((r["subject"], r["sessionIndex"]), []).append(r)
    out = []
    for (subj, sess), reps in by.items():
        t, strokes = 0.0, []
        for r in sorted(reps, key=lambda r: int(r["rep"])):
            for i, k in enumerate(keys):
                hold = float(r[f"H.{k}"])
                strokes.append(KeyStroke(t, t + hold, qmap.get(k, k.lower())))
                if i + 1 < len(keys):
                    t += float(r[f"DD.{k}.{keys[i + 1]}"])
            t += 1.0
        out.append(TypingSession("cmu", subj, strokes, {"session": sess}))
    return out


@dataclass
class MouseSession:
    source: str
    user: str
    t: list[float]
    x: list[int]
    y: list[int]
    buttons: list[str]


def balabit_sessions(limit: int | None = None) -> list[MouseSession]:
    """Balabit: record timestamp, client timestamp, button, state, x, y (CSV per session)."""
    import zipfile
    if not (DATA / "balabit.zip").exists():
        print("balabit.zip missing: pointer devices will stay idle", file=sys.stderr)
        return []
    z = zipfile.ZipFile(DATA / "balabit.zip")
    names = sorted(n for n in z.namelist() if "/training_files/" in n and "/session_" in n)
    out = []
    for n in names[:limit]:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
        t, x, y, b = [], [], [], []
        for r in rows:
            try:
                ct = float(r["client timestamp"])
                xi, yi = int(float(r["x"])), int(float(r["y"]))
            except (KeyError, ValueError):
                continue
            if xi > 10000 or yi > 10000:   # documented sentinel for "outside the window"
                continue
            t.append(ct)
            x.append(xi)
            y.append(yi)
            b.append(f'{r.get("button", "")}:{r.get("state", "")}')
        if len(t) > 100:
            t0 = t[0]
            out.append(MouseSession("balabit", n.split("/")[-2], [v - t0 for v in t], x, y, b))
    return out


# ---------------------------------------------------------------------- stats ---
def typing_stats(sessions: list[TypingSession]) -> dict:
    """Timing statistics that decide the HID report stream, pooled over sessions."""
    import numpy as np
    iki, hold, overlap, gaps1 = [], [], 0, 0
    n = 0
    for s in sessions:
        ks = s.strokes
        for a, b in zip(ks, ks[1:]):
            d = b.down - a.down
            if d <= 0:
                continue
            iki.append(d)
            overlap += b.down < a.up         # next key pressed before this one released
            gaps1 += d > 1.0
            n += 1
        hold += [k.up - k.down for k in ks]
    iki, hold = np.array(iki), np.array(hold)
    return {
        "sessions": len(sessions), "keystrokes": int(sum(len(s.strokes) for s in sessions)),
        "iki_median_s": float(np.median(iki)), "iki_p10_s": float(np.percentile(iki, 10)),
        "iki_p90_s": float(np.percentile(iki, 90)),
        "hold_median_s": float(np.median(hold)), "hold_p90_s": float(np.percentile(hold, 90)),
        "rollover_frac": overlap / max(n, 1), "gaps_over_1s_frac": gaps1 / max(n, 1),
        "keys_per_s_active": float(1.0 / np.median(iki)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fetch", "stats"])
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch_simple(CMU_URL, DATA / "cmu.csv")
        print("cmu: ok", flush=True)
        fetch_aalto(DATA / "aalto")
        print("aalto: ok", flush=True)
        # GitHub builds the zip on demand and can sit silent before the first byte,
        # which trips a stall timeout; retry rather than fail the whole fetch.
        for attempt in range(5):
            try:
                fetch_simple(BALABIT_URL, DATA / "balabit.zip")
                break
            except subprocess.CalledProcessError:
                (DATA / "balabit.zip.part").unlink(missing_ok=True)
                print(f"balabit: attempt {attempt + 1} stalled, retrying", flush=True)
        print("balabit: ok" if (DATA / "balabit.zip").exists() else "balabit: FAILED", flush=True)
        return 0
    al = aalto_sessions()
    cm = cmu_sessions()
    out = {"aalto": typing_stats(al), "cmu": typing_stats(cm)}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
