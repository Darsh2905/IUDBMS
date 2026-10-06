"""
Raw URB trace storage — so the feature set can be revised without re-collecting.

Plan §5 asks for exactly that: "store raw traces, not just features, so the
feature set can be revised without re-collecting." Until this module the store
held only feature vectors and scores, so every change to the extractor meant
recapturing, and a flagged device left no record of what it had actually sent.

The acceptance test is therefore re-featurisation: a stored session, read back
and run through a fresh extractor, must reproduce **bit for bit** the vectors
the live monitor scored. ``refeaturise`` does that and the tests assert it.

Privacy: payloads are pseudonymised by default
----------------------------------------------
A USB trace is a keylogger. Every keyboard report is a keystroke, and a mass
storage stream is file contents. Storing raw payloads would turn a deployed
behavioural monitor into exactly the thing it exists to detect, and would
contradict a property the architecture claims: that keystroke content is not
retained.

The extractor touches payload bytes in only three ways — ``e.length`` for every
size feature, ``if e.data`` to skip uncaptured payloads, and ``e.data`` as a
dictionary key or bigram for the novelty features. All three survive any
**injective, length-preserving** mapping. So ``pseudonymous`` mode (the
default) stores each payload under a keyed permutation of byte strings — a
small Feistel network, plus a keyed table for 1-byte payloads — which:

* preserves equality and order, so **every current feature re-featurises
  exactly**;
* preserves length, so a stored event still satisfies ``URBEvent.validate`` and
  a pseudonymised trace replays through the real monitor like a real one;
* uses a key drawn per recording run and **never written anywhere**, so the
  database alone cannot be reversed by trying candidate keystrokes.

What it does **not** give is anonymity. Within one session the mapping is
consistent — that is what keeps the novelty features exact — so a long typing
session is a substitution cipher over keystrokes, and frequency analysis on
enough text could recover it. The honest claim is "raises reading keystrokes
from inspection to cryptanalysis", not "keystrokes are protected".

``none`` drops payloads entirely (the three payload-equality features then
cannot be reproduced). ``full`` keeps raw bytes and exists for corpus
collection on consenting, owned devices — the plan §5 case — and must be asked
for explicitly.

Retention
---------
Bounded by ``max_events``. Eviction removes the oldest unpinned rows, and rows
inside an alert's evidence span are pinned and never evicted, so an alert keeps
the raw traffic the model actually saw. Completeness is tracked rather than
assumed: ``refeaturise`` reports whether any of a session's rows were evicted,
because ``payload_seq_novelty`` depends on the whole session history and a
truncated session cannot reproduce it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import numpy as np

from ..contract import (
    CONTRACT_VERSION,
    CaptureStats,
    DeviceDescriptor,
    Endpoint,
    Interface,
    URBEvent,
)
from ..features.extractor import WindowFeatureExtractor

__all__ = [
    "PAYLOAD_MODES", "PayloadPseudonymizer", "TraceRecorder", "StoreSource",
    "RefeaturisedSession", "descriptor_to_json", "descriptor_from_json",
    "trace_runs", "trace_sessions", "session_events", "refeaturise",
]

PAYLOAD_MODES = ("pseudonymous", "none", "full")

#: Feistel rounds. Four is the Luby-Rackoff minimum for a strong pseudorandom
#: permutation; correctness (bijectivity) holds for any count.
_ROUNDS = 4


# --------------------------------------------------------------------------
# payload pseudonymisation
# --------------------------------------------------------------------------

class PayloadPseudonymizer:
    """A keyed, length-preserving permutation of byte strings.

    Injective at every length, which is the property the payload features need:
    two payloads collide after mapping if and only if they were equal before.
    A truncated HMAC would be simpler and is not used, for two reasons. It must
    either break ``len(data) <= length`` (16 bytes for an 8-byte HID report) or
    be truncated to the payload's length, at which point a 4-byte mouse report
    has a 32-bit tag and collides within a long session -- silently changing
    ``payload_novelty`` for the windows involved.
    """

    def __init__(self, key: bytes):
        if len(key) < 16:
            raise ValueError("key must be at least 16 bytes")
        self._key = key
        # 1-byte payloads: a two-half Feistel degenerates to XOR with a
        # constant, which leaks the XOR structure of the inputs. A keyed table
        # is a uniformly random permutation of all 256 values instead.
        order = sorted(range(256), key=lambda b: self._mac(b"t", bytes([b])))
        self._table = bytes(order)

    def _mac(self, tag: bytes, msg: bytes) -> bytes:
        return hmac.new(self._key, tag + msg, hashlib.sha256).digest()

    def _prf(self, rnd: int, half: bytes, n: int) -> bytes:
        out = b""
        ctr = 0
        while len(out) < n:
            out += self._mac(bytes([0x46, rnd, ctr]), half)
            ctr += 1
        return out[:n]

    def __call__(self, data: bytes) -> bytes:
        n = len(data)
        if n == 0:
            return b""
        if n == 1:
            return self._table[data[0]:data[0] + 1]
        left, right = data[: n // 2], data[n // 2:]
        for r in range(_ROUNDS):
            # (L, R) -> (R xor F(L), L) is invertible for any F and any split,
            # so the composition is a bijection on n-byte strings. Halves swap
            # length each round; an even round count restores them.
            f = self._prf(r, left, len(right))
            left, right = bytes(a ^ b for a, b in zip(right, f)), left
        return left + right

    def invert(self, data: bytes) -> bytes:
        """Inverse permutation. Exists for the bijectivity tests only -- the
        recorder never stores the key, so nothing in deployment can call it."""
        n = len(data)
        if n == 0:
            return b""
        if n == 1:
            return bytes([self._table.index(data[0])])
        # forward output is (L_k, R_k); undo rounds k-1..0
        a = n // 2 if _ROUNDS % 2 == 0 else n - n // 2
        left, right = data[:a], data[a:]
        for r in reversed(range(_ROUNDS)):
            prev_left = right
            f = self._prf(r, prev_left, len(left))
            prev_right = bytes(x ^ y for x, y in zip(left, f))
            left, right = prev_left, prev_right
        return left + right


# --------------------------------------------------------------------------
# descriptor (de)serialisation
# --------------------------------------------------------------------------

def descriptor_to_json(d: DeviceDescriptor) -> str:
    return json.dumps(asdict(d), sort_keys=True)


def descriptor_from_json(s: str) -> DeviceDescriptor:
    raw = json.loads(s)
    ifaces = tuple(
        Interface(
            bInterfaceClass=i["bInterfaceClass"],
            bInterfaceSubClass=i.get("bInterfaceSubClass", 0),
            bInterfaceProtocol=i.get("bInterfaceProtocol", 0),
            endpoints=tuple(Endpoint(**e) for e in i.get("endpoints", ())),
            first_seen_ts=i.get("first_seen_ts", 0.0),
        )
        for i in raw.get("interfaces", ())
    )
    raw["interfaces"] = ifaces
    return DeviceDescriptor(**raw)


# --------------------------------------------------------------------------
# the recorder (write side)
# --------------------------------------------------------------------------

class TraceRecorder:
    """Writes every URB a capture produces, in the order the monitor drained it.

    Owned by one monitoring run. The monitor calls :meth:`record` with each
    drained batch, :meth:`on_attach` / :meth:`on_detach` as it starts and stops
    scoring a device, and :meth:`close` at the end. Persistence calls
    :meth:`pin` when an alert fires.

    Rows are written in drain order and read back in id order, which is what
    makes re-featurisation exact: the extractor computes inter-arrival times
    with ``np.diff`` on the events *as given*, so within-window order matters.
    """

    def __init__(
        self,
        store,
        *,
        payload_mode: str = "pseudonymous",
        max_events: int = 1_000_000,
        evict_every: int = 20_000,
        source: str = "",
        window: float = 1.0,
        key: bytes | None = None,
    ):
        if payload_mode not in PAYLOAD_MODES:
            raise ValueError(f"payload_mode must be one of {PAYLOAD_MODES}, "
                             f"got {payload_mode!r}")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.cx: sqlite3.Connection = store.cx
        self.payload_mode = payload_mode
        self.max_events = max_events
        self.evict_every = evict_every
        self.window = window
        self._pseudo = (PayloadPseudonymizer(key or os.urandom(32))
                        if payload_mode == "pseudonymous" else None)
        cur = self.cx.execute(
            "INSERT INTO trace_runs (started, source, payload_mode, window_s,"
            " contract_version) VALUES (?,?,?,?,?)",
            (time.time(), source, payload_mode, window, CONTRACT_VERSION))
        self.run_id = int(cur.lastrowid)
        self._open: dict[str, int] = {}          # device_id -> open session_id
        self._since_evict = 0
        self.n_recorded = 0
        self.n_late = 0
        self.n_evicted = 0
        self.n_pinned = 0

    # -- payloads -----------------------------------------------------------

    def _payload(self, data: bytes) -> bytes | None:
        if not data or self.payload_mode == "none":
            return None
        if self.payload_mode == "full":
            return bytes(data)
        return self._pseudo(data)

    # -- write paths --------------------------------------------------------

    def record(self, events: Sequence[URBEvent], *, late: bool = False) -> None:
        """Append a batch. ``late`` marks events the monitor did not score
        because their window had already closed."""
        if not events:
            return
        run = self.run_id
        self.cx.executemany(
            "INSERT INTO urb_events (run_id,device_id,ts,evt,ttype,dir,ep,status,"
            "length,interval,data,data_truncated,setup,late)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run, e.device_key, float(e.ts), e.evt, e.ttype, e.direction, e.ep,
              e.status, e.length, e.interval, self._payload(e.data),
              int(e.data_truncated), e.setup or None, int(late))
             for e in events])
        self.n_recorded += len(events)
        if late:
            self.n_late += len(events)
        self._since_evict += len(events)
        if self._since_evict >= self.evict_every:
            self._since_evict = 0
            self.evict()

    def on_attach(self, device_id: str, t: float, descriptor: DeviceDescriptor) -> None:
        if device_id in self._open:
            return
        cur = self.cx.execute(
            "INSERT INTO trace_sessions (run_id,device_id,t_attach,descriptor)"
            " VALUES (?,?,?,?)",
            (self.run_id, device_id, float(t), descriptor_to_json(descriptor)))
        self._open[device_id] = int(cur.lastrowid)

    def on_detach(self, device_id: str, t: float) -> None:
        sid = self._open.pop(device_id, None)
        if sid is not None:
            self.cx.execute("UPDATE trace_sessions SET t_detach=? WHERE session_id=?",
                            (float(t), sid))

    def close(self, t: float | None = None) -> None:
        """End the run. Open sessions are closed at ``t`` -- the end of the last
        window scored -- so their extent is recorded rather than inferred."""
        for device_id in list(self._open):
            if t is not None:
                self.on_detach(device_id, t)
            else:
                self._open.pop(device_id)
        self.evict()
        self.cx.commit()

    def pin(self, device_id: str, t0: float, t1: float) -> int:
        """Exempt ``[t0, t1)`` of a device's trace from eviction."""
        cur = self.cx.execute(
            "UPDATE urb_events SET pinned=1 WHERE run_id=? AND device_id=?"
            " AND ts>=? AND ts<? AND pinned=0",
            (self.run_id, device_id, float(t0), float(t1)))
        self.n_pinned += cur.rowcount
        return cur.rowcount

    def evict(self) -> int:
        """Drop the oldest unpinned rows beyond ``max_events``.

        Records, per device, the latest timestamp evicted, so a reader can tell
        whether a given session is still complete.
        """
        total = self.cx.execute("SELECT COUNT(*) FROM urb_events").fetchone()[0]
        excess = total - self.max_events
        if excess <= 0:
            return 0
        victims = self.cx.execute(
            "SELECT id, run_id, device_id, ts FROM urb_events WHERE pinned=0"
            " ORDER BY id LIMIT ?", (excess,)).fetchall()
        if not victims:
            return 0
        through: dict[tuple[int, str], tuple[float, int]] = {}
        for _id, run, dev, ts in victims:
            hi, n = through.get((run, dev), (float("-inf"), 0))
            through[(run, dev)] = (max(hi, ts), n + 1)
        self.cx.executemany(
            "INSERT INTO trace_evictions (run_id,device_id,evicted_through,n_evicted)"
            " VALUES (?,?,?,?) ON CONFLICT(run_id,device_id) DO UPDATE SET"
            " evicted_through=MAX(evicted_through, excluded.evicted_through),"
            " n_evicted=n_evicted+excluded.n_evicted",
            [(run, dev, hi, n) for (run, dev), (hi, n) in through.items()])
        last_id = victims[-1][0]
        cur = self.cx.execute(
            "DELETE FROM urb_events WHERE pinned=0 AND id<=?", (last_id,))
        self.n_evicted += cur.rowcount
        return cur.rowcount

    @property
    def stats(self) -> dict:
        return {"run_id": self.run_id, "payload_mode": self.payload_mode,
                "recorded": self.n_recorded, "late": self.n_late,
                "evicted": self.n_evicted, "pinned": self.n_pinned,
                "open_sessions": len(self._open)}


# --------------------------------------------------------------------------
# read side
# --------------------------------------------------------------------------

def _row_to_event(r) -> URBEvent:
    return URBEvent(
        ts=r["ts"], evt=r["evt"], ttype=r["ttype"], direction=r["dir"],
        bus=int(r["device_id"].split("-")[0]), dev=int(r["device_id"].split("-")[1]),
        ep=r["ep"], status=r["status"], length=r["length"],
        data=bytes(r["data"]) if r["data"] is not None else b"",
        interval=r["interval"], data_truncated=bool(r["data_truncated"]),
        setup=bytes(r["setup"]) if r["setup"] is not None else b"")


def trace_runs(store) -> list[dict]:
    return [dict(r) for r in store.cx.execute(
        "SELECT r.*, (SELECT COUNT(*) FROM urb_events u WHERE u.run_id=r.run_id)"
        " AS n_events, (SELECT COUNT(*) FROM trace_sessions s"
        " WHERE s.run_id=r.run_id) AS n_sessions FROM trace_runs r ORDER BY run_id")]


def trace_sessions(store, run_id: int | None = None,
                   device_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM trace_sessions WHERE 1=1", []
    if run_id is not None:
        q += " AND run_id=?"; args.append(run_id)
    if device_id is not None:
        q += " AND device_id=?"; args.append(device_id)
    return [dict(r) for r in store.cx.execute(q + " ORDER BY session_id", args)]


def _span(store, sess: dict) -> tuple[float, float]:
    """The slice of the device's trace that belongs to this session.

    Starts where the previous session of the same device ended, because the
    monitor holds a not-yet-enumerated device's events and folds them into its
    first scored window; ends at detach (or open-ended)."""
    prev = store.cx.execute(
        "SELECT t_detach FROM trace_sessions WHERE run_id=? AND device_id=?"
        " AND session_id<? ORDER BY session_id DESC LIMIT 1",
        (sess["run_id"], sess["device_id"], sess["session_id"])).fetchone()
    lo = prev["t_detach"] if prev and prev["t_detach"] is not None else float("-inf")
    hi = sess["t_detach"] if sess["t_detach"] is not None else float("inf")
    return lo, hi


def session_events(store, session_id: int, *, include_late: bool = False
                   ) -> tuple[DeviceDescriptor, list[URBEvent], bool]:
    """A session's descriptor, its events in drain order, and whether it is
    complete (no row in its span was ever evicted)."""
    sess = store.cx.execute("SELECT * FROM trace_sessions WHERE session_id=?",
                            (session_id,)).fetchone()
    if sess is None:
        raise KeyError(f"no trace session {session_id}")
    sess = dict(sess)
    lo, hi = _span(store, sess)
    q = ("SELECT * FROM urb_events WHERE run_id=? AND device_id=? AND ts>=? AND ts<?"
         + ("" if include_late else " AND late=0") + " ORDER BY id")
    rows = store.cx.execute(q, (sess["run_id"], sess["device_id"],
                                lo if lo != float("-inf") else -1e18,
                                hi if hi != float("inf") else 1e18)).fetchall()
    ev = store.cx.execute(
        "SELECT evicted_through FROM trace_evictions WHERE run_id=? AND device_id=?",
        (sess["run_id"], sess["device_id"])).fetchone()
    complete = ev is None or ev["evicted_through"] < lo
    return descriptor_from_json(sess["descriptor"]), [_row_to_event(r) for r in rows], complete


@dataclass
class RefeaturisedSession:
    session_id: int
    device_id: str
    t: np.ndarray          # window start times
    X: np.ndarray          # [W, N_FEATURES]
    complete: bool         # False if eviction removed part of this session
    n_events: int


def refeaturise(store, session_id: int, *, include_late: bool = False,
                extractor: WindowFeatureExtractor | None = None) -> RefeaturisedSession:
    """Re-run a stored session through a (possibly revised) extractor.

    Windowing mirrors ``LiveMonitor`` exactly: windows are ``floor(ts / w)``
    from the attach window to the last window scored; events recorded before
    attach (a device the monitor saw before sysfs enumerated it) are folded into
    the first window *after* that window's own events, which is where the
    monitor put them; late events are excluded unless asked for, since the
    monitor never scored them.

    Pass a different ``extractor`` to compute a revised feature set over traffic
    that was never re-collected -- which is the whole purpose of this module.
    """
    row = store.cx.execute("SELECT s.*, r.window_s FROM trace_sessions s"
                           " JOIN trace_runs r USING (run_id)"
                           " WHERE session_id=?", (session_id,)).fetchone()
    if row is None:
        raise KeyError(f"no trace session {session_id}")
    sess = dict(row)
    window = sess["window_s"]
    desc, events, complete = session_events(store, session_id, include_late=include_late)
    ex = extractor or WindowFeatureExtractor()

    w_first = int(sess["t_attach"] // window)
    deferred = [e for e in events if int(e.ts // window) < w_first]
    on_time = [e for e in events if int(e.ts // window) >= w_first]
    if sess["t_detach"] is not None:
        w_last = int(round(sess["t_detach"] / window)) - 1
    else:
        w_last = max((int(e.ts // window) for e in on_time), default=w_first)

    buckets: dict[int, list[URBEvent]] = {}
    for e in on_time:
        buckets.setdefault(int(e.ts // window), []).append(e)
    # Held events are older than the attach window's own, so they go first --
    # the order LiveMonitor scores them in.
    buckets[w_first] = deferred + buckets.get(w_first, [])

    t, X = [], []
    for w in range(w_first, w_last + 1):
        t0 = w * window
        X.append(ex.extract(buckets.get(w, []), t0, desc))
        t.append(t0)
    return RefeaturisedSession(
        session_id=session_id, device_id=sess["device_id"],
        t=np.asarray(t), X=np.vstack(X) if X else np.empty((0, 0)),
        complete=complete, n_events=len(events))


# --------------------------------------------------------------------------
# a stored run as an EventSource
# --------------------------------------------------------------------------

class StoreSource:
    """Replay a stored run through anything that consumes an ``EventSource``.

    A fifth capture backend beside usbmon, eBPF, file replay and the simulator,
    for the same reason the simulator is one: a stored trace has to go through
    the identical path a live capture does, not a privileged side door.
    """

    def __init__(self, store, run_id: int, *, include_late: bool = True):
        self._store = store
        self.run_id = run_id
        self.include_late = include_late
        self.stats = CaptureStats()
        self._desc: dict[str, DeviceDescriptor] = {}
        for s in trace_sessions(store, run_id=run_id):
            self._desc[s["device_id"]] = descriptor_from_json(s["descriptor"])

    def descriptor(self, device_key: str) -> DeviceDescriptor:
        return self._desc[device_key]

    def events(self) -> Iterator[URBEvent]:
        q = ("SELECT * FROM urb_events WHERE run_id=?"
             + ("" if self.include_late else " AND late=0") + " ORDER BY id")
        for r in self._store.cx.execute(q, (self.run_id,)):
            e = _row_to_event(r)
            self.stats.lines_seen += 1
            self.stats.events_emitted += 1
            if self.stats.first_ts is None:
                self.stats.first_ts = e.ts
            self.stats.last_ts = e.ts
            yield e
