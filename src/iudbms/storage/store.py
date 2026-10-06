"""L4 — persistence (SQLite, WAL). Schema per plan §4.6."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..contract import FEATURE_NAMES, DeviceDescriptor

__all__ = ["Store", "SCHEMA"]

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS devices (
    device_id      TEXT PRIMARY KEY,
    bus            INTEGER, devnum INTEGER,
    vid            INTEGER, pid    INTEGER,
    class_group    TEXT,
    n_interfaces   INTEGER,
    serial_present INTEGER,
    first_seen     REAL,
    trust_state    TEXT DEFAULT 'observe'
);

CREATE TABLE IF NOT EXISTS windows (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT REFERENCES devices(device_id),
    t_start   REAL,
    stage1    REAL,
    stage2    REAL,
    label     INTEGER,
    features  TEXT
);
CREATE INDEX IF NOT EXISTS ix_windows_dev ON windows(device_id, t_start);

CREATE TABLE IF NOT EXISTS baselines (
    device_id TEXT, feature TEXT, ewma_mean REAL, ewma_var REAL, n INTEGER,
    PRIMARY KEY (device_id, feature)
);

CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT REFERENCES devices(device_id),
    ts        REAL,
    stage1    REAL,
    stage2    REAL,
    severity  TEXT,
    evidence  TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_dev ON alerts(device_id, ts);

CREATE TABLE IF NOT EXISTS actions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id  INTEGER REFERENCES alerts(id),
    device_id TEXT,
    ts        REAL,
    action    TEXT,
    rule      TEXT
);

-- Raw URB traces (storage.traces). Plan §5: "store raw traces, not just
-- features, so the feature set can be revised without re-collecting."
-- IF NOT EXISTS throughout, so databases written before these tables existed
-- gain them on next open.
CREATE TABLE IF NOT EXISTS trace_runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started          REAL,          -- wall clock, epoch seconds
    source           TEXT,
    payload_mode     TEXT,          -- pseudonymous | none | full
    window_s         REAL,
    contract_version TEXT
);

CREATE TABLE IF NOT EXISTS trace_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER REFERENCES trace_runs(run_id),
    device_id  TEXT,
    t_attach   REAL,                -- start of the first window scored
    t_detach   REAL,                -- end of the last window scored; NULL = open
    descriptor TEXT                 -- JSON, as attached by the detector
);
CREATE INDEX IF NOT EXISTS ix_trace_sessions ON trace_sessions(run_id, device_id);

CREATE TABLE IF NOT EXISTS urb_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,   -- drain order
    run_id         INTEGER,
    device_id      TEXT,
    ts             REAL,
    evt            TEXT,
    ttype          TEXT,
    dir            TEXT,
    ep             INTEGER,
    status         INTEGER,
    length         INTEGER,
    interval       INTEGER,
    data           BLOB,            -- per trace_runs.payload_mode
    data_truncated INTEGER,
    setup          BLOB,
    late           INTEGER DEFAULT 0,   -- arrived after its window closed
    pinned         INTEGER DEFAULT 0    -- alert evidence; never evicted
);
CREATE INDEX IF NOT EXISTS ix_urb_dev ON urb_events(run_id, device_id, ts);
CREATE INDEX IF NOT EXISTS ix_urb_evict ON urb_events(pinned, id);

CREATE TABLE IF NOT EXISTS trace_evictions (
    run_id          INTEGER,
    device_id       TEXT,
    evicted_through REAL,           -- latest ts ever evicted for this device
    n_evicted       INTEGER,
    PRIMARY KEY (run_id, device_id)
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cx = sqlite3.connect(str(path))
        self.cx.row_factory = sqlite3.Row
        self.cx.executescript(SCHEMA)

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- writers ----------------

    def add_device(self, device_id: str, d: DeviceDescriptor, group: str,
                   first_seen: float = 0.0) -> None:
        self.cx.execute(
            "INSERT OR REPLACE INTO devices VALUES (?,?,?,?,?,?,?,?,?,?)",
            (device_id, d.bus, d.devnum, d.vid, d.pid, group,
             len(d.interfaces), int(d.serial_present), first_seen, "observe"))

    def add_windows(self, device_id: str, t, s1, s2, y, X,
                    names: tuple[str, ...] = FEATURE_NAMES) -> None:
        self.cx.executemany(
            "INSERT INTO windows (device_id,t_start,stage1,stage2,label,features)"
            " VALUES (?,?,?,?,?,?)",
            # A label of None is stored as NULL: live traffic has no ground
            # truth, and 0 would claim every window was known benign.
            [(device_id, float(a), float(b), float(c), None if d is None else int(d),
              json.dumps({n: round(float(v), 4) for n, v in zip(names, x)}))
             for a, b, c, d, x in zip(t, s1, s2, y, X)])

    def add_baseline(self, device_id: str, baseline) -> None:
        self.cx.executemany(
            "INSERT OR REPLACE INTO baselines VALUES (?,?,?,?,?)",
            [(device_id, n, float(m), float(v), baseline.n)
             for n, m, v in zip(baseline.names, baseline.mean, baseline.var)])

    def add_alert(self, device_id: str, ts: float, s1: float, s2: float,
                  severity: str, evidence) -> int:
        cur = self.cx.execute(
            "INSERT INTO alerts (device_id,ts,stage1,stage2,severity,evidence)"
            " VALUES (?,?,?,?,?,?)",
            (device_id, float(ts), float(s1), float(s2), severity, json.dumps(evidence)))
        return int(cur.lastrowid)

    def add_action(self, alert_id: int, device_id: str, ts: float,
                   action: str, rule: dict) -> None:
        self.cx.execute(
            "INSERT INTO actions (alert_id,device_id,ts,action,rule) VALUES (?,?,?,?,?)",
            (alert_id, device_id, float(ts), action, json.dumps(rule)))

    def set_trust(self, device_id: str, state: str) -> None:
        self.cx.execute("UPDATE devices SET trust_state=? WHERE device_id=?",
                        (state, device_id))

    def commit(self) -> None:
        self.cx.commit()

    def close(self) -> None:
        self.cx.commit()
        self.cx.close()

    # ---------------- readers (back the API) ----------------

    def devices(self) -> list[dict]:
        return [dict(r) for r in self.cx.execute(
            "SELECT * FROM devices ORDER BY device_id")]

    def timeline(self, device_id: str, limit: int = 5000) -> list[dict]:
        return [dict(r) for r in self.cx.execute(
            "SELECT t_start, stage1, stage2, label FROM windows"
            " WHERE device_id=? ORDER BY t_start LIMIT ?", (device_id, limit))]

    def alerts(self, device_id: str | None = None, limit: int = 200) -> list[dict]:
        if device_id:
            rows = self.cx.execute(
                "SELECT * FROM alerts WHERE device_id=? ORDER BY ts DESC LIMIT ?",
                (device_id, limit))
        else:
            rows = self.cx.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"]) if d["evidence"] else []
            out.append(d)
        return out

    def actions(self, device_id: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM actions"
        args: tuple = ()
        if device_id:
            q += " WHERE device_id=?"
            args = (device_id,)
        q += " ORDER BY id DESC LIMIT ?"
        out = []
        for r in self.cx.execute(q, args + (limit,)):
            d = dict(r)
            d["rule"] = json.loads(d["rule"]) if d["rule"] else {}
            out.append(d)
        return out

    def trace(self, device_id: str, limit: int = 500) -> list[dict]:
        """The most recent raw URBs for a device, newest first, for forensics.

        ``data`` is whatever the recording run stored -- a pseudonym, nothing,
        or (only if explicitly configured) raw bytes -- so the run's payload
        mode travels with every row.
        """
        out = []
        for r in self.cx.execute(
                "SELECT u.ts, u.evt, u.ttype, u.dir, u.ep, u.status, u.length,"
                " u.data, u.data_truncated, u.late, u.pinned, u.run_id,"
                " r.payload_mode FROM urb_events u JOIN trace_runs r USING (run_id)"
                " WHERE u.device_id=? ORDER BY u.id DESC LIMIT ?", (device_id, limit)):
            d = dict(r)
            d["data"] = bytes(d["data"]).hex() if d["data"] is not None else None
            out.append(d)
        return out

    def counts(self) -> dict[str, int]:
        return {t: int(self.cx.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                for t in ("devices", "windows", "baselines", "alerts", "actions",
                          "trace_runs", "trace_sessions", "urb_events")}
