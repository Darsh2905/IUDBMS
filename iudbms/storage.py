"""IUDBMS - persistence layer (SQLite)."""

import json
import sqlite3

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS devices (
    device_id     TEXT PRIMARY KEY,
    bus           INTEGER, devnum INTEGER,
    vid           INTEGER, pid    INTEGER,
    class_group   TEXT,
    n_interfaces  INTEGER,
    serial_present INTEGER,
    first_seen    REAL,
    trust_state   TEXT DEFAULT 'learning'
);

CREATE TABLE IF NOT EXISTS windows (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT REFERENCES devices(device_id),
    t_start     REAL,
    stage1      REAL,
    stage2      REAL,
    label       INTEGER,
    features    TEXT
);
CREATE INDEX IF NOT EXISTS ix_windows_dev ON windows(device_id, t_start);

CREATE TABLE IF NOT EXISTS baselines (
    device_id  TEXT, feature TEXT, ewma_mean REAL, ewma_var REAL, n INTEGER,
    PRIMARY KEY (device_id, feature)
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id  TEXT REFERENCES devices(device_id),
    ts         REAL,
    stage1     REAL,
    stage2     REAL,
    severity   TEXT,
    evidence   TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id   INTEGER REFERENCES alerts(id),
    device_id  TEXT,
    ts         REAL,
    action     TEXT,
    rule       TEXT
);
"""


class Store:
    def __init__(self, path):
        self.cx = sqlite3.connect(path)
        self.cx.executescript(SCHEMA)

    def add_device(self, device_id, d, group, t0=0.0):
        self.cx.execute(
            "INSERT OR REPLACE INTO devices VALUES (?,?,?,?,?,?,?,?,?,?)",
            (device_id, d["bus"], d["devnum"], d["vid"], d["pid"], group,
             len(d["interfaces"]), d["serial_present"], t0, "learning"))

    def add_windows(self, device_id, T, s1, s2, y, X, names):
        self.cx.executemany(
            "INSERT INTO windows (device_id,t_start,stage1,stage2,label,features)"
            " VALUES (?,?,?,?,?,?)",
            [(device_id, float(t), float(a), float(b), int(l),
              json.dumps({n: round(float(v), 4) for n, v in zip(names, x)}))
             for t, a, b, l, x in zip(T, s1, s2, y, X)])

    def add_baseline(self, device_id, names, bl):
        self.cx.executemany(
            "INSERT OR REPLACE INTO baselines VALUES (?,?,?,?,?)",
            [(device_id, n, float(m), float(v), bl.n)
             for n, m, v in zip(names, bl.mean, bl.var)])

    def add_alert(self, device_id, ts, s1, s2, severity, evidence):
        cur = self.cx.execute(
            "INSERT INTO alerts (device_id,ts,stage1,stage2,severity,evidence)"
            " VALUES (?,?,?,?,?,?)",
            (device_id, float(ts), float(s1), float(s2), severity, json.dumps(evidence)))
        return cur.lastrowid

    def add_action(self, alert_id, device_id, ts, action, rule):
        self.cx.execute(
            "INSERT INTO actions (alert_id,device_id,ts,action,rule) VALUES (?,?,?,?,?)",
            (alert_id, device_id, float(ts), action, json.dumps(rule)))

    def set_trust(self, device_id, state):
        self.cx.execute("UPDATE devices SET trust_state=? WHERE device_id=?", (state, device_id))

    def commit(self):
        self.cx.commit()

    def counts(self):
        q = lambda t: self.cx.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return {t: q(t) for t in ("devices", "windows", "alerts", "actions", "baselines")}
