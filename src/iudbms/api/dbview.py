"""
Database explorer: a read-only dashboard over the SQLite store.

    python3 -m iudbms.api.dbview [--db out/iudbms.db] [--port 8002]
    make db-dashboard                            # then open http://localhost:8002

``out/iudbms.db`` is what ``make eval`` writes: the 1,000 SIMULATED sessions of the
development corpus (Corpus A), every 1 s window scored by the evaluated model, and the
alerts and dry-run actions it raised. Nothing here comes from real USB hardware. The
database is opened read-only, so the explorer cannot change what a result was computed from.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ..contract import FEATURE_NAMES

REPO = Path(__file__).resolve().parents[3]
PAGE = Path(str(files("iudbms.api").joinpath("static", "dbview.html")))
TABLES = ("devices", "windows", "alerts", "actions", "baselines",
          "trace_runs", "trace_sessions", "urb_events", "trace_evictions")

app = FastAPI(title="IUDBMS database explorer")


def db_path() -> Path:
    return Path(os.environ.get("IUDBMS_DB", REPO / "out" / "iudbms.db"))


def cx() -> sqlite3.Connection:
    p = db_path()
    if not p.exists():
        raise HTTPException(503, f"no database at {p}. Create one with `make eval`.")
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


@lru_cache(maxsize=4)
def _devices(path: str, mtime: float) -> list[dict]:
    """One row per device with its aggregates. Cached per file version: it scans every window."""
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute("""
        SELECT d.device_id id, d.bus, d.devnum, d.vid, d.pid, d.class_group cls, d.n_interfaces,
               d.serial_present serial, d.trust_state state,
               COALESCE(w.n, 0) windows, w.s2max, w.lab,
               COALESCE(a.n, 0) alerts, a.first_alert
        FROM devices d
        LEFT JOIN (SELECT device_id, COUNT(*) n, MAX(stage2) s2max, MAX(label) lab
                   FROM windows GROUP BY device_id) w ON w.device_id = d.device_id
        LEFT JOIN (SELECT device_id, COUNT(*) n, MIN(ts) first_alert
                   FROM alerts GROUP BY device_id) a ON a.device_id = d.device_id""").fetchall()
    return [dict(r) for r in rows]


def devices() -> list[dict]:
    p = db_path()
    cx().close()                                   # 503 with a message if it is missing
    return _devices(str(p), p.stat().st_mtime)


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@app.get("/db/overview")
def overview() -> dict:
    c = cx()
    devs = devices()
    tables = {}
    for t in TABLES:
        try:
            tables[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            tables[t] = None
    attack = [d for d in devs if d["lab"] == 1]
    benign = [d for d in devs if d["lab"] == 0]
    by_cls: dict[str, dict] = {}
    for d in devs:
        s = by_cls.setdefault(d["cls"], {"cls": d["cls"], "benign": 0, "attack": 0})
        s["attack" if d["lab"] == 1 else "benign"] += 1
    return {
        "threshold": _threshold(),
        "path": str(db_path()), "size_mb": round(db_path().stat().st_size / 1e6, 1), "tables": tables,
        "sessions": len(devs), "attack": len(attack), "benign": len(benign),
        "attack_alerted": sum(1 for d in attack if d["alerts"]),
        "benign_alerted": sum(1 for d in benign if d["alerts"]),
        "by_class": sorted(by_cls.values(), key=lambda s: -(s["benign"] + s["attack"])),
        "states": [dict(r) for r in c.execute(
            "SELECT trust_state state, COUNT(*) n FROM devices GROUP BY 1 ORDER BY n DESC")],
        "severity": [dict(r) for r in c.execute(
            "SELECT severity, COUNT(*) n FROM alerts GROUP BY 1 ORDER BY n DESC")],
        "actions": [dict(r) for r in c.execute(
            "SELECT action, COUNT(*) n FROM actions GROUP BY 1 ORDER BY n DESC")],
        "windows": {"total": tables["windows"],
                    "attack_windows": c.execute("SELECT COUNT(*) FROM windows WHERE label=1").fetchone()[0]},
        "stage2_hist": _hist(c),
    }


def _threshold() -> float | None:
    """The operating point of the model that scored this database (``make eval`` writes it)."""
    for d in ("models_corpus_a", "models"):
        try:
            return json.loads((REPO / "out" / d / "deploy.json").read_text())["threshold"]
        except (OSError, KeyError, ValueError):
            continue
    return None


def _hist(c: sqlite3.Connection) -> list[dict]:
    """Stage 2 score distribution by window label, in 'nines' bins (0.9, 0.99, ...)."""
    out = {0: [0] * 6, 1: [0] * 6}
    for lab, s2 in c.execute("SELECT label, stage2 FROM windows WHERE stage2 IS NOT NULL"):
        if lab not in out:
            continue
        b = 0 if s2 < 0.9 else 1 if s2 < 0.99 else 2 if s2 < 0.999 else 3 if s2 < 0.9999 else 4
        out[lab][b] += 1
    labels = ["< 0.9", "0.9 to 0.99", "0.99 to 0.999", "0.999 to 0.9999", ">= 0.9999"]
    return [{"bin": labels[i], "benign": out[0][i], "attack": out[1][i]} for i in range(5)]


@app.get("/db/devices")
def list_devices(q: str = "", label: str = "all", cls: str = "all", alerted: str = "all",
                 sort: str = "id", desc: bool = False,
                 page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=200)) -> dict:
    rows = devices()
    if q:
        ql = q.lower()
        rows = [d for d in rows if ql in d["id"].lower() or ql in f"{d['vid']:04x}:{d['pid']:04x}"]
    if label != "all":
        rows = [d for d in rows if d["lab"] == (1 if label == "attack" else 0)]
    if cls != "all":
        rows = [d for d in rows if d["cls"] == cls]
    if alerted != "all":
        rows = [d for d in rows if bool(d["alerts"]) == (alerted == "yes")]
    keys = {"id": "id", "class": "cls", "windows": "windows", "alerts": "alerts",
            "peak": "s2max", "state": "state"}
    k = keys.get(sort, "id")
    rows = sorted(rows, key=lambda d: (d[k] is None, d[k]), reverse=desc)
    n = len(rows)
    out = [{**d, "vidpid": f"{d['vid']:04x}:{d['pid']:04x}"} for d in rows[(page - 1) * size: page * size]]
    return {"total": n, "page": page, "size": size, "rows": out}


@app.get("/db/device/{device_id}")
def device(device_id: str) -> dict:
    c = cx()
    d = next((x for x in devices() if x["id"] == device_id), None)
    if d is None:
        raise HTTPException(404, f"no device {device_id}")
    w = c.execute("SELECT t_start t, stage1 s1, stage2 s2, label lab FROM windows "
                  "WHERE device_id=? ORDER BY t_start", (device_id,)).fetchall()
    al = c.execute("SELECT id, ts, stage1, stage2, severity, evidence FROM alerts "
                   "WHERE device_id=? ORDER BY ts", (device_id,)).fetchall()
    ac = c.execute("SELECT id, alert_id, ts, action, rule FROM actions "
                   "WHERE device_id=? ORDER BY ts", (device_id,)).fetchall()
    return {"device": {**d, "vidpid": f"{d['vid']:04x}:{d['pid']:04x}"},
            "windows": [dict(r) for r in w],
            "alerts": [{**dict(r), "evidence": json.loads(r["evidence"] or "[]")} for r in al],
            "actions": [{**dict(r), "rule": json.loads(r["rule"] or "{}")} for r in ac]}


@app.get("/db/device/{device_id}/window")
def window(device_id: str, t: float) -> dict:
    r = cx().execute("SELECT t_start, stage1, stage2, label, features FROM windows "
                     "WHERE device_id=? AND t_start=?", (device_id, t)).fetchone()
    if r is None:
        raise HTTPException(404, "no such window")
    f = json.loads(r["features"])
    return {"t": r["t_start"], "stage1": r["stage1"], "stage2": r["stage2"], "label": r["label"],
            "features": [{"name": n, "value": f.get(n)} for n in FEATURE_NAMES if n in f]}


@app.get("/db/table/{name}")
def table(name: str, page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=200)) -> dict:
    if name not in TABLES:
        raise HTTPException(404, f"unknown table {name!r}")
    c = cx()
    total = c.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    cur = c.execute(f"SELECT * FROM {name} LIMIT ? OFFSET ?", (size, (page - 1) * size))
    cols = [x[0] for x in cur.description]
    rows = []
    for r in cur.fetchall():
        row = []
        for v in r:
            row.append(f"<{len(v)} bytes>" if isinstance(v, bytes) else v)
        rows.append(row)
    return {"table": name, "total": total, "page": page, "size": size, "columns": cols, "rows": rows}


def main(argv=None) -> int:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    if a.db:
        os.environ["IUDBMS_DB"] = str(a.db)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
