"""
Score testbed captures through the DEPLOYED monitor loop and summarise per device.

    python3 -m testbed.score [RUN_DIR ...] [--models out/models --models out/models_corpus_a]

Each capture is replayed through ``iudbms.live.cli`` -- the same LiveMonitor,
per-device attach/detach, StreamingDetector, gate and persistence the systemd
service runs -- with an explicit model directory, into its own database. The
summary is computed from that database, so a number here is a property of the
named weights and the deployed code path, not of an offline re-implementation.

Writes out/testbed/summary.json (committed) and per-run databases under
out/testbed/scores/ (ignored).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iudbms.capture.usbmon import UsbmonSource  # noqa: E402
from iudbms.live import cli  # noqa: E402

RAW = ROOT / "out" / "testbed" / "raw"
SCORES = ROOT / "out" / "testbed" / "scores"
SUMMARY = ROOT / "out" / "testbed" / "summary.json"


def first_keystroke(capture: Path) -> dict[str, float]:
    """Per device: time of the first HID report with a key down (the attack onset)."""
    out: dict[str, float] = {}
    for e in UsbmonSource(path=capture).events():
        if (e.evt == "C" and e.ttype == "intr" and len(e.data) == 8 and any(e.data[2:])
                and e.device_key not in out):
            out[e.device_key] = e.ts
    return out


def replay(run: Path, models: Path, name: str) -> Path:
    db = SCORES / run.name / f"{name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["--fast", "--replay", str(run / "usbmon.txt"), "--descriptors",
                       str(run / "descriptors.json"), "--db", str(db), "--models", str(models),
                       "--log", "actions", "--trace", "off", "--color", "never"])
    (db.parent / f"{name}.log").write_text(buf.getvalue())
    if rc not in (0, None):
        raise RuntimeError(f"monitor replay of {run.name} with {models} exited {rc}")
    return db


def summarise(run: Path, db: Path, threshold: float) -> dict:
    labels = json.loads((run / "labels.json").read_text())
    onset = first_keystroke(run / "usbmon.txt")
    cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    devs = {}
    for dev, in cx.execute("SELECT DISTINCT device_id FROM windows"):
        rows = cx.execute("SELECT t_start, stage2, features FROM windows WHERE device_id=? "
                          "ORDER BY t_start", (dev,)).fetchall()
        t = np.array([r[0] for r in rows])
        s2 = np.array([r[1] if r[1] is not None else np.nan for r in rows], float)
        busy = np.array([json.loads(r[2]).get("n_events", 0) > 0 for r in rows])
        alerts = [a for a, in cx.execute("SELECT ts FROM alerts WHERE device_id=? ORDER BY ts",
                                         (dev,))]
        lab = labels.get(dev, {"role": "unlabelled", "label": 0})
        d = {"role": lab["role"], "label": lab["label"], "windows": len(rows),
             "busy_windows": int(busy.sum()), "hours": len(rows) / 3600.0,
             "stage2_median_busy": float(np.nanmedian(s2[busy])) if busy.any() else None,
             "stage2_max": float(np.nanmax(s2)) if np.isfinite(s2).any() else None,
             "windows_over": int(np.nansum(s2 >= threshold)), "alerts": len(alerts),
             "first_alert": alerts[0] if alerts else None}
        if lab["label"] == 1:
            o = onset.get(dev)
            d["onset"] = o
            post = [a for a in alerts if o is not None and a >= o]
            d["detected"] = bool(post)
            d["alert_before_onset"] = bool([a for a in alerts if o is not None and a < o])
            d["latency_s"] = (post[0] - o) if post else None
        devs[dev] = d
    return devs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--models", action="append", type=Path,
                    help="model directory (repeatable); default: deployed + Corpus A")
    a = ap.parse_args(argv)
    runs = a.runs or sorted(p for p in RAW.iterdir()
                            if (p / "usbmon.txt").exists() and (p / "labels.json").exists())
    sets = a.models or [ROOT / "out" / "models", ROOT / "out" / "models_corpus_a"]
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {"runs": {}}
    for run in runs:
        env = json.loads((run / "env.json").read_text())
        wl = json.loads((run / "workload.json").read_text())
        src = UsbmonSource(path=run / "usbmon.txt")
        n_ev = sum(1 for _ in src.events())
        entry = {"distro": wl["distro"], "kind": wl["kind"], "seed": wl["seed"],
                 "minutes": wl["minutes"], "kernel": env["kernel"], "os": env["os"],
                 "drivers": sorted({v["driver"] or "(none)" for v in env["interfaces"].values()}),
                 "usbmon_lines": src.stats.lines_seen, "unparsed": src.stats.lines_unparsed,
                 "events": n_ev, "keys_typed": wl["keys_typed"],
                 "pointer_events": wl["mouse_events"] + wl["tablet_events"],
                 "thread_errors": wl["thread_errors"], "models": {}}
        for m in sets:
            thr = json.loads((m / "deploy.json").read_text())["threshold"]
            db = replay(run, m, m.name)
            devs = summarise(run, db, thr)
            ben = [d for d in devs.values() if d["label"] == 0]
            atk = [d for d in devs.values() if d["label"] == 1]
            entry["models"][m.name] = {
                "threshold": thr, "devices": devs,
                "benign_devices": len(ben), "benign_hours": sum(d["hours"] for d in ben),
                "benign_alarmed": sum(d["alerts"] > 0 for d in ben),
                "benign_alerts": sum(d["alerts"] for d in ben),
                "attacks": len(atk), "attacks_detected": sum(d.get("detected", False) for d in atk)}
            e = entry["models"][m.name]
            print(f"{run.name:34s} {m.name:16s} benign: {e['benign_alarmed']}/{e['benign_devices']} "
                  f"devices alarmed over {e['benign_hours']:.2f} device-h;  attacks "
                  f"{e['attacks_detected']}/{e['attacks']}", flush=True)
            for k, d in sorted(devs.items()):
                if d["alerts"] or d["label"]:
                    print(f"    {k:6s} {d['role']:30s} alerts {d['alerts']:2d}  "
                          + (f"latency {d['latency_s']:.1f}s" if d.get("latency_s") is not None
                             else ("MISSED" if d["label"] else "")))
        summary["runs"][run.name] = entry
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
