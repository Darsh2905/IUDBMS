"""
Summarise a live monitor database (stdlib only, so it runs on the VM unchanged).

    python3 eval/live_summary.py DB [--out out/live_x.json] [--label NAME]

What the monitor actually stored for real devices: per device, how many windows it
scored, how many carried traffic, the identity features it computed (the scanner
defect showed up as ``late_iface_flag == 1.0`` in every window of every device), the
class group, the Stage-2 score distribution, and the alerts and trust state.
"""

import argparse
import json
import sqlite3
import statistics
import sys


def summarise(db: str, label: str = "") -> dict:
    cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {"label": label, "db": db, "devices": {}}
    for did, trust, grp in cx.execute("SELECT device_id, trust_state, class_group FROM devices ORDER BY device_id"):
        rows = cx.execute("SELECT stage2, features FROM windows WHERE device_id=? ORDER BY id", (did,)).fetchall()
        feats = [json.loads(f) for _, f in rows]
        s2 = [s for s, _ in rows]
        busy = [s for (s, _), f in zip(rows, feats) if f["n_events"] > 0]
        alerts = cx.execute("SELECT severity, ts FROM alerts WHERE device_id=? ORDER BY ts", (did,)).fetchall()
        out["devices"][did] = {
            "class_group": grp, "trust_state": trust, "windows": len(rows), "busy_windows": len(busy),
            "late_iface_flag_mean": statistics.mean(f["late_iface_flag"] for f in feats) if feats else None,
            "stage2_median": statistics.median(s2) if s2 else None,
            "stage2_max": max(s2) if s2 else None,
            "stage2_busy_median": statistics.median(busy) if busy else None,
            "alerts": len(alerts), "first_alert_ts": alerts[0][1] if alerts else None,
            "severities": sorted({a[0] for a in alerts}),
        }
    out["totals"] = {"windows": sum(d["windows"] for d in out["devices"].values()),
                     "alerts": sum(d["alerts"] for d in out["devices"].values()),
                     "devices": len(out["devices"]),
                     "devices_alerted": sum(1 for d in out["devices"].values() if d["alerts"])}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    r = summarise(a.db, a.label)
    for did, d in r["devices"].items():
        print(f"  {did:6s} group={d['class_group']:9s} windows={d['windows']:4d} busy={d['busy_windows']:4d} "
              f"late_flag={d['late_iface_flag_mean']:.2f} s2 median={d['stage2_median']:.3f} max={d['stage2_max']:.3f} "
              f"alerts={d['alerts']} state={d['trust_state']}")
    print(f"  totals: {r['totals']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
