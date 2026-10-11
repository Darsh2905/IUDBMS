"""
Aggregate scored testbed runs into the tables the paper reports.

    python3 -m testbed.report            # reads out/testbed/summary.json

Per distribution and in total: benign device-hours by role, false alarms of the
deployed path (database alerts: calibrated threshold, model support), what judging
every device would have raised, the capability rule, and the original Corpus-A
model; then every attack episode with its detection and latency. Writes
out/testbed/report.json (committed).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "out" / "testbed" / "summary.json"
REPORT = ROOT / "out" / "testbed" / "report.json"


def main() -> int:
    runs = json.loads(SUMMARY.read_text())["runs"]
    by_role: dict = defaultdict(lambda: defaultdict(float))
    distros, attacks = {}, []
    for name, r in sorted(runs.items()):
        if "-s99" in name:                                  # smoke tests are not reported
            continue
        dep = r["models"].get("models")
        orig = r["models"].get("models_corpus_a")
        if dep is None:
            continue
        dd = distros.setdefault(r["distro"], {
            "kernel": r["kernel"], "os": r["os"], "drivers": r["drivers"], "minutes": 0.0,
            "usbmon_lines": 0, "unparsed": 0, "keys_typed": 0, "pointer_events": 0})
        dd["minutes"] += r["minutes"]
        dd["usbmon_lines"] += r["usbmon_lines"]
        dd["unparsed"] += r["unparsed"]
        dd["keys_typed"] += r["keys_typed"] or 0
        dd["pointer_events"] += r["pointer_events"] or 0
        for key, d in dep["devices"].items():
            if d["label"] == 1:
                attacks.append({"run": name, "distro": r["distro"], "device": key,
                                "episode": d["role"], "detected": d.get("detected"),
                                "latency_s": d.get("latency_s"),
                                "alert_before_onset": d.get("alert_before_onset"),
                                "rule_detected": d.get("rule_detected"),
                                "corpus_a_detected": (orig or {}).get("devices", {}).get(key, {})
                                .get("detected")})
                continue
            role = d["role"]
            b = by_role[role]
            b["devices"] += 1
            b["hours"] += d["hours"]
            b["deployed_alarmed"] += d["alerts"] > 0
            b["judging_all_alarmed"] += d["offline_first_alarm"].get("deployed") is not None
            prev = [v for k, v in d["offline_first_alarm"].items() if k.startswith("previous")]
            b["previous_threshold_alarmed"] += any(v is not None for v in prev)
            b["rule_alarmed"] += d["rule_first_alarm"] is not None
            if orig and key in orig["devices"]:
                b["corpus_a_alarmed"] += orig["devices"][key]["alerts"] > 0
    tot = defaultdict(float)
    for b in by_role.values():
        for k, v in b.items():
            tot[k] += v
    out = {"distributions": distros, "benign_by_role": {k: dict(v) for k, v in by_role.items()},
           "benign_total": dict(tot), "attacks": attacks}
    print("DISTRIBUTIONS")
    for k, d in distros.items():
        print(f"  {k:14s} {d['os']:34s} kernel {d['kernel']:28s} {d['minutes']:5.0f} min  "
              f"{d['usbmon_lines']:>9,} lines ({d['unparsed']} unparsed)  {d['keys_typed']:,} keys")
    print("\nBENIGN DEVICES (devices alarmed)       devices  hours  deployed  judge-all  "
          "prev-thr  rule  corpus-A")
    for role, b in sorted(by_role.items()) + [("TOTAL", tot)]:
        print(f"  {role:34s} {int(b['devices']):6d} {b['hours']:6.1f} {int(b['deployed_alarmed']):9d} "
              f"{int(b['judging_all_alarmed']):10d} {int(b['previous_threshold_alarmed']):9d} "
              f"{int(b['rule_alarmed']):5d} {int(b['corpus_a_alarmed']):9d}")
    print("\nATTACK EPISODES (real kernels, emulated keyboards, innocuous text)")
    det = defaultdict(lambda: [0, 0, 0])
    for x in attacks:
        c = det[x["episode"]]
        c[0] += 1
        c[1] += bool(x["detected"])
        c[2] += bool(x["rule_detected"])
    for ep, (n, k, r) in sorted(det.items()):
        lat = sorted(x["latency_s"] for x in attacks if x["episode"] == ep and x["latency_s"] is not None)
        med = lat[len(lat) // 2] if lat else None
        print(f"  {ep:20s} model {k}/{n}  rule {r}/{n}  median latency "
              f"{(f'{med:.1f} s' if med is not None else '-')}")
    REPORT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
