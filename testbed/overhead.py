"""
What does monitoring cost? usbmon capture vs eBPF probes, under USB bulk load.

    python3 -m testbed.overhead [--distro debian-13] [--reps 10]

Reviewer point 12: the eBPF prototype's overhead was never measured, and neither
was usbmon's. This boots a guest whose USB storage stick is driven at full speed by
direct I/O (bulk transfers are the highest URB rate a USB device produces, so this is
the worst case for per-URB cost) and measures, per condition:

  none         nothing attached
  usbmon       the deployed capture path: `cat /sys/kernel/debug/usb/usbmon/0u`
  ebpf_count   bpftrace on __usb_hcd_giveback_urb, aggregation only (cheapest probe)
  ebpf_event   the same probe emitting one record per URB (what a capture backend needs)
  both         usbmon + ebpf_event

Each repetition writes then reads STICK_MB with O_DIRECT and records throughput and
guest CPU (user+system+softirq jiffies from /proc/stat). Conditions are run in a
fresh random order every repetition so drift (host load, caches) cannot bias one
condition. With kernel.bpf_stats_enabled=1, bpftool also reports the exact run time
of the eBPF program per invocation -- a direct per-URB cost, not an inference.

**Read with care.** The device side is emulated (QEMU), so absolute throughput is
QEMU's, not a real stick's; what transfers is the RELATIVE cost of each monitor on
the same kernel and the eBPF program's ns per completion.

Writes out/testbed/overhead.json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .vm import ROOT, Guest

OUT = ROOT / "out" / "testbed"
STICK_MB = 64
CONDITIONS = ("none", "usbmon", "ebpf_count", "ebpf_event", "both")
G = "/var/tmp/tb"

PROBE_COUNT = r"""kfunc:__usb_hcd_giveback_urb { @c = count(); }"""
PROBE_EVENT = r"""kfunc:__usb_hcd_giveback_urb {
  printf("C %d %d %d %d\n", args->urb->dev->devnum, args->urb->ep->desc.bEndpointAddress,
         args->urb->status, args->urb->actual_length); }"""


def _stat(g: Guest) -> int:
    f = g.ssh("head -1 /proc/stat").stdout.split()
    # user nice system idle iowait irq softirq ...
    return int(f[1]) + int(f[2]) + int(f[3]) + int(f[6]) + int(f[7])


def run(distro: str, reps: int) -> dict:
    rd = ROOT / "out" / "testbed" / "raw" / f"{distro}-overhead"
    g = Guest(distro, ("storage", "kbd"), rd, mem_mb=2048, cpus=2)
    res: dict = {"distro": distro, "stick_mb": STICK_MB, "reps": reps, "runs": []}
    try:
        g.start()
        g.ssh(f"mkdir -p {G} && sudo chmod 777 {G}")
        inst = g.ssh("(command -v bpftrace && command -v bpftool) || "
                     "(DEBIAN_FRONTEND=noninteractive timeout 900 apt-get -qq update && "
                     "DEBIAN_FRONTEND=noninteractive timeout 900 apt-get -qq install -y "
                     "bpftrace bpftool) || (timeout 900 dnf -q -y install bpftrace bpftool)",
                     sudo=True, check=False, timeout=2000)
        have = g.ssh("command -v bpftrace; command -v bpftool", check=False).stdout
        res["bpftrace"] = "bpftrace" in have
        res["bpftool"] = "bpftool" in have
        res["kernel"] = g.ssh("uname -r").stdout.strip()
        if not res["bpftrace"]:
            raise RuntimeError(f"bpftrace unavailable: {inst.stderr[-400:]}")
        (rd / "count.bt").write_text(PROBE_COUNT)
        (rd / "event.bt").write_text(PROBE_EVENT)
        g.scp_to(rd / "count.bt", f"{G}/count.bt")
        g.scp_to(rd / "event.bt", f"{G}/event.bt")
        dev = g.ssh("for b in /sys/block/sd*; do readlink -f $b | grep -q usb && "
                    "echo /dev/$(basename $b) && break; done").stdout.strip()
        res["device"] = dev
        g.ssh("modprobe usbmon; mountpoint -q /sys/kernel/debug || mount -t debugfs none "
              "/sys/kernel/debug; sysctl -q kernel.bpf_stats_enabled=1", sudo=True)
        start = {
            "usbmon": f"setsid nohup cat /sys/kernel/debug/usb/usbmon/0u > {G}/u.txt 2>/dev/null </dev/null &",
            "ebpf_count": f"setsid nohup bpftrace {G}/count.bt > {G}/c.txt 2>&1 </dev/null &",
            "ebpf_event": f"setsid nohup bpftrace {G}/event.bt > {G}/e.txt 2>&1 </dev/null &",
        }
        start["both"] = start["usbmon"] + " " + start["ebpf_event"]
        rng = random.Random(7)
        for rep in range(reps):
            order = list(CONDITIONS)
            rng.shuffle(order)
            for cond in order:
                g.ssh(f"rm -f {G}/u.txt {G}/c.txt {G}/e.txt", sudo=True, check=False)
                if cond != "none":
                    g.ssh(start[cond], sudo=True)
                    time.sleep(4.0 if "ebpf" in cond or cond == "both" else 1.0)
                c0 = _stat(g)
                w = g.ssh(f"dd if=/dev/zero of={dev} bs=1M count={STICK_MB} oflag=direct 2>&1 | tail -1",
                          sudo=True).stdout
                r = g.ssh(f"dd if={dev} of=/dev/null bs=1M count={STICK_MB} iflag=direct 2>&1 | tail -1",
                          sudo=True).stdout
                c1 = _stat(g)
                ns = None
                if "ebpf" in cond or cond == "both":
                    out = g.ssh("bpftool prog show 2>/dev/null | grep -A0 kfunc", sudo=True, check=False).stdout
                    for ln in out.splitlines():
                        if "run_time_ns" in ln:
                            t = ln.split()
                            rt = int(t[t.index("run_time_ns") + 1])
                            rc = int(t[t.index("run_cnt") + 1])
                            ns = rt / rc if rc else None
                g.ssh("pkill -f 'usbmon/0u'; pkill -INT bpftrace; sleep 1", sudo=True, check=False)
                lines = g.ssh(f"wc -l < {G}/u.txt 2>/dev/null; wc -l < {G}/e.txt 2>/dev/null",
                              sudo=True, check=False).stdout.split()
                res["runs"].append({"rep": rep, "cond": cond, "write": w.strip(), "read": r.strip(),
                                    "write_MBps": _mbps(w), "read_MBps": _mbps(r),
                                    "cpu_jiffies": c1 - c0, "ebpf_ns_per_call": ns,
                                    "trace_lines": [int(x) for x in lines if x.isdigit()]})
            print(f"  rep {rep + 1}/{reps} done", flush=True)
    finally:
        g.stop()
    summ = {}
    for cond in CONDITIONS:
        rr = [x for x in res["runs"] if x["cond"] == cond]
        summ[cond] = {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
                      for k in ("write_MBps", "read_MBps", "cpu_jiffies")
                      for v in [[x[k] for x in rr if x[k] is not None]] if v}
        ns = [x["ebpf_ns_per_call"] for x in rr if x["ebpf_ns_per_call"]]
        if ns:
            summ[cond]["ebpf_ns_per_call"] = (statistics.mean(ns), statistics.stdev(ns) if len(ns) > 1 else 0.0)
    base = summ["none"]
    for cond in CONDITIONS:
        for k in ("write_MBps", "read_MBps"):
            summ[cond][k + "_vs_none"] = summ[cond][k][0] / base[k][0] - 1.0
    res["summary"] = summ
    return res


def _mbps(dd_line: str) -> float | None:
    # "... copied, 1.234 s, 54.3 MB/s"
    try:
        num, unit = dd_line.rsplit(",", 1)[1].split()
        v = float(num)
        return v * {"kB/s": 1e-3, "MB/s": 1.0, "GB/s": 1e3}.get(unit, 1.0)
    except (ValueError, IndexError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distro", default="debian-13")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args(argv)
    res = run(a.distro, a.reps)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overhead.json").write_text(json.dumps(res, indent=1))
    print(f"\n{res['distro']} kernel {res['kernel']}, {STICK_MB} MB direct I/O, {a.reps} reps")
    print(f"  {'condition':12s} {'write MB/s':>14s} {'read MB/s':>14s} {'CPU jiffies':>13s} {'eBPF ns/call':>13s}")
    for c, s in res["summary"].items():
        ns = s.get("ebpf_ns_per_call")
        print(f"  {c:12s} {s['write_MBps'][0]:7.1f} ({s['write_MBps_vs_none']:+5.1%}) "
              f"{s['read_MBps'][0]:7.1f} ({s['read_MBps_vs_none']:+5.1%}) {s['cpu_jiffies'][0]:8.0f}+/-"
              f"{s['cpu_jiffies'][1]:<4.0f} " + (f"{ns[0]:9.0f}" if ns else "        -"))
    print(f"wrote {OUT / 'overhead.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
