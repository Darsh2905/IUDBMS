#!/usr/bin/env bash
# eBPF go/no-go: does a kfunc-based probe agree with usbmon on how many URBs
# completed? Plan §4.1's literal criterion: "usbmon and eBPF agree within 1%".
# Never measured before this script, because no eBPF probe existed before
# scripts/ebpf_probe.bt.
#
#   sudo ./scripts/ebpf_vs_usbmon.sh [seconds] [outdir]
#
# Runs scripts/ebpf_probe.bt and a usbmon capture over the IDENTICAL wall-clock
# window (both started from the same loop iteration, both stopped by the same
# `timeout`), then parses the usbmon side through the project's own parser
# (UsbmonSource) so the comparison uses the SAME completions-only accounting
# rule L2 already applies (CLAUDE.md, Milestone 3 notes: "completions and
# errors only (C, E), never submissions (S)") -- which is also why the eBPF
# probe counts at __usb_hcd_giveback_urb, not usb_submit_urb.
#
# Writes <outdir>/ebpf_vs_usbmon.json. Needs root (debugfs + bpftrace both do).
set -euo pipefail

SECONDS_TO_RUN="${1:-30}"
OUTDIR="${2:-/tmp/iudbms-ebpf}"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ "$(uname -s)" == "Linux" ]] || { echo "Needs Linux (the VM)." >&2; exit 1; }
[[ "${EUID}" -eq 0 ]] || { echo "Needs root: sudo $0 $*" >&2; exit 1; }
command -v bpftrace >/dev/null || { echo "bpftrace not installed." >&2; exit 1; }

mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
modprobe usbmon 2>/dev/null || true
NODE=/sys/kernel/debug/usb/usbmon/0u
[[ -r "$NODE" ]] || { echo "Cannot read $NODE." >&2; exit 1; }

mkdir -p "$OUTDIR"
EBPF_LOG="$OUTDIR/ebpf-${STAMP}.log"
USBMON_CAP="$OUTDIR/usbmon-${STAMP}.txt"
RESULT="$OUTDIR/ebpf_vs_usbmon.json"

echo "Running both captures for ${SECONDS_TO_RUN}s..."
echo "(generate some USB traffic now if you want more than idle polling: lsusb -v, type, etc.)"

# Started in the same breath so both windows are as close to identical as two
# independently-scheduled `timeout` processes can be; bpftrace's own startup
# (attaching kfunc probes) is the larger source of skew, noted in the output.
T0=$(date +%s.%N)
timeout "${SECONDS_TO_RUN}" bpftrace "$REPO/scripts/ebpf_probe.bt" > "$EBPF_LOG" 2>&1 &
EBPF_PID=$!
timeout "${SECONDS_TO_RUN}" cat "$NODE" > "$USBMON_CAP" 2>/dev/null &
CAT_PID=$!
wait "$EBPF_PID" "$CAT_PID" 2>/dev/null || true
T1=$(date +%s.%N)

EBPF_COMPLETIONS=$(grep -c '^COMPLETION ' "$EBPF_LOG" || true)
USBMON_LINES=$(wc -l < "$USBMON_CAP" | tr -d ' ')

echo "  eBPF log      : $EBPF_LOG  ($EBPF_COMPLETIONS completions)"
echo "  usbmon capture: $USBMON_CAP  ($USBMON_LINES raw lines)"

# Parse the usbmon side through the REAL parser so "completions" means what L2
# means by it, not a naive grep for " C ". Prefer the deployed venv's
# interpreter: the project's dependencies (numpy et al.) live there, not in
# the system python3 this script otherwise runs under as root.
PY_BIN="python3"
[[ -x /opt/iudbms/venv/bin/python3 ]] && PY_BIN=/opt/iudbms/venv/bin/python3

"$PY_BIN" - "$USBMON_CAP" "$EBPF_COMPLETIONS" "$T1" "$T0" "$RESULT" "$SECONDS_TO_RUN" "$REPO" <<'PY'
import json, sys
cap_path, ebpf_n, t1, t0, result_path, secs, repo = sys.argv[1:8]
sys.path.insert(0, repo + "/src")   # the repo root the bash script resolved, not
                                    # __file__ -- meaningless when read from stdin
from iudbms.capture.usbmon import UsbmonSource
from iudbms.features.extractor import COUNTED_KINDS  # the single source of truth

ebpf_n, t1, t0, secs = int(ebpf_n), float(t1), float(t0), float(secs)

src = UsbmonSource(path=cap_path)
events = list(src.events())
st = src.stats
usbmon_completions = sum(1 for e in events if e.evt in COUNTED_KINDS)

diff = abs(usbmon_completions - ebpf_n)
denom = max(usbmon_completions, 1)
agreement_pct = 100.0 * (1 - diff / denom)
within_1pct = (diff / denom) <= 0.01

result = {
    "seconds_requested": secs,
    "wall_elapsed_s": round(t1 - t0, 3),
    "usbmon": {
        "lines_seen": st.lines_seen, "lines_unparsed": st.lines_unparsed,
        "loss_rate": st.loss_rate, "events_total": len(events),
        "completions_and_errors": usbmon_completions,
    },
    "ebpf": {"completions_at_giveback_hook": ebpf_n},
    "diff": diff, "agreement_pct": round(agreement_pct, 3),
    "within_1pct_criterion": within_1pct,
}
json.dump(result, open(result_path, "w"), indent=2)

print(f"\nusbmon completions (C+E, via the real parser): {usbmon_completions}")
print(f"eBPF completions (__usb_hcd_giveback_urb)     : {ebpf_n}")
print(f"difference                                    : {diff}  "
      f"({100 - agreement_pct:.3f}% of usbmon's count)")
print(f"plan §4.1 criterion (agree within 1%)         : "
      f"{'PASS' if within_1pct else 'FAIL'}")
print(f"\nwrote {result_path}")
PY
