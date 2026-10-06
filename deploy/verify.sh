#!/usr/bin/env bash
# Post-install checks for an IUDBMS deployment. Run on the host:
#
#   sudo ./deploy/verify.sh
#
# FAIL is a broken install. warn is something worth knowing that is not
# necessarily wrong -- most often an idle bus, where the monitor has nothing to
# score until a device produces traffic. Exits non-zero if anything FAILs.
set -uo pipefail

ENV=/etc/iudbms/iudbms.env
FAILED=0
ok()   { echo "  ok    $*"; }
fail() { echo "  FAIL  $*"; FAILED=1; }
warn() { echo "  warn  $*"; }

[[ ${EUID} -eq 0 ]] || { echo "verify: run with sudo (usbmon nodes are root-only)" >&2; exit 1; }
[[ -r $ENV ]] || { echo "verify: $ENV missing -- has deploy/install.sh been run?" >&2; exit 1; }
# The file is systemd EnvironmentFile syntax, not shell: sourcing it as root both
# executes whatever it contains and breaks on a value with a space
# (`IUDBMS_MONITOR_ARGS=--bus 0` ran `0` as a command). Read the keys we need.
envval() {   # envval KEY DEFAULT -- last assignment wins, surrounding quotes stripped
  local v
  v=$(grep -E "^[[:space:]]*$1=" "$ENV" | tail -n1 | cut -d= -f2-)
  v="${v%\"}"; v="${v#\"}"
  echo "${v:-$2}"
}
IUDBMS_MODELS=$(envval IUDBMS_MODELS /var/lib/iudbms/models)
IUDBMS_API_HOST=$(envval IUDBMS_API_HOST 127.0.0.1)
IUDBMS_API_PORT=$(envval IUDBMS_API_PORT 8000)
API="http://${IUDBMS_API_HOST}:${IUDBMS_API_PORT}"

api() {   # api PATH [PY-EXPR over the decoded JSON as 'd']
  /usr/bin/python3 - "$API$1" "${2:-d}" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as r:
        d = json.load(r)
except Exception as e:
    print(f"ERROR {e}"); sys.exit(1)
print(eval(sys.argv[2], {}, {"d": d}))
PY
}

echo "kernel"
grep -q '^usbmon ' /proc/modules && ok "usbmon module loaded" || fail "usbmon not loaded (sudo modprobe usbmon)"
[[ -r /sys/kernel/debug/usb/usbmon/0u ]] && ok "usbmon node /sys/kernel/debug/usb/usbmon/0u readable" \
  || fail "usbmon node not readable -- is debugfs mounted at /sys/kernel/debug?"

echo "install"
[[ -x /opt/iudbms/venv/bin/iudbms-monitor ]] && ok "iudbms-monitor installed" || fail "/opt/iudbms/venv missing"
for f in deploy.json stage1.pkl stage2.pt; do
  [[ -f "$IUDBMS_MODELS/$f" ]] && ok "model $f" || fail "missing $IUDBMS_MODELS/$f"
done
stat -c '%a %G' /var/lib/iudbms | grep -q '^2770 iudbms$' && ok "state dir is 2770 root:iudbms" \
  || warn "state dir permissions are $(stat -c '%a %U:%G' /var/lib/iudbms); the API may not reach the DB"

echo "services"
for u in iudbms-monitor iudbms-api; do
  if systemctl is-active --quiet "$u"; then ok "$u active"
  else fail "$u not active -- journalctl -u $u -n 50"; fi
done
if journalctl -u iudbms-monitor --since "-15min" -p err --no-pager -q 2>/dev/null | grep -q .; then
  warn "errors in the monitor journal in the last 15 min -- journalctl -u iudbms-monitor -p err"
else
  ok "no errors in the monitor journal"
fi

echo "api"
h=$(api /health) ; [[ $h != ERROR* ]] && ok "API answers on $API" || fail "API not answering on $API ($h)"
[[ $(api /health 'd["enforcement_reviewed"]') == False ]] \
  && ok "enforcement is dry-run (ENFORCEMENT_REVIEWED False)" || warn "enforcement_reviewed is not False"
[[ $(api /health 'd["database_present"]') == True ]] && ok "database present" \
  || warn "no database yet -- the monitor creates it on start"
[[ $(api /health 'd["policy"].get("gate")') == 4-of-6 ]] && ok "deployed policy loaded (gate 4-of-6)" \
  || warn "API cannot read the deployed policy from \$IUDBMS_MODELS/deploy.json"
cors=$(/usr/bin/python3 - "$API/devices" <<'PY'
import sys, urllib.request
r = urllib.request.Request(sys.argv[1], headers={"Origin": "https://evil.example"})
try:
    with urllib.request.urlopen(r, timeout=5) as resp:
        print(resp.headers.get("access-control-allow-origin") or "none")
except Exception as e:
    print("none" if "503" in str(e) else f"ERROR {e}")
PY
)
[[ $cors == none ]] && ok "no cross-origin grant to an arbitrary site" || fail "API grants CORS to an arbitrary origin ($cors)"

echo "monitor activity"
w1=$(api /stats 'd["windows"]' 2>/dev/null); sleep 6; w2=$(api /stats 'd["windows"]' 2>/dev/null)
if [[ $w1 =~ ^[0-9]+$ && $w2 =~ ^[0-9]+$ && $w2 -gt $w1 ]]; then
  ok "persisting windows ($w1 -> $w2 in 6 s)"
else
  warn "no new windows in 6 s -- an idle bus scores nothing until a device sends traffic (move the mouse / type in the VM, then re-run)"
fi
n=$(api /stats 'd["urb_events"]' 2>/dev/null)
[[ $n =~ ^[0-9]+$ && $n -gt 0 ]] && ok "raw trace recording ($n URBs stored)" || warn "no URBs in the raw trace yet"
mode=$(api /traces/runs 'd[-1]["payload_mode"] if d else "no runs"' 2>/dev/null)
case "$mode" in
  pseudonymous|none) ok "trace payload mode: $mode" ;;
  full) warn "trace payload mode is FULL: keystrokes are stored in clear" ;;
  *)    warn "trace payload mode: $mode" ;;
esac

echo
if [[ $FAILED -eq 0 ]]; then echo "verify: no failures"; else echo "verify: FAILED"; fi
exit $FAILED
