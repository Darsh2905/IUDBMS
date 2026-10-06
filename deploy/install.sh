#!/usr/bin/env bash
# Install IUDBMS on a Linux host as two systemd services.
#
#   sudo ./deploy/install.sh [--models DIR] [--enable] [--cpu-torch]
#
#   --models DIR   trained models to deploy (default: out/models in this repo).
#                  Must contain deploy.json, stage1.pkl and stage2.pt.
#   --enable       enable and start both services now (default: install only)
#   --cpu-torch    install PyTorch's CPU-only build first. Recommended on
#                  x86_64, where the default wheel pulls in gigabytes of CUDA.
#                  Unnecessary on ARM64, where the default wheel is CPU-only.
#
# Layout:
#   /opt/iudbms/venv            the package, in its own virtualenv
#   /etc/iudbms/iudbms.env      configuration for both units (never overwritten)
#   /var/lib/iudbms/            database, raw trace, deployed models
#   /etc/systemd/system/iudbms-{monitor,api}.service
#
# Re-running is safe: it upgrades the package and models in place and keeps
# your configuration. Undo with ./deploy/uninstall.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$REPO/out/models"
ENABLE=0
CPU_TORCH=0
PREFIX=/opt/iudbms
STATE=/var/lib/iudbms
ETC=/etc/iudbms

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)    MODELS="$2"; shift 2 ;;
    --enable)    ENABLE=1; shift ;;
    --cpu-torch) CPU_TORCH=1; shift ;;
    -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "install: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

die()  { echo "install: $*" >&2; exit 1; }
step() { echo "==> $*"; }

[[ "$(uname -s)" == Linux ]] || die "this installs systemd services; run it on the Linux host"
[[ ${EUID} -eq 0 ]]          || die "needs root: sudo $0"
command -v systemctl >/dev/null || die "systemd not found"
PY="$(command -v python3)"   || die "python3 not found"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
                             || die "Python >= 3.11 required, found $("$PY" -V)"
"$PY" -c 'import ensurepip' 2>/dev/null \
                             || die "python3-venv missing: sudo apt install python3-venv"
for f in deploy.json stage1.pkl stage2.pt; do
  [[ -f "$MODELS/$f" ]] || die "$MODELS/$f missing -- train with 'make eval' or pass --models DIR"
done

step "service account 'iudbms'"
getent group iudbms >/dev/null || groupadd --system iudbms
id iudbms >/dev/null 2>&1 || useradd --system --gid iudbms --no-create-home \
    --home-dir /nonexistent --shell /usr/sbin/nologin iudbms

step "directories"
# The state directory is setgid iudbms, so files the monitor creates as root
# belong to group iudbms; with the units' UMask=0007 the API -- which runs as
# iudbms -- can read and write the database and its WAL files.
install -d -m 0755 "$PREFIX"
install -d -m 2770 -o root -g iudbms "$STATE"
install -d -m 2750 -o root -g iudbms "$STATE/models"
install -d -m 0750 -o root -g iudbms "$ETC"

step "virtualenv and package -> $PREFIX/venv"
[[ -x "$PREFIX/venv/bin/python" ]] || "$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
if [[ $CPU_TORCH -eq 1 ]]; then
  step "PyTorch, CPU-only build"
  "$PREFIX/venv/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PREFIX/venv/bin/pip" install --quiet --upgrade "$REPO"
"$PREFIX/venv/bin/iudbms-monitor" --help >/dev/null || die "installed monitor does not start"

step "models -> $STATE/models"
install -m 0640 -o root -g iudbms \
  "$MODELS/deploy.json" "$MODELS/stage1.pkl" "$MODELS/stage2.pt" "$STATE/models/"

step "configuration -> $ETC/iudbms.env"
if [[ -f "$ETC/iudbms.env" ]]; then
  install -m 0640 -o root -g iudbms "$REPO/deploy/iudbms.env" "$ETC/iudbms.env.new"
  echo "    kept your existing iudbms.env; the shipped default is iudbms.env.new"
else
  install -m 0640 -o root -g iudbms "$REPO/deploy/iudbms.env" "$ETC/iudbms.env"
fi

step "usbmon, now and at every boot"
install -m 0644 "$REPO/deploy/modules-load.d/iudbms.conf" /etc/modules-load.d/iudbms.conf
modprobe usbmon || echo "    warning: modprobe usbmon failed -- kernel built without CONFIG_USB_MON?"

step "systemd units"
install -m 0644 "$REPO/deploy/systemd/iudbms-monitor.service" \
                "$REPO/deploy/systemd/iudbms-api.service" /etc/systemd/system/
install -m 0644 "$REPO/docs/DEPLOYMENT.md" "$PREFIX/DEPLOYMENT.md"
systemctl daemon-reload

if [[ $ENABLE -eq 1 ]]; then
  step "enabling and starting"
  systemctl enable --now iudbms-monitor.service iudbms-api.service
fi

cat <<MSG

Installed. Enforcement is dry-run: actions are logged and stored, and nothing is
written to /sys/bus/usb/.../authorized (policy.engine.ENFORCEMENT_REVIEWED).

  start      sudo systemctl enable --now iudbms-monitor iudbms-api
  verify     sudo $REPO/deploy/verify.sh
  logs       journalctl -u iudbms-monitor -f
  dashboard  ssh -L 8000:127.0.0.1:8000 <you>@<this-host>, then http://localhost:8000
  config     $ETC/iudbms.env
MSG
