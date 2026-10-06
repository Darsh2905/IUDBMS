#!/usr/bin/env bash
# Remove the IUDBMS services and package.
#
#   sudo ./deploy/uninstall.sh            # keeps data and configuration
#   sudo ./deploy/uninstall.sh --purge    # also deletes them, and the account
#
# Without --purge the database, raw traces and models in /var/lib/iudbms and the
# configuration in /etc/iudbms are kept, so a reinstall picks up where it left
# off. --purge deletes the stored traces irrecoverably.
set -euo pipefail
PURGE=0
[[ "${1:-}" == "--purge" ]] && PURGE=1
[[ ${EUID} -eq 0 ]] || { echo "uninstall: needs root: sudo $0" >&2; exit 1; }

systemctl disable --now iudbms-monitor.service iudbms-api.service 2>/dev/null || true
rm -f /etc/systemd/system/iudbms-monitor.service /etc/systemd/system/iudbms-api.service
systemctl daemon-reload
rm -f /etc/modules-load.d/iudbms.conf
rm -rf /opt/iudbms

if [[ $PURGE -eq 1 ]]; then
  rm -rf /var/lib/iudbms /etc/iudbms
  userdel iudbms 2>/dev/null || true
  groupdel iudbms 2>/dev/null || true
  echo "Removed IUDBMS, its data, its configuration and its account."
else
  echo "Removed IUDBMS. Kept /var/lib/iudbms (data) and /etc/iudbms (config);"
  echo "re-run with --purge to delete them."
fi
