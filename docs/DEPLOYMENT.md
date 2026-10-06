# Deploying IUDBMS

IUDBMS runs as two systemd services on the Linux host it protects:

| Service | Runs as | Does |
|---|---|---|
| `iudbms-monitor` | root, group `iudbms` | Reads usbmon, scores every device once a second, persists windows, alerts, actions and the raw URB trace |
| `iudbms-api` | `iudbms` (unprivileged) | Serves the operator dashboard and API from the same database, on loopback |

**Enforcement is dry-run.** Escalations (THROTTLE, DEAUTHORIZE) are generated,
logged and stored, but nothing is written to `/sys/bus/usb/devices/*/authorized`.
That gate is `policy.engine.ENFORCEMENT_REVIEWED`, which stays `False` until the
safety review is signed off.

**The models were trained on simulated traffic.** On a real host the scores are
out of distribution. What a deployment demonstrates today is that capture,
scoring, persistence and the operator path run end to end on real kernel output,
not that its verdicts are calibrated for real devices.

## Requirements

- Linux with `CONFIG_USB_MON` (every mainstream distribution kernel) and debugfs at
  `/sys/kernel/debug`. Tested target: **Ubuntu Server 24.04 LTS (ARM64 or x86_64)**.
- Python ≥ 3.11 with `venv` (`sudo apt install python3-venv`).
- Trained models: `deploy.json`, `stage1.pkl`, `stage2.pt` (in `out/models/`, or
  regenerate with `make eval`).
- About 2 GB of disk for the virtualenv; PyTorch is most of it.

## Install

```bash
git clone <repo> iudbms && cd iudbms
sudo ./deploy/install.sh --enable          # add --cpu-torch on x86_64
sudo ./deploy/verify.sh
```

`install.sh` creates the `iudbms` account, a virtualenv in `/opt/iudbms`, the
state directory `/var/lib/iudbms` (database, trace, models), the configuration
`/etc/iudbms/iudbms.env`, loads `usbmon` now and at boot, and installs both units.
Without `--enable` it installs but does not start anything. Re-running upgrades in
place and never overwrites an edited configuration.

`verify.sh` checks the kernel side, the install, both services, the API, that no
arbitrary origin is granted CORS, and that windows and URBs are being stored. On
an idle bus it warns rather than fails: the monitor only scores devices that have
sent traffic, so move the mouse or type into the VM and run it again.

## Operate

| | |
|---|---|
| Logs | `journalctl -u iudbms-monitor -f` — escalations, and a heartbeat every 60 windows |
| Dashboard | `ssh -L 8000:127.0.0.1:8000 you@host`, then open http://localhost:8000 |
| Status | `curl -s 127.0.0.1:8000/health` |
| Configuration | `/etc/iudbms/iudbms.env`, then `sudo systemctl restart iudbms-monitor iudbms-api` |
| Kill switch | set `IUDBMS_KILL_SWITCH=1` in the env file and restart: suppresses every escalation |
| Stop | `sudo systemctl stop iudbms-monitor` — finishes the current window and commits |

The API has state-changing endpoints and no authentication. It binds to
`127.0.0.1` and grants no cross-origin access; reach it through an SSH tunnel
rather than binding it to a routable address.

## The raw URB trace

Every URB is stored, so a revised feature set can be recomputed over traffic that
was never re-collected (plan §5), and every escalation keeps the traffic that
caused it. `iudbms-monitor --replay-db /var/lib/iudbms/iudbms.db --run-id N` replays
a stored run through the same monitor loop.

| `IUDBMS_TRACE` | Payload bytes stored | Every feature recomputable | Use |
|---|---|---|---|
| `pseudonymous` (default) | keyed permutation, key never stored | yes, bit-exact | normal operation |
| `none` | nothing | all but the three payload-equality features | maximum privacy |
| `full` | **raw bytes — every keystroke in clear** | yes | consented corpus collection on owned devices |
| `off` | no trace at all | — | features and scores only |

Pseudonymous is not anonymous. The mapping is consistent within a session, which
is what keeps the features exact, so a long typing session is a substitution cipher
over keystrokes and open to frequency analysis. It moves reading keystrokes from
inspection to cryptanalysis; it does not make them unreadable.

Retention is capped by `IUDBMS_TRACE_MAX_EVENTS` (default 1,000,000 URBs),
evicting the oldest first. The URBs from the ten windows the model saw before each
escalation are pinned and never evicted. A session that has lost rows to eviction
is reported as incomplete rather than silently recomputed from a partial history.

## Uninstall

```bash
sudo ./deploy/uninstall.sh            # keeps /var/lib/iudbms and /etc/iudbms
sudo ./deploy/uninstall.sh --purge    # deletes data, configuration and the account
```

## What has and has not been verified

Verified on the development host (macOS), by tests and by an install from a built
wheel into a clean virtualenv outside the repository:

- the package installs with both entry points and the dashboard as package data;
- `iudbms-monitor` replays a capture and a stored run, persisting windows, alerts,
  actions and a pseudonymised trace that recomputes every feature bit-exactly;
- `iudbms-api` serves the dashboard, the deployed policy and the trace endpoints,
  and grants no cross-origin access;
- the shell scripts parse.

**Not yet verified — needs the Linux VM:** the systemd units themselves (sandboxing
directives, the setgid state directory shared between root and `iudbms`, SIGTERM
handling under systemd), `install.sh` and `verify.sh` end to end, and live capture
from `/sys/kernel/debug/usb/usbmon`. Run `sudo ./deploy/verify.sh` on the VM and
treat this section as outdated until it passes there.
