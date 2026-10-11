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

## What the monitor decides, and what it declines to decide

| Mechanism | Where it is set | What it does |
|---|---|---|
| Alarm threshold | `deploy.json` `threshold`, `threshold_rule` | Chosen so that benign devices raise at most 0.01 alarms per device-hour on 80 h of long simulated sessions (`eval/analysis_calibration.py`). The earlier per-window rule raised ~0.2 per keyboard-hour. |
| 4-of-6 gate | `deploy.json` `k`, `n` | A device alarms when 4 of its last 6 one-second windows are over the threshold. |
| Model support | `deploy.json` `support` | Interface classes of the benign training devices. A device that declares any other class -- a USB network adapter, a serial converter, a smart-card reader -- is shown as **unmodelled**: scored and stored, never escalated. |
| Keystroke capability | read from each HID interface's report descriptor | A HID device that may send keystrokes (Keyboard or Consumer usage page, or descriptor unreadable) is **always** judged. A pure pointer (mouse, tablet, pen) cannot inject keystrokes and is shown as unmodelled. |
| Capability rule | `deploy.json` `capability_rule` (if present) | A keyboard sending at least 30 reports/s in 4 of 6 seconds alarms on its own gate: faster than any of 1,972 real typists sustained. Absent from a model set that has not adopted it. |

"Unmodelled" is a statement about the model, not a clean bill of health: decide
those device types by policy (allow or deny by class), as you would without IUDBMS.

## Requirements

- Linux with `CONFIG_USB_MON` (every mainstream distribution kernel) and debugfs at
  `/sys/kernel/debug`. Tested target: **Ubuntu Server 24.04 LTS (ARM64 or x86_64)**.
- Python ≥ 3.11 with `venv` (`sudo apt install python3-venv`).
- Trained models: `deploy.json`, `stage1.pkl`, `stage2.pt` in `out/models/`, produced by
  `make train-deployed` (trained on the realistic corpus). `make eval` writes the
  Corpus-A set to `out/models_corpus_a/` and does not touch the deployment.
- About 2 GB of disk for the virtualenv. PyTorch is installed from its **CPU-only**
  index (~160 MB); the PyPI wheel now pulls the whole CUDA stack on every Linux
  architecture, which is several GB for a model that only runs on CPU. If you
  install the package by hand, do the same:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

## Install

```bash
git clone <repo> iudbms && cd iudbms
sudo ./deploy/install.sh --enable
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
wheel into a clean virtualenv outside the repository: both entry points, the
dashboard as package data, persistence, the pseudonymised trace, every API endpoint,
and no cross-origin grant.

**Verified on a real Linux host** (Ubuntu 24.04.5 aarch64, kernel 6.8.0-146, UTM/QEMU
guest), 06 Oct 2026:

- `deploy/install.sh --enable` completes; both units start and stay active;
- `deploy/verify.sh` passes all 18 checks, including a live check that windows are
  being persisted;
- the service reads `/sys/kernel/debug/usb/usbmon`, parses real kernel output with
  0.0000% loss, attaches the described devices and scores them every second;
- the same real capture replayed on the Mac and on the VM scores identically to six
  decimals.

**Not yet verified:** that the services come back after a reboot; SIGTERM handling
under systemd's stop timeout; any real keystroke traffic (the VM had no display
device, so the virtual USB keyboard was never fed); and, above all, behaviour on
real *physical* devices or a real attack. The devices seen so far are QEMU's
virtual ones, mostly idle.

**A green `verify.sh` is necessary, not sufficient.** The first service on a real
kernel passed every structural check and scored nothing (CLAUDE.md finding 28);
only reading the monitor's heartbeat (`devices=0 late=174`) showed it. After
installing, confirm `journalctl -u iudbms-monitor` shows a heartbeat with
`devices>=1` and `late=0`.
