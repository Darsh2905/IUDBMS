# IUDBMS — Intelligent USB Device Behaviour Monitoring System

Watch what a USB device *does*, not only what it *declares*. A BadUSB presents a
valid keyboard descriptor, so identity-based defences (USBGuard allowlists, GoodUSB
profiles) authorise it before any malicious behaviour exists. IUDBMS reads the
kernel's `usbmon` stream, scores every device once a second, and raises an alarm
when a keyboard's behaviour is not a person's.

**What this is:** a working Linux monitor (systemd services, dashboard, raw-trace
store) and a simulation-based study of how far behavioural detection goes, with a
real-kernel testbed. **What it is not:** a validated security product. The models
are trained on simulated traffic; no physical BadUSB has been tested; enforcement
is generated but deliberately not applied (dry-run).

Specification: [`docs/IUDBMS_Implementation_Plan.md`](docs/IUDBMS_Implementation_Plan.md).
Model card (features, units, settings, data, splits): [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).
Working notes and every finding: [`CLAUDE.md`](CLAUDE.md).

## How a device is judged

1. **Capture** — `usbmon` text from debugfs, parsed losslessly (0 unparsed lines across
   every real-kernel capture), cut into 1-second windows per device.
2. **Features** — 30 per window (timing, payload, protocol, identity, host); the deployed
   model uses 28 ([model card](docs/MODEL_CARD.md)).
3. **Stage 1** — class-conditioned Isolation Forests plus a per-device EWMA baseline,
   trained on benign traffic only.
4. **Stage 2** — a one-layer LSTM over the last 10 windows, trained on labelled simulated
   attacks.
5. **Decision** — a device alarms when 4 of its last 6 windows are over a threshold that
   is **calibrated in false alarms per device-hour** (≤ 0.01 on long benign sessions), or
   when a keyboard sustains ≥ 30 reports/s — faster than any of 1,972 real typists.
6. **Competence** — devices of a class the model never saw, and pure pointing devices
   (by their HID report descriptor), are reported *unmodelled*: scored, never escalated.
   A device that may send keystrokes is always judged.

## Status

| Component | State |
|---|---|
| usbmon reader, features, Stage 1, Stage 2, policy, storage, API, dashboard | done |
| systemd deployment | done; verified on a real Ubuntu 24.04 kernel ([`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)) |
| Real-kernel testbed (4 distributions, 11 device types, replayed human input) | done — `testbed/` |
| Real users on physical devices | tooling done (`scripts/field_capture.py`, [`docs/FIELD_CAPTURE.md`](docs/FIELD_CAPTURE.md)); no contributions yet |
| Enforcement (THROTTLE / DEAUTHORIZE) | generated and logged; **not applied** (`ENFORCEMENT_REVIEWED = False`) |
| eBPF capture | hook points verified (exact agreement with usbmon); not a capture backend |

## Results (all regenerable; every number comes from a seeded script)

| Question | Answer | Script |
|---|---|---|
| Fresh simulated attacks detected (Corpus C, 600 sessions) | **74.7%** (448/600) at **0/3000** benign FP | `eval/analysis_calibration.py` |
| ... before per-device-hour calibration | 82.7% (496/600) at 2/3000 FP | same |
| Keyboard false alarms on fresh long sessions | **0.013 per hour** (1 in 80 h); 0.225 before calibration | same |
| Attack families never seen in training (leave-one-out) | **68.2%** with the capability rule; 34.8% model alone | `eval/analysis_hybrid.py` |
| Hand-written rules alone | 66.7% at 0/3000 FP | `eval/analysis_rules.py` |
| Real kernels (4 distributions, 9 h): benign devices alarmed | **0 of 114** device-sessions (41.4 device-hours) | `testbed/` |
| Real kernels: attacks detected | injection-speed **24/24** (median 2.6–2.9 s); human-speed mimicry **0/16** | `python3 -m testbed.report` |

**Read the real-kernel row carefully.** Those detections are the capability rule's. The
learned model alone, at its calibrated threshold, caught 1 of 40 real-kernel attacks; at
the old per-window threshold it caught the 24 but alarmed on 6 of 12 human-typed
keyboards. In simulation, what the model adds over the rules is one family — a device
replaying a keystroke loop at human speed — and calibrating for a quiet working day
halves that (95% → 48%); on real kernels it caught none of them. Nothing here detects a
payload typed once at human speed.

## Live demo

```bash
make demo-dashboard        # then open http://localhost:8000/demo
```

Plug in simulated devices (a person typing, a mouse, a drive, several injectors, and the
two cases that are not caught) or recorded real-kernel devices, and watch each one's
Stage 2 score, the 4-of-6 gate and the policy state advance once a second. The page
streams the deployed models through the same `StreamingDetector` the monitor runs
(`tests/test_demo_api.py` checks the streamed scores equal the detector's own). Enforcement
is dry-run and the page says so. Rehearsal shortcut: `/demo?run=kbd,mouse,inject&speed=4`.

## Live lab (real kernel, live usbmon)

```bash
make lab                   # then open http://localhost:8001  (the demo page on :8000 can keep running)
```

Boots a Debian 13 guest under QEMU, streams its `usbmon` over SSH into the deployed monitor
loop, and scores the guest's USB bus once a second. Buttons on the page type as a person
(replayed timing of real typists), hot-plug a keystroke injector, move the mouse, read the
drive, and trigger the two cases that are not caught. The kernel, drivers and usbmon traffic
are real; the devices are QEMU's emulation. Needs QEMU and `testbed/images/debian-13.qcow2`
(`sh testbed/fetch_images.sh`). Stops the VM when you stop the server.

## Reproduce

```bash
make install && make test                  # editable install; the test suite
make eval                                  # Corpus-A numbers (the original study)
make train-deployed                        # deployed model on Corpus B, calibrated per device-hour
python3 -m eval.analysis_powered           # Corpus C (fresh, seed 2025)
python3 -m eval.analysis_calibration       # operating point and threshold curve
python3 -m eval.analysis_hybrid            # capability rule, leave-one-family-out
python3 -m eval.analysis_transfer          # why unseen families fail (transfer matrix)
python3 -m eval.model_card                 # docs/MODEL_CARD.md
make testbed-images && make testbed        # real-kernel testbed (~2 h, needs QEMU)
sudo ./deploy/install.sh --enable          # install on a Linux host
```

## Caveats that bound every claim

- **Simulated training data.** Both attack and benign traffic for training come from a
  simulator; Corpus B's typing is close to real people's on the measured statistics
  (`eval/analysis_human_timing.py`) but not identical.
- **Emulated devices.** The testbed's kernels and drivers are real; the devices are
  QEMU's, driven by recorded (not live) human input. No physical device or live user.
- **Attack realism.** Attacks are innocuous text typed with attack-shaped timing inside
  disposable VMs. They are not real BadUSB hardware.
- **Dry-run enforcement.** No device is ever deauthorised.
