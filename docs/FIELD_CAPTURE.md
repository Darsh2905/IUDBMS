# Contributing a real session

Everything IUDBMS has measured on real kernels so far uses QEMU's emulated USB
devices driven by recorded human input. The missing evidence is **physical devices
used by real people on their own machines**. If you have a Linux laptop or desktop,
an hour of your normal work is the most useful thing you can give the project.

## What you need

* Linux with `usbmon` (every mainstream distribution kernel) and root.
* A checkout of this repository and Python 3.10+ with the package installed
  (`pip install -e .`).
* Your own USB devices: keyboard, mouse, a USB stick, a headset, a webcam,
  whatever you normally use. Nothing borrowed, nothing belonging to an employer.

## Record

```bash
sudo python3 scripts/field_capture.py --minutes 60 --alias <a-pseudonym>
```

Then work normally: type, browse, copy a file to and from a USB stick, take a call.
Plugging a device in or out during the run is useful; it is recorded. Ctrl-C ends
early. The result is `iudbms-field-<alias>-<stamp>.tar.gz`.

## What leaves your machine

* Timing, sizes, endpoints and transfer types of every USB transfer.
* Payload bytes **replaced as they are read** by a keyed, length-preserving
  permutation whose key exists only in memory for that run. Equal payloads stay
  equal, so every detector feature is exact (`tests/test_field_capture.py` checks
  this on real kernel output). Your keystrokes are never written to disk in clear.
* What each device declares (vendor/product IDs and strings, interface classes),
  the kernel version, the distribution, and which driver bound to each interface.

**Pseudonymous is not anonymous.** The mapping is consistent within a run, so a long
typing session is a substitution cipher and open to frequency analysis. If that is
not acceptable, record with `--no-payload`: payload bytes are dropped entirely and
three payload features cannot be computed. Share the archive only with the project
team.

## Score it

```bash
tar xzf iudbms-field-*.tar.gz -C out/testbed/raw/
python3 -m testbed.score out/testbed/raw/iudbms-field-*
```

Every device in a field capture is labelled benign, so any alert is a false alarm on
real hardware. That is the number this data exists to measure.
