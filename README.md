# IUDBMS — Intelligent USB Device Behaviour Monitoring System

Trust USB peripherals based on **observed runtime behaviour**, not declared
VID/PID/class descriptors. A BadUSB device presents a valid keyboard
descriptor, so identity-based defences (USBGuard whitelists, GoodUSB profiles)
authorise it before any malicious behaviour exists to observe. IUDBMS scores
behaviour continuously and revokes authorisation mid-session.

Architectural ancestor: **LBM** (Tian et al., IEEE S&P 2019) — eBPF kernel
mediation with static, administrator-written rules. We extend it along four
axes:

| | Pillar | Evidence |
|---|---|---|
| N1 | Multi-modal feature fusion (28-dim, four modalities) | per-modality ablation |
| N2 | Per-device adaptive baselines (online EWMA + class-conditioned models) | cold-start / drift tests |
| N3 | LSTM temporal modelling over window sequences | Stage-1-only ablation |
| N4 | Closed-loop **learned** enforcement | end-to-end revocation latency |

Full specification: [`docs/IUDBMS_Implementation_Plan.md`](docs/IUDBMS_Implementation_Plan.md).
Working notes and decisions: [`CLAUDE.md`](CLAUDE.md).
Report documents (standalone HTML, print to PDF): [`docs/report/`](docs/report/).

## Status

| # | Milestone | State |
|---|-----------|-------|
| 1 | Repo skeleton + frozen data contract | **done** |
| 2 | `usbmon` reader (debugfs) | **done** (fixture-validated; live capture pending VM) |
| 3 | Feature extractor (28-dim) | **done** |
| 4 | Stage 1 — iForest + per-device EWMA baseline | **done** |
| 5 | Stage 2 — PyTorch LSTM | **done** |
| 6 | Policy engine — k-of-n gate, escalation, safety interlocks | **done** |
| 7 | SQLite storage + FastAPI | **done** — plus the raw URB trace (plan §5), pseudonymised by default |
| 8 | eBPF capture path | deferred (plan §7 W11 — designated sacrifice) |
| 9 | React dashboard | **done** (`iudbms-api` → http://127.0.0.1:8000) |
| 10 | Evaluation harness + ablations | **done** |
| — | Deployment: systemd services, install/verify/uninstall | **built**; verified off-repo from a wheel on macOS, systemd side pending the Linux VM — [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |

`prototype/` holds the Sprint-0 vertical slice (NumPy LSTM, synthetic data,
dry-run enforcement) that milestone work is ported from and measured against.
Its numbers are in [`out/metrics.json`](out/metrics.json) and are **not**
project results — see the caveats in plan §10 and the defect list in
`CLAUDE.md`.

## Quick start

```bash
make install                              # editable install + dev extras
make test                                 # 371 tests
make eval                                 # regenerate every reported number (~5 min cold)
python3 -m eval.leakage_audit             # plan §5 mandatory leakage audit
python3 -m eval.run_eval --persist        # also populate out/iudbms.db
make dashboard                            # operator dashboard on :8000 (evaluation DB)
make demo                                 # replay: watch the loop close (any OS)
make monitor-replay                       # the real monitor on a recorded capture -> out/live.db
sudo iudbms-monitor                       # LIVE capture (Linux + root only)
sudo ./deploy/install.sh --enable         # install as systemd services (docs/DEPLOYMENT.md)
make demo-hard                            # the same, on the mimicry adversary
make baseline                             # re-run the Sprint-0 prototype for comparison
```

## Measured results

Regenerate with `make eval` (seeded, ~115 s). Corpus: 1000 synthetic sessions,
11,367,542 usbmon lines, 62,791 windows. Split by session, stratified
by profile: fit=562 / val=140 / test=298. Deployed model uses
28 of 29 features, K=10 window horizon.

| model | window ROC-AUC | precision | recall | session detection |
|---|---|---|---|---|
| IUDBMS (Stage 1 + Stage 2) | 0.9762 | 0.999 | 0.749 | **89.2%** |
| Stage 1 only (no temporal model) | 0.8847 | 0.985 | 0.145 | 15.8% |
| Identity whitelist (USBGuard-style) | — | — | — | 0.0% |

0.0% benign session false positives across 178
benign test sessions; median latency 4.3 s.

| attack | n | IUDBMS | Stage-1 only | median latency |
|---|---|---|---|---|
| `class_exfil` | 24 | 100% | 21% | 4.2 s |
| `hid_injection` | 21 | 100% | 0% | 4.4 s |
| `identity_switch` | 20 | 100% | 70% | 3.5 s |
| `mimicry` | 15 | 93% | 0% | 24.4 s |
| `mimicry_adaptive` | 20 | 40% | 0% | 51.2 s |
| `staged_delay` | 20 | 100% | 0% | 4.3 s |

### The two mimicry variants, and why they differ

Both clone a keyboard descriptor and draw keystroke timing from the same
distribution as a human operator. They differ in one respect:

- **`mimicry`** cycles a short script, so its payload transitions exhaust
  themselves. `payload_seq_novelty` catches this — **93%**.
- **`mimicry_adaptive`** types a finite payload **once** and the device then
  falls silent, which is what a real BadUSB does. No transition repeats, so that
  feature has nothing to bite on. Detection is **40%** at
  51 s, resting entirely on the sustained silence after the payload.

The limiting case — a payload of unbounded length drawing fresh keycodes forever
— is *statistically the benign keyboard process*, and a per-feature
Kolmogorov–Smirnov scan finds zero of the 28 deployed features separate it. That
is a bound on what single-device HID behaviour can reveal. The principled next
defence is cross-device correlation: a human typing produces concurrent mouse
activity, a BadUSB typing alone does not.

Single-modality ablation (novelty pillar N1):

| modality | features | window AUC | session detection |
|---|---|---|---|
| timing | 9 | 0.9662 | 77.5% |
| payload | 8 | 0.9752 | 90.8% |
| protocol | 6 | 0.9458 | 77.5% |
| identity | 6 | 0.8884 | 39.2% |
| **deployed (fused)** | **28** | **0.9762** | **89.2%** |

## Honest caveats

- Every number currently in the repo comes from **synthetic** traffic produced
  by `prototype/simulator.py`. Real captures will be noisier.
- Attack families are generated by the same code the feature set was designed
  against, so per-profile detection rates are optimistic by construction.
- The capture layer has not yet been validated against real `usbmon` output.
  Four real line forms are known to be dropped by the Sprint-0 parser; see
  `CLAUDE.md` § "Sprint-0 defects".
