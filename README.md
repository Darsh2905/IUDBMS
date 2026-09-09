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

## Status

| # | Milestone | State |
|---|-----------|-------|
| 1 | Repo skeleton + frozen data contract | **done** |
| 2 | `usbmon` reader (debugfs) | **done** (fixture-validated; live capture pending VM) |
| 3 | Feature extractor (28-dim) | **done** |
| 4 | Stage 1 — iForest + per-device EWMA baseline | **done** |
| 5 | Stage 2 — PyTorch LSTM | **done** |
| 6 | Policy engine — k-of-n gate, escalation, safety interlocks | **done** |
| 7 | SQLite storage + FastAPI | **done** |
| 8 | eBPF capture path | deferred (plan §7 W11 — designated sacrifice) |
| 9 | React dashboard | not started |
| 10 | Evaluation harness + ablations | **done** |

`prototype/` holds the Sprint-0 vertical slice (NumPy LSTM, synthetic data,
dry-run enforcement) that milestone work is ported from and measured against.
Its numbers are in [`out/metrics.json`](out/metrics.json) and are **not**
project results — see the caveats in plan §10 and the defect list in
`CLAUDE.md`.

## Quick start

```bash
make install                              # editable install + dev extras
make test                                 # 189 tests
make eval                                 # regenerate every reported number (~70 s)
python3 -m eval.leakage_audit             # plan §5 mandatory leakage audit
python3 -m eval.run_eval --persist        # also populate out/iudbms.db
uvicorn iudbms.api.app:app                # control API on :8000
make baseline                             # re-run the Sprint-0 prototype for comparison
```

## Measured results

Regenerate with `make eval` (seeded, ~105 s). Corpus: 1000 synthetic sessions,
12,823,600 usbmon lines, 62,986 windows. Split by session, stratified
by profile: fit=562 / val=140 / test=298.

The deployed model uses **27 of the 28 features** —
`log_time_since_enum` is excluded because it leaks the label (see below).

| model | window ROC-AUC | precision | recall | session detection |
|---|---|---|---|---|
| IUDBMS (Stage 1 + Stage 2) | 0.9837 | 0.998 | 0.848 | **86.7%** |
| Stage 1 only (no temporal model) | 0.9387 | 0.997 | 0.450 | 43.3% |
| Identity whitelist (USBGuard-style) | — | — | — | 0.0% |

Benign session false-positive rate 0.0%; median detection latency
3.5 s (p90 3.8 s).

| attack | n | IUDBMS | Stage-1 only | median latency |
|---|---|---|---|---|
| `class_exfil` | 29 | 100% | 100% | 3.5 s |
| `hid_injection` | 26 | 100% | 0% | 3.6 s |
| `identity_switch` | 23 | 100% | 100% | 3.3 s |
| `mimicry` | 16 | 0% | 0% | — |
| `staged_delay` | 26 | 100% | 0% | 3.4 s |

**`mimicry` is detected 0 % of the time, on 16 test sessions.** It clones a keyboard
descriptor *and* draws keystroke timing from the same distribution as a human operator.
Only 1 of the 28 per-window features separates it from benign typing — and that one is
`log_time_since_enum`, which we removed as a leak. There is no per-window signal left for
the LSTM to integrate, which is why a sequence-horizon sweep (10 / 20 / 30 windows) moved
it not at all. Detecting it needs a *cross-window* payload feature the frozen 28 do not
contain. The plan anticipated this case (§8) and specified reporting it as a bound.

Single-modality ablation (novelty pillar N1):

| modality | features | window AUC | session detection |
|---|---|---|---|
| timing | 9 | 0.9836 | 87.5% |
| payload | 7 | 0.9832 | 86.7% |
| protocol | 6 | 0.9752 | 86.7% |
| identity | 6 | 0.9111 | 45.0% |
| **deployed (fused)** | **27** | **0.9837** | **86.7%** |

Fusion is *not* clearly earning its keep: timing alone reaches 87.5%,
which is one test session more than the fused model out of 120. On this corpus N1 is
unsupported; the honest reading is that our attack families differ from benign traffic on
several axes at once.

Operating point (plan §6 — the curve, not one number):

| target per-window FPR | threshold | detection | benign session FP |
|---|---|---|---|
| 0.01% | 0.7952 | 86.7% | 0.0% |
| 0.05% | 0.7386 | 86.7% | 0.0% |
| 0.10% | 0.7127 | 86.7% | 0.0% |  ← deployed
| 0.50% | 0.6093 | 87.5% | 3.9% |
| 1.00% | 0.5491 | 90.0% | 7.9% |
| 2.00% | 0.4621 | 94.2% | 12.9% |

The curve is descriptive. The deployed point is calibrated on *validation* benign windows;
picking a better row off this table would be selecting on the test split.

## Honest caveats

- Every number currently in the repo comes from **synthetic** traffic produced
  by `prototype/simulator.py`. Real captures will be noisier.
- Attack families are generated by the same code the feature set was designed
  against, so per-profile detection rates are optimistic by construction.
- The capture layer has not yet been validated against real `usbmon` output.
  Four real line forms are known to be dropped by the Sprint-0 parser; see
  `CLAUDE.md` § "Sprint-0 defects".
