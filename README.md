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
| 7 | SQLite storage + FastAPI | **done** |
| 8 | eBPF capture path | deferred (plan §7 W11 — designated sacrifice) |
| 9 | React dashboard | **done** (`uvicorn iudbms.api.app:app` → http://localhost:8000) |
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
make dashboard                            # operator dashboard on :8000
make demo                                 # live replay: watch the loop close
make demo-hard                            # the same, on the mimicry adversary
make baseline                             # re-run the Sprint-0 prototype for comparison
```

## Measured results

Regenerate with `make eval` (seeded, ~110 s). Corpus: 1000 synthetic sessions,
11,333,822 usbmon lines, 62,070 windows. Split by session, stratified
by profile: fit=562 / val=140 / test=298. Deployed model uses
28 of 29 features.

| model | window ROC-AUC | precision | recall | session detection |
|---|---|---|---|---|
| IUDBMS (Stage 1 + Stage 2) | 0.9753 | 0.997 | 0.747 | **84.2%** |
| Stage 1 only (no temporal model) | 0.8922 | 0.986 | 0.162 | 18.3% |
| Identity whitelist (USBGuard-style) | — | — | — | 0.0% |

0.0% benign session false positives across 178
benign test sessions; median latency 3.7 s.

| attack | n | IUDBMS | Stage-1 only | median latency |
|---|---|---|---|---|
| `class_exfil` | 24 | 100% | 21% | 3.4 s |
| `hid_injection` | 21 | 100% | 0% | 3.6 s |
| `identity_switch` | 20 | 100% | 85% | 3.5 s |
| `mimicry` | 15 | 100% | 0% | 27.9 s |
| `mimicry_adaptive` | 20 | 5% | 0% | 74.7 s |
| `staged_delay` | 20 | 100% | 0% | 3.7 s |

### The bound: a single-shot human-paced injection is undetectable

`mimicry_adaptive` is the strongest adversary in the corpus and **we detect
5% of it.** It clones a keyboard descriptor, draws its keystroke
timing from the same distribution as a human operator, and — the part that
matters — **types its payload once instead of cycling a script.** That is what a
real BadUSB does: it types `powershell -enc <base64>` and stops.

Our original `mimicry` cycled a 27-key script three times per session, and a
cycling script necessarily exhausts its own payload transitions. That is what
`payload_seq_novelty` detects. Against a payload typed once, a per-feature
Kolmogorov–Smirnov scan finds **zero of the 28 deployed features separate it
from benign typing** (the only separable feature is `log_time_since_enum`, which
we removed as a label leak). The 5% is noise, not detection.

**This is a bound on what single-device HID behaviour can reveal, not a tuning
failure.** The principled next defence is cross-device correlation — a human
typing produces concurrent mouse activity, a BadUSB typing alone does not — and
that sits outside the current single-device architecture.

One caveat in the other direction: our adaptive adversary draws fresh keycodes
indefinitely, so it is the *limiting case*. A real payload has finite content and
may be marginally easier.

Single-modality ablation (novelty pillar N1):

| modality | features | window AUC | session detection |
|---|---|---|---|
| timing | 9 | 0.9665 | 70.8% |
| payload | 8 | 0.9755 | 85.0% |
| protocol | 6 | 0.9509 | 70.8% |
| identity | 6 | 0.8940 | 39.2% |
| **deployed (fused)** | **28** | **0.9753** | **84.2%** |

## Honest caveats

- Every number currently in the repo comes from **synthetic** traffic produced
  by `prototype/simulator.py`. Real captures will be noisier.
- Attack families are generated by the same code the feature set was designed
  against, so per-profile detection rates are optimistic by construction.
- The capture layer has not yet been validated against real `usbmon` output.
  Four real line forms are known to be dropped by the Sprint-0 parser; see
  `CLAUDE.md` § "Sprint-0 defects".
