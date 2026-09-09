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

Regenerate with `make eval` (seeded, ~106 s). Corpus: 1000 synthetic sessions,
12,823,600 usbmon lines, 62,986 windows. Split by session, stratified
by profile: fit=562 / val=140 / test=298.

Deployed model uses **28 of the 29 features**
(`log_time_since_enum` excluded as a label leak). Threshold calibrated at a
0.01% per-window FPR — selected on a held-out
*selection* split by `eval/tune.py`, never off the test curve.

| model | window ROC-AUC | precision | recall | session detection |
|---|---|---|---|---|
| IUDBMS (Stage 1 + Stage 2) | 0.9945 | 0.998 | 0.929 | **100.0%** |
| Stage 1 only (no temporal model) | 0.9363 | 0.985 | 0.094 | 11.7% |
| Identity whitelist (USBGuard-style) | — | — | — | 0.0% |

0.0% benign session false positives across
178 benign test sessions; median latency
3.6 s, p90 22.1 s.

| attack | n | IUDBMS | Stage-1 only | median latency |
|---|---|---|---|---|
| `class_exfil` | 29 | 100% | 100% | 3.5 s |
| `hid_injection` | 26 | 100% | 0% | 3.6 s |
| `identity_switch` | 23 | 100% | 100% | 3.5 s |
| `mimicry` | 16 | 100% | 0% | 24.9 s |
| `staged_delay` | 26 | 100% | 4% | 3.5 s |

### Read this before quoting 100 %

A perfect detection rate is a leakage signal more often than a win, and the
evaluation harness prints a warning when it sees one. Three things were checked
before reporting it:

- **The leakage audit passes** — no separable duration, window-count or VID:PID
  axis between benign and malicious sessions.
- **The new feature is behaviour, not session position.** `payload_seq_novelty`
  decays over a session by nature, and malicious windows are post-onset (hence
  later) by construction — the exact shape of the leak that got
  `log_time_since_enum` removed. Under positional control it holds up: benign
  typing stays at ~0.99 at *every* window position while mimicry decays to 0.72,
  and late benign windows (0.991) score **above** early mimicry ones (0.975).
  A regression test asserts that ordering.
- **`mimicry` costs 25 s to detect against ~3.5 s for every other class.**
  It is still by far the hardest case; the script has to cycle before its
  novelty decays. That 7× latency gap is itself the result.

**The honest caveat: `payload_seq_novelty` was designed after seeing which
attack we were failing, knowing how the simulator builds it.** That is
overfitting to the corpus and must be declared. The defence is that the
constraint is real — a keystroke-injection payload has to type specific
commands, so it cannot randomise its sequence without ceasing to be a payload.
The claim is "raises the cost", not "solves".

Single-modality ablation (novelty pillar N1):

| modality | features | window AUC | session detection |
|---|---|---|---|
| timing | 9 | 0.9836 | 87.5% |
| payload | 8 | 0.9937 | 100.0% |
| protocol | 6 | 0.9752 | 86.7% |
| identity | 6 | 0.9111 | 45.0% |
| **deployed (fused)** | **28** | **0.9937** | **100.0%** |

**N1 remains unsupported.** The payload modality alone now matches the fused
model exactly. Establishing that fusion earns its keep needs attack families
that differ from benign traffic on one axis at a time; ours differ on several.

Operating point (plan §6 — the curve, not one number):

| target per-window FPR | threshold | detection | benign session FP |
|---|---|---|---|
| 0.01% | 0.9147 | 100.0% | 0.0% |  ← deployed
| 0.05% | 0.8147 | 100.0% | 0.6% |
| 0.10% | 0.6750 | 100.0% | 2.8% |
| 0.50% | 0.4702 | 100.0% | 5.1% |
| 1.00% | 0.3775 | 100.0% | 6.7% |
| 2.00% | 0.2606 | 100.0% | 10.7% |

Descriptive only. The deployed row was chosen on a selection split, not read off
this table — that would be selecting on test.

## Honest caveats

- Every number currently in the repo comes from **synthetic** traffic produced
  by `prototype/simulator.py`. Real captures will be noisier.
- Attack families are generated by the same code the feature set was designed
  against, so per-profile detection rates are optimistic by construction.
- The capture layer has not yet been validated against real `usbmon` output.
  Four real line forms are known to be dropped by the Sprint-0 parser; see
  `CLAUDE.md` § "Sprint-0 defects".
