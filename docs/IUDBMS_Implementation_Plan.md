# IUDBMS — Implementation Plan

**Intelligent USB Device Behaviour Monitoring System**
Operating Systems course project · team of three · 14-week schedule
Plan version 1.0 · baseline prototype measured 07 Sep 2026

---

## 0. Scope and thesis

IUDBMS decides whether a USB peripheral is trustworthy from **what it does after
enumeration**, not from what its descriptors claim to be. A BadUSB device
presents a valid keyboard descriptor, so identity-based defences (USBGuard
whitelists, GoodUSB's user-confirmed profiles) authorise it before any malicious
behaviour exists to observe. IUDBMS closes that window by scoring runtime
behaviour continuously and revoking authorisation mid-session.

The architectural reference is **LBM (Tian et al., IEEE S&P 2019)**, which
established eBPF-based peripheral mediation inside the Linux kernel. LBM
enforces *static, administrator-written* filter rules. IUDBMS extends that line
of work along four axes, which are the novelty claims this plan must deliver
evidence for:

| # | Pillar | What must exist at the end | Verified by |
|---|--------|----------------------------|-------------|
| N1 | Multi-modal feature fusion | 28-dim vector fusing identity, timing, payload and protocol modalities | Per-modality ablation (§8.3) |
| N2 | Per-device adaptive baselines | Online EWMA baseline per device, class-conditioned reference models | Cold-start and drift tests (§8.4) |
| N3 | Temporal sequence modelling | LSTM over window sequences detecting staged, delayed and mimicry attacks | Stage-1-only ablation (§8.3) |
| N4 | Closed-loop learned enforcement | Kernel action driven by model output, not a static rule table | End-to-end revocation latency (§8.2) |

Everything below is scheduled so that no pillar depends on a component that
lands after it.

---

## 1. System architecture

```
  ┌────────────────────────────────────────────────────────────────┐
  │  L5  Dashboard (React)  ── device trust, alerts, evidence view │
  ├────────────────────────────────────────────────────────────────┤
  │  L4  Control API (FastAPI)  ── policy CRUD, dry-run toggle     │
  ├────────────────────────────────────────────────────────────────┤
  │  L3  Decision engine                                           │
  │      Stage 1  per-device baseline + class-conditioned iForest  │
  │      Stage 2  LSTM over K-window sequences                     │
  │      Policy   k-of-n persistence gate → action                 │
  ├────────────────────────────────────────────────────────────────┤
  │  L2  Feature extractor  ── 1.0 s windows, 28 features          │
  ├────────────────────────────────────────────────────────────────┤
  │  L1  Normaliser  ── URB event records + descriptor snapshot    │
  ├────────────────────────────────────────────────────────────────┤
  │  L0  Capture  ── eBPF on usb_submit_urb / __usb_hcd_giveback   │
  │                  (fallback: usbmon debugfs reader)             │
  └────────────────────────────────────────────────────────────────┘
                 ↑ enforcement path ↓
        /sys/bus/usb/devices/<dev>/authorized   (deauthorise)
        USBGuard IPC rule injection             (block/reject)
```

**Data contract between layers is fixed on day one and never renegotiated.**
L1 emits `URBEvent{ts, evt, ttype, direction, bus, dev, ep, status, length,
data_prefix}`. L2 consumes an event list plus a descriptor dict and emits a
float32[28] per window. Every layer is independently testable because of this.

---

## 2. Capture layer: eBPF vs usbmon

This is the highest-risk decision in the project, so the plan de-risks it by
**building the fallback first and the eBPF path second**, behind one interface.

| | usbmon (debugfs) | eBPF (libbpf/CO-RE) |
|---|---|---|
| Effort | ~2 days | ~2.5 weeks |
| Kernel build needed | no | no (CO-RE), but headers + clang |
| Overhead | text formatting + userspace parse | in-kernel filtering, ring buffer |
| Enforcement | out-of-band only | can attach to the same hooks |
| Novelty framing | weak (well-trodden) | strong (direct LBM lineage) |
| Failure mode | none | verifier rejection, kernel version drift |

**Decision rule (evaluated end of W6):** if the usbmon path is stable, features
are frozen, and Stage 1 is training, proceed to eBPF. Otherwise ship usbmon and
reframe eBPF as documented future work. Both feed the identical `URBEvent`
stream, so this choice never blocks L2–L5.

eBPF attach points, once taken: `usb_submit_urb` (kprobe, submission) and
`__usb_hcd_giveback_urb` (kprobe, completion) with a `BPF_MAP_TYPE_RINGBUF`
carrying a 40-byte fixed record. Per-device state lives in a
`BPF_MAP_TYPE_HASH` keyed by `(busnum << 8) | devnum` so the kernel side can
drop uninteresting devices without a userspace round trip.

---

## 3. Technology stack

| Layer | Choice | Justification |
|-------|--------|---------------|
| L0 | C + libbpf (CO-RE); usbmon reader in Python | CO-RE avoids per-kernel rebuilds |
| L1 | Python 3.11, `dataclass(slots=True)` | slots keep 10⁶-event parses in RAM |
| L2 | NumPy | vectorised windowing, no framework tax |
| L3 Stage 1 | scikit-learn `IsolationForest` | unsupervised, no attack labels needed |
| L3 Stage 2 | PyTorch LSTM | standard, exportable to TorchScript |
| Storage | SQLite (WAL mode) | single-file, zero-admin, adequate write rate |
| L4 | FastAPI + uvicorn | async, OpenAPI docs for free |
| L5 | React + Recharts | live score timelines |
| Attack rig | Raspberry Pi Pico (TinyUSB) + Digispark | scriptable HID + composite-device tricks |
| Test host | Ubuntu 24.04 VM, snapshotted | enforcement bugs must not brick a laptop |

---

## 4. Module specifications

### 4.1 `capture/` — L0
- **Deliverable:** `usbmon_reader.py`, `iudbms_probe.bpf.c`, `loader.c`
- **Interface:** yields `URBEvent` at ≥ 20k events/s sustained
- **Acceptance:** replaying a 60 s keyboard+storage capture loses < 0.1 % of URBs;
  `usbmon` and eBPF paths produce event streams that agree on count within 1 %
- **Owner:** Member A (Darsh)

### 4.2 `features/` — L2
- **Window:** 1.0 s, hop 1.0 s (tumbling). Chosen so an injection burst of
  ~200 keystrokes/s is unmistakable within one window while a human typist's
  natural pause does not empty the window.
- **Feature vector (28), by modality:**
  - *timing (9)*: `n_events, ev_rate, mean_iat, std_iat, cv_iat, min_iat, p10_iat, iat_entropy, burstiness_fano`
  - *payload (7)*: `mean_len, std_len, max_len, bytes_in_rate, bytes_out_rate, payload_novelty, repeat_ratio`
  - *protocol (6)*: `n_endpoints, frac_intr, frac_bulk, frac_iso, frac_ctrl, frac_in`
  - *identity (6)*: `n_interfaces, has_hid, has_storage, class_behaviour_mismatch, late_iface_flag, log_time_since_enum`
- **Acceptance:** feature extraction ≥ 500 windows/s; every feature finite for
  empty windows (zero-traffic windows are legitimate input, not an error)
- **Owner:** Member B

### 4.3 `stage1/` — memoryless anomaly scoring
- Class-conditioned `IsolationForest` (one per device group: `hid`, `storage`,
  `av`, `other`) trained on benign windows only.
- Per-device `DeviceBaseline`: EWMA mean/variance per feature, α = 0.05,
  producing a deviation score against the device's *own* recent history (N2).
- Final Stage-1 score = calibrated max of the two, scaled by the 99th percentile
  of benign validation scores.
- **Acceptance:** benign window score distribution has p99 < 0.35; scoring
  latency < 1 ms/window.
- **Owner:** Member B

### 4.4 `stage2/` — temporal model
- Input: sequence of K = 10 windows × (28 features + Stage-1 score) = 29 dims.
- LSTM, hidden 32, per-timestep sigmoid output so detection latency is
  measurable (first window whose probability crosses threshold).
- Trained on labelled sessions; label applies from attack onset onward, so
  pre-onset windows of a malicious session are labelled benign. This is what
  forces the model to learn *transition*, not device identity.
- **Acceptance:** window AUC ≥ 0.95 on held-out sessions; detects the mimicry
  profile that Stage 1 cannot.
- **Owner:** Member C

### 4.5 `enforce/` — closed-loop actions
- `PolicyEngine` with a **k-of-n persistence gate** (currently 4-of-6). A single
  anomalous window never triggers enforcement; this is the main false-positive
  control and it is what makes automatic revocation safe enough to enable.
- Escalation: `OBSERVE → THROTTLE → DEAUTHORIZE`. Deauthorise writes
  `0` to `/sys/bus/usb/devices/<dev>/authorized`.
- Every action carries a generated rule object (device, trigger, evidence
  features, kernel command) which is what makes enforcement *learned* rather
  than static (N4).
- **Safety:** `dry_run=True` is the default; an allowlist protects the boot
  keyboard and root filesystem device; a physical kill switch (env var checked
  each tick) disables all writes.
- **Owner:** Member A

### 4.6 `storage/` and API — L4/L5
- SQLite tables: `devices`, `windows`, `baselines`, `alerts`, `actions`.
  `alerts(id, device_id, ts, stage1, stage2, severity, evidence)`.
- API endpoints: `GET /devices`, `GET /devices/{id}/timeline`,
  `POST /devices/{id}/authorize`, `GET /alerts`, `POST /policy`.
- **Owner:** Member C

---

## 5. Data plan

Three corpora, in this order:

1. **Benign, real (target 8–10 h).** Two team laptops + the test VM, everyday
   use: typing, mouse, USB storage copies, webcam calls, audio headsets. Capture
   with the usbmon reader; store raw traces, not just features, so the feature
   set can be revised without re-collecting.
2. **Malicious, real (target 60–80 sessions).** Pi Pico programmed with five
   payload families: immediate HID injection, delayed injection, composite
   class-switch (storage→HID), covert bulk-out exfiltration, and a **rate-limited
   mimicry payload** that throttles keystrokes to a human inter-arrival
   distribution. The mimicry family is the honest hard case and is deliberately
   included rather than avoided.
3. **Synthetic (unbounded).** The simulator in §12 generates statistically
   grounded traces in usbmon's on-wire format. Used for early development and
   for scaling experiments, never as the sole evidence in the final report.

**Split discipline:** split by *session*, stratified by profile — never by
window. Windows from one session in both train and test is the most likely way
this project accidentally reports fake accuracy.

**Leakage audit (mandatory, W9):** confirm benign and malicious sessions share
one duration distribution, one device-model distribution, and one capture-host
distribution. A finding from the prototype: an early version drew attack
sessions from a longer duration range, which let the models separate classes on
session length alone.

---

## 6. Evaluation protocol

- **Window level:** ROC-AUC, precision/recall/F1 at a threshold calibrated to a
  target FPR on *validation* benign windows only.
- **Session level:** attack detection rate, benign false-positive rate,
  alerts per benign hour, detection latency (first alert − attack onset).
- **Baselines to beat:**
  1. identity whitelist (USBGuard-style VID:PID allowlist) — expected 0 %
  2. Stage 1 only (no temporal model) — isolates pillar N3
  3. single-modality variants — isolates pillar N1
- **Threshold calibration is a reported result, not a hidden knob.** Per-window
  FPR compounds across a session: at 0.5 % per window over ~70 windows the
  prototype produced a 9.1 % benign session FP rate; tightening to 0.1 % and
  4-of-6 persistence brought it to 3.0 % with no loss of detection. Report the
  full trade-off curve.

---

## 7. Schedule (14 weeks from 08 Sep 2026)

| Week | Milestone | Exit criterion |
|------|-----------|----------------|
| W1 | Literature review locked (12–15 papers, 4 themes) | References + inference section written |
| W2 | Repo, CI, VM snapshot, data contract frozen | `URBEvent` schema merged; unit tests green |
| W3 | usbmon reader + parser | 60 s live capture parsed, < 0.1 % loss |
| W4 | Feature extractor + unit tests | 28 features, empty-window safe |
| W5 | **Benign collection begins** (runs in background to W10) | ≥ 2 h captured |
| W6 | Stage 1 + per-device baselines | Benign p99 < 0.35; **eBPF go/no-go decision** |
| W7 | Attack rig: Pico payloads 1–3 | 30 malicious sessions captured |
| W8 | Stage 2 LSTM trained on synthetic + early real | Window AUC ≥ 0.95 on held-out |
| W9 | Attack rig: payloads 4–5 (exfil, mimicry); **leakage audit** | Audit signed off in writing |
| W10 | Policy engine + enforcement, dry-run only | Revocation fires on replayed attack |
| W11 | eBPF capture path (if W6 said go) | Event counts agree with usbmon within 1 % |
| W12 | API + dashboard | Live timeline for a plugged-in device |
| W13 | Full evaluation + all ablations | Every table in §6 populated |
| W14 | Report, demo video, viva rehearsal | Frozen tag + reproducible run script |

**Slack:** W11 is the designated sacrifice week. If the schedule slips, eBPF
drops to future work and nothing else moves.

**Parallelism:** A on capture/enforcement, B on features/Stage 1, C on Stage
2/API/dashboard. The frozen data contract (W2) is what lets three people work
without blocking each other.

---

## 8. Risk register

| Risk | L | I | Mitigation |
|------|---|---|------------|
| eBPF verifier / kernel version pain | H | M | usbmon fallback built first; W6 go/no-go |
| Enforcement bricks the dev machine | M | H | VM + snapshots; dry-run default; boot-keyboard allowlist |
| Too little benign data → overfit baselines | M | H | Collection starts W5 and runs 6 weeks in background |
| False positives make the system unusable | H | M | k-of-n gate; FPR reported as a curve, not one number |
| Mimicry attacks defeat the model | M | M | Included deliberately; if undetected, report honestly as a bound |
| Session-level leakage inflates results | M | H | Session-stratified splits + W9 audit |
| Team member unavailable | M | M | Layer interfaces frozen; any layer stubbable |

---

## 9. Deliverables checklist

- [ ] Literature review, 12–15 papers, four themes, IEEE-format references
- [ ] Source repository with the six-layer implementation
- [ ] Benign + malicious capture corpus with a datasheet (hosts, devices, durations)
- [ ] Trained Stage 1 and Stage 2 models with training scripts
- [ ] Evaluation notebook reproducing every reported table and figure
- [ ] Dashboard demo (video, in case hardware fails during the viva)
- [ ] Final report: problem statement, gap, novelty (N1–N4), objectives, results, limitations
- [ ] Reproduction script: one command, fixed seed

---

## 10. Appendix A — Sprint-0 prototype: measured results

A complete, runnable vertical slice of every layer exists and was executed to
produce these numbers. It uses a synthetic corpus in usbmon's on-wire format,
an in-NumPy LSTM (PyTorch was unavailable in that environment), and dry-run
enforcement. It is a plan-validation artefact, **not** the project result.

**Corpus:** 350 sessions (210 benign / 140 malicious), 2,121,580 usbmon lines,
22,155 one-second windows, 28 features. Session-stratified split:
192 fit / 48 validation / 110 test. Total runtime 46 s.

**Window-level, held-out test set**

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Stage 1 only | 0.835 | 0.994 | 0.360 | 0.528 |
| Stage 1 + Stage 2 | **0.998** | 0.987 | 0.925 | 0.955 |

**Session-level detection** (threshold at 0.1 % validation FPR, 4-of-6 gate)

| Attack profile | n | IUDBMS | Stage-1 only | Identity whitelist | Median latency |
|----------------|---|--------|--------------|--------------------|----------------|
| hid_injection | 7 | 100 % | 0 % | 0 % | 3.6 s |
| identity_switch | 11 | 100 % | 100 % | 0 % | 3.5 s |
| staged_delay | 8 | 100 % | 0 % | 0 % | 3.8 s |
| class_exfil | 7 | 100 % | 100 % | 0 % | 3.2 s |
| mimicry | 11 | 100 % | 0 % | 0 % | 12.2 s |
| **overall** | 44 | **100 %** | 40.9 % | 0 % | 3.8 s (p90 12.5 s) |

Benign false positives: 3.0 % of sessions, 1.72 alerts per benign hour.

**What this validates, pillar by pillar**

- **N1** — `class_exfil` (an HID-declared device issuing bulk-OUT transfers) is
  invisible to timing features alone and is caught by the protocol/identity
  modality. Fusion is doing work.
- **N2** — class-conditioned reference models plus per-device EWMA keep the
  benign p99 low across five very different device classes; a single global
  model conflates a 125 Hz mouse with a human typist.
- **N3** — the decisive result. Stage 1 catches the two attacks with a static
  signature (`identity_switch`, `class_exfil`) and misses all three that are
  defined by a *transition over time*. Stage 2 takes overall session detection
  from 40.9 % to 100 %. The mimicry profile — human-calibrated inter-arrival
  timing — is undetectable per window and is caught only over a sequence, at
  roughly 4× the latency of the crude attacks. That latency gap is itself a
  result worth reporting.
- **N4** — enforcement fires from model output with a median 3.8 s from attack
  onset to revocation, carrying a generated rule with its evidence features. In
  the plotted session the LSTM produces one isolated spike at t ≈ 19 s that the
  4-of-6 gate correctly suppresses; the real onset at t = 37 s produces a
  sustained crossing and revocation at t = 41 s.

**Caveats to state plainly in the report:** synthetic traffic, small per-profile
test counts (n = 7–11), attack families generated by the same code that the
feature set was designed against. Real captures will be noisier and the numbers
should be expected to fall.

---

## 11. Appendix B — repository layout and reproduction

```
iudbms/
├── iudbms/
│   ├── simulator.py       # synthetic usbmon traces + descriptors (dev only)
│   ├── usbmon_parser.py   # L1: usbmon text → URBEvent
│   ├── features.py        # L2: 1 s windows → 28 features
│   ├── stage1.py          # L3: baselines + class-conditioned iForest
│   ├── stage2_lstm.py     # L3: LSTM with BPTT
│   ├── enforcement.py     # policy engine, k-of-n gate, rule generation
│   └── storage.py         # SQLite schema and writers
├── run_demo.py            # end-to-end: capture → … → enforce → figures
├── out/                   # run.log, metrics.json, iudbms.db, figures
└── docs/
```

```bash
python3 run_demo.py     # seed=7, deterministic, ~47 s
```

Swapping the simulator for live capture is one line in `run_demo.py`: replace
`sim.session(...)` with the usbmon reader. Every layer below L0 already consumes
the real on-wire format, which is the point of building it this way.
