# IUDBMS — working notes for Claude Code

Read this before touching code. It records why things are shaped the way they
are, what is already known to be broken, and what has been decided so future
sessions do not re-litigate it.

Specification: `docs/IUDBMS_Implementation_Plan.md` (authoritative for scope,
schedule and acceptance criteria). This file is authoritative for *decisions
taken since* the plan was written.

---

## 1. Current state

Milestones 1-7 and 10 are done (contract, usbmon reader, features, Stage 1,
Stage 2, policy engine, SQLite storage, evaluation harness). Not started:
the FastAPI layer, the React dashboard (M9), and eBPF (M8, deferred by plan §7).
`make eval` regenerates every reported number in ~65 s. `prototype/` is the Sprint-0 vertical slice: it runs end to end and
produced the numbers in `out/metrics.json`, but it is being ported module by
module into `src/iudbms/` and should not be extended in place.

**Nothing in this repo has ever seen real USB traffic.** Every measurement to
date comes from `prototype/simulator.py`.

---

## 2. Layout and why

```
src/iudbms/          the system, installed as a package (editable)
  contract.py        FROZEN interfaces — see §3
  capture/           L0/L1: usbmon, eBPF, replay, simulator (all EventSource)
  features/          L2: 28-dim extractor
  stage1/ stage2/    L3: iForest + EWMA baseline; PyTorch LSTM
  policy/ storage/ api/
prototype/           Sprint-0 code, kept runnable for comparison
tests/               mirrors src layout; fixtures/usbmon holds real line forms
eval/                one script per reported table; seeded
```

Two departures from plan §11:

- **`src/` layout.** The prototype's root-level `iudbms/` package shadowed
  anything else named `iudbms` whenever Python ran from the repo root, which
  silently defeats an editable install. `src/` makes the installed package the
  only importable one.
- **The simulator lives under `capture/`**, as one `EventSource` among four.
  This is deliberate: it forces the simulator to satisfy the same interface as
  the real reader instead of being privileged, so the §8.3 ablations can swap
  capture backends without touching anything downstream.

---

## 3. The frozen contract (`src/iudbms/contract.py`)

`CONTRACT_VERSION` is `1.0.0`. Bump it only alongside a retrain of every
checkpoint. `tests/test_contract.py::test_feature_order_is_frozen` holds a
SHA-256 digest of the feature names and will fail loudly on any reorder — that
order is baked into checkpoints, the SQLite `windows.features` JSON and every
published figure.

Four deviations from the prototype's `URBEvent`, each a bug fix rather than a
preference:

1. **`length` is the data length; `interval` is its own field.** usbmon prints
   the polling interval adjacent to the status (`0:8`), and the Sprint-0 parser
   read it *as* the length. See §4 defect 2.
2. **`data_truncated`** distinguishes a zero-length transfer from a payload
   usbmon chose not to capture (the `-` data tag). Without it, an uncaptured
   bulk stream looks like a perfectly repeating one to `payload_novelty` and
   `repeat_ratio`.
3. **`URBEvent` is frozen as well as slotted.** Events are values; freezing
   means one feature extractor cannot mutate the stream a later one sees.
4. **Interface arrival time moved onto each `Interface`** as `first_seen_ts`.
   The prototype kept one `late_iface_ts` on the descriptor and reconstructed
   visibility as `interfaces[:-1]` — correct only when exactly one interface
   arrives late *and* it sorts last. `visible_at(t)` is now total.

`CaptureStats` is part of the contract, not a debugging aid: plan §4.1 requires
"< 0.1 % of URBs lost" and "usbmon and eBPF agree within 1 %", and neither is
measurable unless every source reports what it discarded.

Validation is **not** in `__post_init__` — plan §4.1 wants ≥ 20k events/s.
Use `URBEvent.validate()` in tests and at source boundaries.

---

## 4. Sprint-0 defects found (audit, 07–09 Sep 2026)

Found by adversarial reading before any porting. Do not rediscover these.

1. **FIXED in M2.** **The parser silently dropped four real `usbmon` line forms.** Tested against
   `Documentation/usb/usbmon.rst`: control submissions with a setup packet
   (`s 80 06 ...`), *every* submission line (`<` data tag), every line where
   the payload was not captured (`-` tag), and iso completions carrying
   `status:interval:start_frame:errcount`. Dropped with no counter, so the
   plan's "< 0.1 % loss" criterion could not be evaluated at all.
2. **FIXED in M2.** **`length` read the polling interval.** Given `0:4` with a true data length
   of 8, the old parser returned 4. Invisible on simulator output only because
   the simulator emits `0:8` for 8-byte HID reports, so the two coincide.
   Affects `mean_len`, `max_len`, `bytes_in_rate`, `bytes_out_rate`.
3. **The simulator emits a stream real usbmon never produces.** Across 35
   sessions: `{iso/C: 79564, bulk/C: 36607, intr/C: 98827}` — no `S` events at
   all, and zero control transfers (`CTRL` is defined and never used). Two
   consequences: `frac_ctrl` is a **constant-zero feature** across the entire
   corpus, and real usbmon emits an S *and* a C per URB, so switching to live
   capture roughly doubles `n_events`/`ev_rate` and halves `mean_iat`. Plan §11's
   claim that the swap "is one line in `run_demo.py`" is therefore not true —
   every Stage-1 baseline and the Stage-2 threshold recalibrate.
4. **Residual duration leakage in `staged_delay`.** Aggregate means match
   (benign 63.7 s vs malicious 63.9 s), which is why the earlier fix looked
   complete. Per profile they do not: every profile draws `uniform(40, 85)`
   except `staged_delay`, which redraws `uniform(65, 85)` (mean 75.7). Any
   window past t ≈ 62 s is disproportionately `staged_delay`, whose post-onset
   windows are labelled malicious — and `log_time_since_enum` is a feature, so
   there is a direct path for the model to shortcut on session length.

---

## 5. Decisions log

| Date | Decision | Rationale |
|------|----------|-----------|
| 09 Sep | Fix simulator: emit S/C pairs and real control traffic | Makes `frac_ctrl` a live feature and moves the S/C recalibration to now rather than milestone 8. All plan §10 numbers regenerate. |
| 09 Sep | Fix `staged_delay` duration draw to the shared `uniform(40, 85)` | Defect 4. Numbers may fall; that is the correct outcome and is reported, not hidden. |
| 09 Sep | Keep `log_time_since_enum`, add leave-one-feature-out ablation | Legitimate in deployment, but it is the channel defect 4 flows through, so its contribution is measured rather than assumed. |
| 09 Sep | VM used for *recorded* captures, not live demos | Capture 60 s of real usbmon on the Linux VM, commit it under `tests/fixtures/usbmon/`, replay through the identical `EventSource` path. Avoids USB passthrough and root prompts during a review, and doubles as a permanent regression fixture. |
| 09 Sep | `src/` layout; simulator demoted to one `EventSource` | See §2. |

### Milestone 2 notes

`capture/usbmon.py` parses by whitespace tokenisation, not one large regex: the
optional-field combinatorics in the status word make a single pattern slower
and much harder to audit, and tokenisation tolerates unknown fields (iso frame
descriptors) by scanning forward for the data tag instead of failing outright.
Measured 194k events/s — ~10x the plan §4.1 floor of 20k.

`UsbmonSource` reads a live debugfs node and a recorded file through the same
code path, which is what makes a VM-recorded trace replayable on macOS.
`scripts/vm_capture.sh` records both the trace and a sysfs descriptor snapshot;
run it on the VM and commit the output under `tests/fixtures/usbmon/`.

Contract bumped to **1.1.0**: `URBEvent.setup` was added when control
submissions turned out to have nowhere to put their 8-byte SETUP packet. It is
kept out of `data` because `length` is the requested *data* length, routinely
smaller than 8, so SETUP in `data` would break the `len(data) <= length`
invariant. Append-only with a default, and no checkpoints existed yet, so
nothing retrained.

### Milestone 3 notes

**Features are computed over completions and errors only (`C`, `E`), never
submissions (`S`).** Every submission has a matching completion, so counting
both doubles every rate while adding no information and makes the result depend
on kernel scheduling rather than device behaviour; the completion is also the
event that actually moved data (a submission's `length` is only the *requested*
size).

This resolves defect 3's second half. Adding `S` events to the simulator makes
the trace realistic **without shifting any baseline**, because L2 ignores them —
`test_submissions_do_not_change_the_vector` asserts the vector is bit-identical
with and without them. It is also what lets the usbmon and eBPF backends agree
on event counts (plan §4.1) regardless of whether a backend reports submissions.

`frac_ctrl` is live as soon as the corpus contains control traffic; that half of
defect 3 is fixed by the simulator work, not here.

### Milestones 4-7, 10 notes — two defects found by the ablations

Both were found by *running* an ablation that the plan had only asserted the
result of. Neither was visible in any previously reported number.

**5. Pillar N2 was inert — the per-device baseline contributed nothing.**
`DeviceBaseline` z-scores in standardised space, and many features are
near-constant for a given device (a keyboard's `frac_iso` is always 0, `has_hid`
always 1, `frac_ctrl` is 0 except during enumeration). Their EWMA variance
collapses toward zero, so any change at all produces an enormous |z|. Measured
on a 200-session corpus the benign device-deviation had median 0.8 but a 99th
percentile of 90.8 and a maximum of 1267 — and since `calibrate_dev()` divides
by that 99th percentile, the 0.45-weighted device term contributed ~0.009 to a
score in [0, 1]. Pillar N2 was in the architecture diagram and absent from the
arithmetic.

Fixed with `VAR_FLOOR = 0.05` (never claim a device is more stable than 5% of
population variance for a feature). Swept {0, 0.01, 0.05, 0.1, 0.25, 0.5}; 0.05
maximised Stage-1 AUC and cut benign window FPR at the operating threshold from
0.26% to 0.07%. Symptom that led here: the Stage-1-only ablation reported
*exactly zero* detection, because Stage-1 scores were compressed into a narrow
band where the benign and malicious upper tails overlapped.

**6. The `mimicry` profile was not actually timing-indistinguishable.**
It drew think-pauses from `uniform(0.35, 1.1)` while human typing used
`uniform(0.4, 3.0)`, leaving a blatant timing signature in the one profile whose
entire purpose is to lack one. Consequence: a **timing-only** model detected
mimicry at 100%, and every single modality independently reached 100% session
detection — which made the N1 fusion ablation look like fusion was pointless,
while simultaneously inflating overall detection to 100%.

Fixed by sharing one set of human timing constants between both generators
(`HUMAN_IAT_LOGNORM`, `HUMAN_BURST_KEYS`, `HUMAN_PAUSE_RANGE`), with a KS test
asserting the two are not separable. Mimicry now leaves only a structural tell
(a replayed keystroke script).

**This is why overall detection fell from 100% to 81%, and it is the correct
direction.** The plan's §10 claim that "class_exfil is invisible to timing
features alone" was asserted, never measured; the ablation shows it false.
Report the 81% and the mimicry bound, not the old 100%.

### Current headline numbers (`make eval`, seed 7)

| model | window AUC | session detection |
|---|---|---|
| IUDBMS (S1+S2) | 0.9733 | 81.0% |
| Stage 1 only | 0.8894 | 42.9% |
| identity whitelist | — | 0.0% |

Benign session FP 0.0%, median latency 3.9 s. Per class: `class_exfil`,
`hid_injection`, `identity_switch`, `staged_delay` all 100%; **`mimicry` 20%**.
No single modality beats the fused 28 features (best single: protocol, 78.6%).

`log_time_since_enum` leave-one-out: AUC −0.0018, detection +2.4%. The feature
is not load-bearing, so decision C (keep it) stands — but it is measured now,
not assumed.

### eBPF go/no-go — **PENDING**

Plan §2 sets the decision at end of W6: proceed only if the usbmon path is
stable, features are frozen, and Stage 1 is training. Not yet evaluated.
Plan §7 designates W11 as the sacrifice week, so the fallback position is
"ship usbmon, reframe eBPF as documented future work" and nothing else moves.

---

## 6. Environment

Development host is **macOS/arm64** — there is no `/sys/kernel/debug` and no
eBPF. The `usbmon` reader is therefore built and unit-tested against recorded
fixtures here; live validation ("60 s capture, < 0.1 % loss") requires the
Linux VM. Do not take a VM capture until defect 1 is fixed, or most of the file
will be silently unparsed.

A `CLAUDE.md` for an unrelated project ("Is It Legal?", a legal-RAG system)
sits at `/Users/darshsharma/Downloads/CLAUDE.md` and is picked up as project
instructions for anything above this directory. **None of it applies here.**
This file shadows it for everything inside the repo.

---

## 7. Working rules

- **Session-stratified splits, always.** Never split by window. Windows from
  one session appearing in both train and test is the most likely way this
  project reports fake accuracy (plan §5).
- **Flag results that look too good.** A window AUC of 1.000, or epoch-1
  validation accuracy of 1.0, is a leakage signal, not a win. Defect 4 is a
  live example that survived one round of fixing.
- **Every claimed metric comes from a seeded script in `eval/`** that can be
  re-run. No number goes into a report or slide that wasn't printed by code.
- **Tests as you go**, one file per module, real assertions with thresholds —
  not "it didn't crash".
- Commit after each numbered milestone, referencing it in the message.
