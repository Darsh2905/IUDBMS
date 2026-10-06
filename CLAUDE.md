# IUDBMS — working notes for Claude Code

Read this before touching code. It records why things are shaped the way they
are, what is already known to be broken, and what has been decided so future
sessions do not re-litigate it.

Specification: `docs/IUDBMS_Implementation_Plan.md` (authoritative for scope,
schedule and acceptance criteria). This file is authoritative for *decisions
taken since* the plan was written.

---

## 1. Current state

Milestones 1-7, 9 and 10 are done. Only M8 (eBPF) remains, deferred by plan §7.
Run the dashboard with `iudbms-api` (or `make dashboard`) and open http://127.0.0.1:8000.
Deploy with `sudo ./deploy/install.sh` -- see `docs/DEPLOYMENT.md` and finding 25.
`iudbms.live.StreamingDetector` is the long-lived detector a deployment runs:
per-device extractor, baseline, sequence window and gate, advanced one window at
a time. `scripts/live_replay.py` drives it at wall-clock speed
(`make demo` / `make demo-hard`).
`make eval` regenerates every reported number (~5 min from cold, ~65 s with
a warm `out/corpus_cache.npz`). `make fusion-ablation`, `make
baseline-ablation` and `make cross-device` run the three novelty
experiments; `make paper` rebuilds the IEEE .docx from their outputs. `prototype/` is the Sprint-0 vertical slice: it runs end to end and
produced the numbers in `out/metrics.json`, but it is being ported module by
module into `src/iudbms/` and should not be extended in place.

**Nothing in this repo has ever seen real USB traffic.** Every measurement to
date comes from the simulator. This is the principal limitation and bounds every
claim below.

### Novelty scorecard (current, 05 Oct)

Each pillar now has an experiment built to falsify it, with its acceptance
criterion fixed before the first run. Two do not survive.

**Numbering.** The paper's §3.3 order is N1 fusion, N2 per-device baseline,
**N3 closed-loop enforcement, N4 temporal modelling** — and that is the order
the Review 1 submission established. Findings 1-16 in this file use N3 and N4
the other way round; they were written against the pipeline order and are left
as written rather than silently rewritten. **When reading an older finding,
check which claim is meant rather than trusting the number.** Everything from
finding 17 onward, and the paper, use the §3.3 order.

| pillar | verdict | evidence |
|---|---|---|
| N1 multi-modal fusion | **supported only non-uniformly** | `make fusion-ablation`, finding 21 |
| N2 per-device baseline | **not supported** | `make baseline-ablation`, finding 18 |
| N3 closed-loop enforcement | **dry-run only** | never executed against a kernel |
| N4 temporal modelling | **strongly supported** | 15.8% → 89.2%, findings 14-15 |

N1 is the nuanced one: fusion is the *only* configuration that detects
descriptor-only attacks (100% vs 3.7% for the best specialist, because the
identity modality alone cannot calibrate), and it gains 19.8 points on a
moderately diffuse attack — but it is beaten by a 9-feature subset of itself by
11.1 points when the attack is subtle enough. Report the curve, not a verdict.

N2's negative result is now *stronger* than before, not weaker: finding 14
attributed the baseline's inertness to device homogeneity in the corpus, and
removing that homogeneity does not rescue it.

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

`CONTRACT_VERSION` is `3.0.0` (1.1.0 added `setup`; 2.0.0 appended
`payload_seq_novelty`; 3.0.0 appended `concurrent_device_rate`, the first
host-level feature). Bump it only alongside a retrain of every
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

### Scale-up and tuning (09 Sep, second pass)

Corpus tripled to **1000 sessions** (12.8 M usbmon lines, 62,986 windows).
Test attack sessions per class went from 7-11 to 16-29, which is what makes the
per-class rates worth quoting at all. `out/corpus_cache.npz` caches it; delete
the file to rebuild.

**Tuning discipline.** `eval/tune.py` carves three disjoint inner splits out of
fit+val (train / calibrate / select) and **never constructs the test set**.
`run_configuration` now takes its splits as *roles* (`train`, `calib`,
`report`) rather than `fit/val/test`, and raises if calibration and scoring name
the same split — calibrating and scoring on the same sessions makes the FP rate
hit its target by construction, so every sweep would look equally good. Naming
them fit/val/test caused exactly that bug within minutes of being written.

**7. Sequence horizon is not a lever.** Swept T ∈ {10, 20, 30} × hidden ∈ {32, 64}
on the selection split: detection is 86.3% for all six configurations, longer
horizons slightly *reduce* AUC, and latency grows 3.6 s → 4.8 s. The hypothesis
that mimicry needs a longer memory is wrong, and finding 9 says why. Deployed
config stays T=10, hidden=32.

**8. `log_time_since_enum` is a systematic label leak — decision C REVERSED.**
Post-onset windows are later in a session *by construction*, so the feature
separates every attack class from benign traffic without encoding any
behaviour:

| profile | KS D | critical | verdict |
|---|---|---|---|
| hid_injection | 0.0333 | 0.0239 | leak |
| identity_switch | 0.3115 | 0.0301 | leak |
| staged_delay | 0.4095 | 0.0335 | leak |
| class_exfil | 0.1223 | 0.0239 | leak |
| mimicry | 0.1155 | 0.0309 | leak |

It encodes *our onset distribution*, which a real attacker chooses
adversarially. **The deployed model now uses 27 of 28 features.** Including it
buys AUC +0.0012 and detection +0.8% — reported as an ablation, not shipped.
The contract still emits 28; the model selects columns, so no version bump.

**9. `mimicry` is 0%, and that is the honest number.** The earlier 20% was 2 of
10 sessions — noise. On 16 sessions it is 0. Only **1 of 28 per-window features**
separates mimicry post-onset windows from benign keyboard windows, and it is the
leaking `log_time_since_enum`. The structural tell (a replayed keystroke script)
is real but **cross-window**: a 1 s window holds ~5 mostly-distinct keys, so
`payload_novelty` and `repeat_ratio` are indistinguishable per window. There is
nothing for the LSTM to integrate, which is why the horizon sweep moved nothing.
Detecting mimicry needs a cross-window payload feature the frozen 28 do not
contain. Do not tune at this; add the feature or report the bound.

### Current headline numbers

`make eval` == `python3 -m eval.run_eval --cache out/corpus_cache.npz
--target-fpr 0.0001 --persist`, seed 7.

| model | window AUC | recall | session detection |
|---|---|---|---|
| IUDBMS (S1+S2), 28 of 29 features, K=10 | 0.9762 | 0.749 | **89.2%** |
| Stage 1 only | 0.8847 | 0.145 | 15.8% |
| identity whitelist | — | — | 0.0% |

Benign session FP **0.0%** (0 of 180 benign test sessions). Median latency 4.3 s,
p90 22.1 s. Every attack class 100%: `class_exfil` (n=29), `hid_injection` (26),
`identity_switch` (23), `staged_delay` (26), **`mimicry` (16)**.

Target per-window FPR is now **0.0001**, selected by `eval/tune.py` on the
selection split (0.0001 / 0.0005 / 0.001 all gave 100% detection at 0% FP there;
the tightest was taken). It was NOT read off the test operating curve.

**Before quoting 100%, read this.** The harness prints three warnings on this
result and they are correct to fire. What was checked:

1. Leakage audit passes (duration, window count, VID:PID).
2. `payload_seq_novelty` is behaviour, not session position — the failure mode
   that killed `log_time_since_enum`. Positional control: benign typing holds
   ~0.99 at *every* window position while mimicry decays to 0.72, and late
   benign windows (0.991) score **above** early mimicry ones (0.975).
   `test_seq_novelty_is_behaviour_not_session_position` asserts that ordering.
3. `mimicry` costs **24.9 s** to detect against ~3.5 s for every other class —
   a 7× latency gap. It is still much the hardest case, which is what you would
   expect if the detection is real rather than leaked.

**Declare the overfitting.** `payload_seq_novelty` was designed after seeing
which attack we were failing, using knowledge of how the simulator builds
mimicry. The defence is that the constraint is real — an injection payload must
type specific commands and cannot randomise its sequence without ceasing to be a
payload — but the claim is "raises the cost", not "solves". An adversary who
pads or reorders within limits will degrade this.

**N1 is still not supported.** The payload modality alone now matches the fused
model exactly (AUC 0.9937, 100%). Establishing fusion needs attack families that
differ from benign traffic on one axis at a time; ours differ on several.

Operating curve: detection is 100% at every target FPR from 0.01% to 2%; what
changes is benign FP (0.0% → 10.7%). The 4-of-6 gate remains well placed —
3-of-6 matches it, 5-of-6 and 6-of-6 start losing detection.

### Contract 2.0.0 — the cross-window feature (09 Sep, third pass)

Finding 9 diagnosed mimicry precisely: its tell is a replayed keystroke script,
which is a **cross-window** property, and the 28 per-window features had nowhere
to put it. Contract 2.0.0 appends feature 29, `payload_seq_novelty`.

**What it measures.** The fraction of this window's payload *transitions*
(consecutive payload pairs) not seen earlier in the session, judged causally and
carried across window boundaries. A replayed script cycles a fixed set of
transitions so its novelty decays; a human types novel pairs indefinitely.
Measured on the simulator before implementing:

| profile | 1st third | 2nd third | 3rd third |
|---|---|---|---|
| keyboard (benign) | 0.997 | 0.994 | 0.982 |
| mimicry (post-onset) | 0.988 | 0.870 | 0.683 |
| hid_injection | 0.079 | 0.000 | 0.000 |
| staged_delay | 0.149 | 0.008 | 0.001 |

Note this is a *transition* count, not a payload-set count: both human and
scripted typing draw keycodes from a similar alphabet, so distinct-payload
novelty does **not** separate them. Order does.

**The extractor is now stateful.** One `WindowFeatureExtractor` belongs to one
device session; call `reset()` between sessions or the next device inherits the
previous one's history and reads as suspiciously familiar traffic.
`eval/corpus.py` constructs one per session. Remembered transitions are capped
at 8192 so a bulk device emitting thousands of distinct random payloads per
second cannot grow the set without bound.

**Honesty caveat to state in the report.** This feature was designed *after*
seeing which attack we were failing, using knowledge of how the simulator builds
mimicry. That is overfitting to the corpus and must be declared. The defence is
that the constraint is real, not synthetic: a keystroke-injection payload has to
type specific commands, so it cannot randomise its sequence without ceasing to
be a payload. An adversary can pad or reorder within limits, and the honest
claim is "raises the cost", not "solves".

### Milestone 9 — dashboard, and the defect it found (09 Sep, fourth pass)

`src/iudbms/api/static/index.html` (was `dashboard/index.html`; moved into the package
in finding 25), served by FastAPI at `/`. **React from cdnjs, no build
step, hand-rolled SVG chart instead of Recharts.** Deviation from plan §3, taken
deliberately: a Vite toolchain and a version-pinned chart library are two ways to
fail on a projector the night before a review. If the dashboard is ever extended
past a single view, revisit.

**10. An exploding standardisation scale — found by looking at live data.**
The dashboard rendered alert evidence as `late_iface_flag 4472091.23`. Three
features are *exactly* constant across benign traffic
(`class_behaviour_mismatch`, `late_iface_flag`, `n_interfaces` — 0 or 1 for every
benign device, by construction), and `Stage1._std` divided by `sd + 1e-6`. A
single 0→1 flip therefore became a standardised value of ~1e6 that swamped all
29 dimensions in both the forest and the baseline z-scores. Same family as the
`VAR_FLOOR` defect, one level up.

Fixed with `SD_FLOOR = 1e-2` and `STD_CLIP = 10`. Such a feature IS informative
when it changes — that is the whole point of `class_behaviour_mismatch` — so the
fix bounds its influence rather than removing it.

| | before | after |
|---|---|---|
| IUDBMS AUC | 0.9937 | 0.9945 |
| IUDBMS detection | 100% | 100% |
| IUDBMS benign FP | 0.0% | 0.6% |
| Stage-1 detection | 44.2% | **11.7%** |
| Stage-1 recall | 0.458 | 0.094 |

**The Stage-1 collapse is the finding.** Both runs used the same 0.0001 target
FPR, so it is entirely this fix. Its old 44.2% was substantially artefact:
`class_exfil` was caught because `class_behaviour_mismatch` exploded to 1e6, not
because the model ranked it. Stage 1 now catches only `identity_switch` (60.9%)
while its window AUC stays high (0.9363) — it still *ranks* well, it is the tight
operating point that costs recall. This strengthens N3 rather than weakening it.

### Milestone 11 (unplanned) — the streaming detector

Everything else scores a *stored* session: the harness builds an extractor and a
baseline per session, scores the whole array, throws the state away. Fine for
measurement, useless as a monitor, and the reason the `baselines` table was
always empty. It also meant **nothing demonstrated the closed loop the thesis
rests on** — a reviewer could fairly say we had built a batch classifier.

`src/iudbms/live/detector.py` holds the per-device state a deployment needs and
advances one window at a time with no future information. Causality is
structural rather than promised: it cannot see a future window because it has
not been given one.

`test_streaming_matches_offline_scoring` asserts live scores are numerically
identical to the offline ones. **Keep that test.** If they ever diverge, either
the reported numbers stop describing the deployed system or the demo becomes a
mock, and both are silent failures.

`eval/run_eval.py --persist` now also writes `out/models/` (stage1.pkl,
stage2.pt, deploy.json) so the replay runs the exact weights the reported
numbers came from, and `Stage1.save/load` refuses a contract major mismatch the
way the LSTM already did.

### Finding 11 — payload_seq_novelty detected the LOOP, not the attack (10 Sep)

`mimicry` cycled a 27-key script ~3 times per session. A cycling script
necessarily exhausts its own payload transitions, which is exactly what feature
29 measures. **A real BadUSB types its payload once and stops.**

`mimicry_adaptive` is the same profile with the payload typed once. Result:

| | mimicry (looping) | mimicry_adaptive |
|---|---|---|
| seq-novelty, post-onset | 0.848 | 0.992 (benign: 0.996) |
| detection | 100% | **5.0%** (n=20) |
| latency | 27.9 s | 74.7 s |

A per-feature KS scan against benign keyboard windows finds **zero of the 28
deployed features separate it**; the only separable feature is
`log_time_since_enum`, removed as a leak. The 5% is noise.

**Overall detection therefore falls 100% → 84.2%, and that is the correct
number.** Do not remove this profile to restore the headline. The four crude
classes and the looping mimicry are still caught at 100% with 0.0% benign FP,
which is a real and defensible result; the single-shot case is a bound.

**The bound is architectural.** Within single-device URB behaviour there is no
signal — the adversary is statistically the same process as a human typing. The
principled next defence is **cross-device correlation**: a human typing produces
concurrent mouse activity, a BadUSB typing alone does not. That needs the
simulator to generate multi-device host sessions and a feature over concurrent
devices, which the current single-device architecture does not have. Named as
the next step rather than half-built before a review.

Caveat in the other direction: `_adaptive_typing` draws fresh keycodes forever,
so it is the limiting case. A real payload has finite content.

### Audit power (same session)

Adding a sixth attack profile at fixed corpus size dropped per-profile n from
~16 to ~13, and the KS critical value scales as 1/√n — so
`test_audit_catches_a_reintroduced_duration_leak` began passing *because the
audit could no longer fail*. The audit now reports **UNDERPOWERED** when a
profile's critical value exceeds `TARGET_DETECTABLE_D = 0.35`, and says how many
sessions it would need. Silence from a test that could not have failed is not
evidence. At the deployed 600/400 corpus every profile has power.

### Finding 12 — the adaptive adversary was not an adversary (10 Sep)

`mimicry_adaptive` as first written drew a **fresh random keycode every time,
for the whole session** — which is `_human_typing` with a label of 1. It was not
an attack; it was the benign keyboard generator. Nothing could detect it, and
nothing should: a corpus profile that is indistinguishable from benign *by
construction* teaches a model nothing and measures nothing.

A real payload is **finite**. It types a dropper one-liner and the device then
has nothing left to send. `PAYLOAD_KEYS = (60, 220)`; at the human inter-arrival
distribution that takes 20-50 s and leaves the rest of the session silent.

| | benign keyboard | mimicry (loop) | mimicry_adaptive |
|---|---|---|---|
| trailing silence | 0.6 s | 0.8 s | **15.2 s** |
| largest mid-session gap | 3.1 s | 3.1 s | 3.1 s |
| detection | — | 93.3% | **40.0%** (was 5.0%) |

Sustained typing followed by *permanent* silence is a real behavioural
transition and the only signal this profile leaves. Overall detection
84.2% → **89.2%**. This is a realism fix, not a weakening: typing random
characters forever is less like a payload than typing a command once.

**The bound still stands** and is documented rather than simulated: an unbounded
fresh-keycode payload is statistically the benign process, and a per-feature KS
scan separates it on zero of the 28 deployed features.

### Finding 13 — a sweep that laundered noise into a decision

The horizon sweep was re-run after finding 12, on the reasoning that K=10 could
not see a ~15-window silence. On the selection split K=30 won: 86.0% against
K=10's 84.0%.

**On the test split K=30 scored 83.3% against K=10's 89.2%**, and
`mimicry_adaptive` fell 40% → 10%.

The selection margin was 2 points over 50 attack sessions — **one session**. A
detection rate over n sessions cannot resolve differences smaller than 1/n, so
that margin was never evidence. `eval/tune.py` now computes the noise floor,
compares the challenger against the **incumbent** configuration, and recommends
KEEP unless the margin exceeds it. Deployed stays K=10.

Note the discipline: K=10 was restored because the sweep margin was
insignificant — a judgement available from the selection split alone — not
because the test number was better. Choosing on the test result would be exactly
the selection-on-test this harness exists to prevent.

An earlier conclusion also went stale here and is worth remembering: "sequence
horizon is not a lever" was correct when no cross-window signal existed, and
became wrong the moment feature 29 created one. **Re-run sweeps after changing
the feature set.**

### Finding 14 — N2 has no ablation, and when run, does not support N2 (10 Sep)

Four headline pillars, three ablations. N1 had the modality sweep, N3 the
Stage-1-only comparison, N4 the enforcement path — **N2, the per-device adaptive
baseline, had never been isolated.** Pointed, because this session claimed N2 was
"inert", fixed it with `VAR_FLOOR`, and never measured whether it contributes now.

`Stage1` now takes fusion `weights`; `(1.0, 0.0)` is forest-only on the same
score scale (weights renormalise, so the comparison is fair).

| configuration | window AUC | session detection |
|---|---|---|
| full system | 0.9762 | 89.2% |
| forest only, no device baseline | 0.9755 | **90.8%** |
| Stage 1 with baseline | 0.8847 | 15.8% |
| Stage 1, forest only | **0.8876** | 0.0% |

**N2 is not supported.** Removing the device baseline leaves full-system
detection unchanged within noise (2 sessions of 120) and marginally *better*.
Its AUC contribution is negative at both levels — forest-only ranks slightly
higher in both rows. The one place it looks decisive, Stage-1-only going
0.0% → 15.8%, is a **threshold artefact**: with near-identical AUC, the baseline
shifts the score distribution so that a fixed operating point happens to land
better. That is fragile, not a capability.

A plausible explanation, worth stating rather than asserting: the forests are
already **class-conditioned**, so a device's class-specific normality is largely
modelled at population level, leaving the per-device term little to add on a
corpus where devices within a class are near-identical. Real devices vary more,
so N2 may earn its keep on real captures — which is a hypothesis, not a defence.

**Do not rescue this by sweeping the fusion weight.** The measurement is that
the component contributes nothing on this corpus, and tuning the weight until it
appears to would be fitting the conclusion.

**Scorecard as of 10 Sep: N1 unsupported, N2 unsupported, N3 strongly supported
(15.8% → 89.2%), N4 demonstrated.** Two of four pillars lack evidence.

**Superseded on 05 Oct.** Both negative verdicts were reached in the absence of
a purpose-built test rather than by one. Findings 18 and 21 build those tests;
see §1 for the current scorecard. The headline figures in this section still
hold — they regenerate to four decimals.

### Finding 15 — model-family comparison (`make zoo`, 10 Sep)

The project had never justified its algorithm choices against alternatives.
`eval/model_zoo.py` compares them, scored on the **selection split** by default
so the table can drive a decision; `--on test` is exploratory only. Noise floor
on that split is one session = 2.0%.

**Stage 1 — unsupervised, benign-only.** IsolationForest is the right choice:

| model | window AUC |
|---|---|
| **IsolationForest (incumbent)** | **0.8694** |
| EllipticEnvelope | 0.8291 |
| LocalOutlierFactor | 0.7872 |
| OneClassSVM (rbf) | 0.4388 — degenerate, alerts on everything |

**Stage 2 — supervised, sequence input.** The LSTM is *not* justified over much
simpler alternatives, though it is not wrong either:

| model | AUC | detect | train |
|---|---|---|---|
| LSTM (incumbent) | 0.9798 | 88.0% | 6.8 s |
| GRU | 0.9790 | 88.0% | 5.5 s |
| 1-D CNN (no recurrence) | 0.9753 | 88.0% | 37.8 s |
| HistGradientBoosting (flat sequence) | 0.9770 | 86.0% | **1.7 s** |
| LogisticRegression (flat sequence) | 0.9271 | 72.0% | 2.4 s |

GRU and the CNN tie the LSTM exactly; gradient boosting is one session behind at
a quarter of the training cost. All four are within noise. **Recurrence is not
doing the work.** Incumbent kept because no challenger clears the noise floor —
the same rule as finding 13, not a preference.

**Architecture control — one supervised model, one window, no pipeline:**
HistGradientBoosting 80.0%, RandomForest 76.0%, against 88.0% for the sequence
models. So **temporal context is worth ~8 points and N3 survives** — but N3 is
about *sequence*, not about *recurrence*, and the report should say so.

**The cost of the labelling constraint.** A supervised single-window model
reaches 80.0%; our unsupervised Stage 1 reaches 0.0% at the deployed threshold
(AUC 0.87). That gap is what training on benign traffic alone costs, and it is
the strongest argument for the two-stage design: Stage 2 recovers it using
labels that Stage 1 is not allowed to see.

### Finding 16 — the corpus cache had no simulator fingerprint

`load_corpus` checked the contract version but not the code that generated the
windows, so changing the simulator and forgetting to rebuild silently evaluated
the *previous* adversary. This happened during development. The cache now stores
a hash of `capture/simulator.py` plus `features/extractor.py` and refuses to
load against different code.

### Finding 17 — the single-modality ablation was scoring a known leak (05 Oct)

`modality_indices("identity")` still contains `log_time_since_enum`. Finding 8
removed that feature from the **deployed** model as a systematic label leak;
every other configuration in `run_eval.py` excluded it, and the modality
ablation silently did not. So the reported identity-modality row — one of the
three N1 ablations — was measuring the leak rather than the modality.

Caught sideways: a new diagnostic profile whose identity features are
*byte-identical* to benign keyboards was being detected by the identity
modality at 27%, which is structurally impossible. With the leak excluded it
correctly drops to 0%.

The correction to the reported table is material:

| identity modality | features | AUC | detect | benign FP |
|---|---|---|---|---|
| as reported | 6 (incl. leak) | 0.8884 | 39.2% | — |
| corrected | 5 | 0.8472 | 100.0% | **42.1%** |

With the leak gone the identity modality is **degenerate**: it flags 42% of
benign sessions, so its 100% detection is alarming rather than detecting. Its
previously reported 39.2% was substantially the leak doing the work. Timing,
payload and protocol are unchanged to four decimals, which is the check that the
fix was surgical.

Fixed by centralising the rule in `pipeline.deployed_features()`, which also
drops **zero-variance columns on the training split**. Three eval scripts had
drifted from the same rule; one place now owns it. The guard matters beyond
this bug — `frac_ctrl` was a constant-zero column for all of Sprint 0, and
contract 3.0.0's `concurrent_device_rate` is identically zero on any
single-device corpus.

### Finding 18 — N2 is refuted, and finding 14's explanation is wrong (05 Oct)

Finding 14 measured the per-device baseline as inert and offered a reason: every
device of a class draws timing from one shared parameter set, so a
class-conditioned forest already models them all. It flagged this explicitly as
"a hypothesis, not a defence", and noted N2 "may earn its keep on real
captures".

The hypothesis names a property of the **simulator**, so it is testable without
real captures. `Simulator(device_variation=True)` gives each device a persona
drawn from a hyperprior (typing speed, burst length, pause tendency). Benign and
malicious draw from the same prior — if only benign devices varied, "behaves
like the population mean" would itself become a tell, and the mimicry profiles
exist precisely to be timing-indistinguishable.

| corpus | kbd rate sd | full | forest-only | baseline contributes |
|---|---|---|---|---|
| homogeneous | 0.173 | 89.2% | 90.0% | −0.8% (noise 0.8%) |
| heterogeneous | 0.485 | 85.8% | 85.0% | +0.8% (noise 0.8%) |

The manipulation worked — between-device spread rose 2.8× — and the baseline is
inert on **both**. +0.8% is one session of 120, exactly the noise floor. So
device homogeneity was not the reason, and **N2's negative result is stronger
than previously reported rather than explained away.** `make baseline-ablation`.

Detection also falls 89.2% → 85.8% under heterogeneity. Realistic device
variation makes the problem harder; that is the correct direction, and the
heterogeneous number is the more honest one to quote.

### Finding 19 — subtlety needs benign variance (05 Oct)

Building a *fair* test for N1 needed attack families that deviate on one axis at
a time (`AXIS_PROFILES`), since every profile in `ATTACK_PROFILES` deviates on
several at once and so any single modality catches all of them. Three attempts
to build a **subtle** protocol deviation all failed the same way, and the reason
generalises:

A deviation can only be made small in a feature where benign traffic has
variance. Against a keyboard-only population `n_endpoints` is exactly 1,
`frac_ctrl` exactly 0 after enumeration, `frac_intr`/`frac_in` exactly 1. Any
non-zero value is perfectly separable from a constant, so a protocol deviation
is either invisible or caught at 100% — never marginal. Both a second endpoint
and extra GET_REPORT control transfers were detected by the protocol modality
alone at 100% even at 5% strength.

Same shape as finding 10, one level up. `axis_diffuse` therefore deviates on
timing and payload only, which are the two modalities with real benign variance
for a keyboard — and that restriction is a finding about the feature set, not a
limitation of the experiment.

### Finding 20 — cross-device correlation does not clear the single-shot bound (05 Oct)

Finding 11 named the defence: "a human typing produces concurrent mouse
activity, a BadUSB typing alone does not." Contract **3.0.0** appends
`concurrent_device_rate` (log1p event rate of every other device on the host,
supplied through `HostContext`), `Simulator.host_session` generates
multi-device hosts, and `make cross-device` measures it. One corpus, scored
twice, differing only in that column.

The leakage guard was built first, because this feature is the most obvious
candidate in the project to become the next `log_time_since_enum`.
`P_COMPANION` is applied identically to benign and attacked hosts, so both a
lone benign keyboard and a BadUSB beside a working user occur. Measured at 500
hosts: companion prevalence gap 1.5%, and KS of the feature across benign vs
pre-onset attack *keyboard* windows D=0.027 against crit=0.033. Guard holds.

| profile | without | with | delta | noise |
|---|---|---|---|---|
| mimicry_adaptive | 28.0% ±8.4 | 25.9% ±3.9 | **−2.1% ±6.2** | 4.5% |
| mimicry | 93.2% ±5.6 | 94.7% ±3.8 | +1.5% ±2.1 | 4.9% |
| overall | — | — | −0.0% ±1.3 | — |

**It does nothing.** A single-seed run first showed +4.2% on
`mimicry_adaptive`, which is one session of 24 — exactly the noise floor, made
to look like a win by floating-point `>`. Under three seeds it reverses. Same
lesson as finding 13; the seeded harness now refuses a margin smaller than its
own spread.

**The diagnosis is specific and matters more than the number.**
`host_session` generates each device *independently* — the companion mouse is
drawn without reference to what the keyboard is doing. So in this corpus a
benign keyboard and its mouse are statistically independent, exactly as a
BadUSB and a bystander's mouse are. The coupling the feature exists to detect
is absent on **both** sides, so no feature over these streams could separate
them. The negative result is therefore about the corpus, not about the idea.

What a real human produces, and this corpus does not, is temporal interleaving:
one agent alternates between typing and mousing, so the streams are
anti-correlated at sub-second scale, while two independent agents overlap at the
chance rate. Testing that needs a *coupled* benign generator and a simultaneity
feature rather than a rate.

**Why that was not simply built.** Modelling the coupling in the simulator and
then detecting it is circular — the same objection finding 11's
`payload_seq_novelty` had to answer, but weaker here, because keyboard/mouse
interleaving parameters cannot be defended from first principles the way "a
payload must type specific commands" can. This is the point at which the
question needs real traffic, and it is named as such rather than half-built.

### Finding 21 — N1 gets a fair test at last; fusion helps non-uniformly (05 Oct)

Every prior N1 measurement was confounded: each profile in `ATTACK_PROFILES`
deviates on several modalities at once, so any one modality catches all of them
and fusion cannot show a gain. `AXIS_PROFILES` fixes that with four single-axis
families plus `axis_diffuse`, which deviates weakly on timing and payload
together. `make fusion-ablation`. Axis purity is **measured** by per-feature KS
rather than assumed, and printed before the detection table.

Two conditions, pre-registered before the first run.

**(a) Does fusion lose what a specialist sees? No — PASS.**

| axis | fused | best specialist |
|---|---|---|
| timing | 100% | 100% (timing) |
| payload | 100% | 100% (payload) |
| protocol | 100% | 100% (payload) |
| identity | **100%** | **3.7%** |

The identity row is the strongest single argument for fusion in the project.
With `log_time_since_enum` correctly excluded (finding 17) the identity modality
has 5 near-constant features, cannot calibrate a threshold, and goes degenerate
at 100% benign FP. Only the fused model detects a descriptor-only attack.

**(b) Does fusion beat every specialist on the diffuse attack? No — FAIL.**
Strength swept, three seeds per point, every point reported:

| strength | fused | best specialist | margin | verdict |
|---|---|---|---|---|
| 0.05 | 24.7% ±10.6 | 35.8% ±7.6 | **−11.1% ±3.0** | fusion LOSES |
| 0.10 | 33.3% ±6.0 | 30.9% ±6.3 | +2.5% ±1.7 | tie |
| 0.20 | 32.1% ±14.3 | 49.4% ±13.6 | −17.3% ±27.4 | tie |
| 0.33 | 80.2% ±6.3 | 60.5% ±6.3 | **+19.8% ±3.5** | fusion WINS |
| 0.50 | 100% | 100% | +0.0% | saturated, invalid |

**Fusion's benefit is not uniform, and that is the finding.** At moderate
strength the complementary evidence is worth 19.8 points. At the subtlest
strength the 28-feature model is *beaten by a 9-feature subset of itself* by
11.1 points, consistently across seeds — with a weak signal the extra dimensions
cost more variance than they contribute. By the pre-registered criterion (win
somewhere, never lose) **N1 is not supported**, and the criterion stands as
written.

The nuance is worth more than the verdict: report the curve, not the binary.
Fusion is the only configuration that covers all four axes, and it is the wrong
choice against a maximally subtle single-channel adversary.

Two methodological notes, both earned the hard way:
* An unseeded version of this sweep flipped between SUPPORTED and NOT SUPPORTED
  depending on which strengths were sampled. A margin smaller than its own
  seed spread is now refused outright.
* An earlier run reported SUPPORTED while the fused configuration sat at 7.4%
  benign FP against the specialists' 0.0% — it detected more because it alarmed
  more. The benign pool was too small to place a 1e-4 quantile. Every
  configuration is now gated on benign FP before its detection counts, fused
  included.

### Finding 22 — `make eval` did not reproduce the headline (05 Oct)

CLAUDE.md documents `make eval` as the command that regenerates every reported
number. It did not. `run_eval` defaulted to `--n-benign 210 --n-attack 140`
(350 sessions) while the deployed corpus is 600/400 (1000 sessions), and it
**read** `out/corpus_cache.npz` without ever **writing** it — the cache was
produced by `eval/tune.py` alone. So the documented command reproduced the
headline only while a cache built by a different script happened to be sitting
on disk, and silently reported numbers from a corpus a third of the size once it
was not. Found by deleting the cache.

Three fixes: defaults are now 600/400 so the script and the documentation agree;
`run_eval` writes the cache it was given; and the cache is gitignored, because a
tracked 2.6 MB binary tied to both `CONTRACT_VERSION` and the simulator
fingerprint is a stale artefact waiting to be loaded.

Worth noting what did *not* go wrong: rebuilding the corpus from scratch under
contract 3.0.0 reproduced the headline to four decimals — 89.2% detection, 0.0%
benign FP, AUC 0.9762, identical per-class rates. That is the check that the
axis profiles, device personas and host-session machinery left the default
generator paths untouched.

### Finding 23 — raw URB trace storage, pseudonymised (06 Oct)

Plan §5: "store raw traces, not just features, so the feature set can be
revised without re-collecting." The store held only features and scores.
`storage/traces.py` adds the trace, and its acceptance test is the plan's own
purpose: a stored session re-featurises **bit for bit** to the vectors the live
monitor actually scored — asserted against the real `LiveMonitor` across five
concurrent devices, including mimicry (whose `payload_seq_novelty` needs the
whole session), bulk storage and a mid-session interface. Verified again from an
installed wheel: a 57 s capture's stored trace re-featurises to the same 58
windows the monitor scored, and replaying it through `--replay-db` produces
identical Stage-1 and Stage-2 scores.

**A USB trace is a keylogger**, so payloads are pseudonymised by default. The
extractor reads payload bytes only through `e.length`, `if e.data`, and
equality, so an injective, length-preserving keyed permutation keeps every
feature exact. It is a Feistel network (keyed table for 1-byte payloads) because
a truncated HMAC fails both ways: full-length it breaks `len(data) <= length`,
so stored events would no longer validate; truncated to the payload length it
collides on 4-byte mouse reports within a long session and silently changes
`payload_novelty`. The key is drawn per run and never written. **Not anonymity**
— a consistent per-session mapping is a substitution cipher, so long typing is
open to frequency analysis; the docstring and DEPLOYMENT.md say so. `none` drops
payloads, `full` keeps them for consented collection only.

Retention is bounded, oldest first; an escalation pins the URBs from the ten
windows the model saw, and per-session completeness is tracked rather than
assumed. `StoreSource` makes a stored run a fifth capture backend.

### Finding 24 — three live-monitor bugs, found by building exact replay (06 Oct)

None was visible to the existing tests, and each was fixed test-first.

1. **Deferred events produced negative inter-arrival times.** A device seen
   before sysfs lists it has its events held and folded into its first scored
   window — appended *after* that window's own events, and in reverse window
   order across repeated deferrals. The extractor diffs timestamps as given:
   `min_iat` −1.2 s on a six-event input. A real hotplug's enumeration URBs
   precede its sysfs entry, so this corrupted the first scored window of every
   new device, which is the window a BadUSB starts typing in. The existing test
   checked only that the device was eventually attached.
2. **`run()` never returned for a finite source.** End-of-stream was reported
   only by the drain that dequeued the sentinel; if later windows still held
   buffered events then, every following drain called the source live again.
   So `live_monitor.py --replay` could not exit by itself. Every existing test
   passed `max_windows`. A 57 s replay now exits after 58 windows.
3. **`run()` ignored `stop()`.** Stopping relied on the reader thread noticing,
   and on an idle live bus that thread is blocked in a debugfs read. Ctrl-C, and
   systemd's SIGTERM, did not stop the monitor; systemd would SIGKILL it.

No reported number changes: the evaluation never runs through `LiveMonitor`.

### Finding 25 — the deployment did not deploy (06 Oct)

Packaging the system for a host turned up five defects, each of which would
have failed silently or badly on a real install:

* **The live monitor persisted nothing.** It printed to a terminal; the store
  the dashboard reads was only ever filled by a batch evaluation. `live/persist.py`
  now writes windows (label NULL — live traffic has no ground truth), alerts and
  actions on escalation, and trust state.
* **A fresh clone ran Stage 1 alone, silently.** A blanket `*.pt` gitignore rule
  excluded the 36 KB `stage2.pt` while the 9.6 MB `stage1.pkl` beside it was
  tracked, and the monitor loaded Stage 2 only "if it exists" — so every clone
  ran the configuration measured at 15.8% while presenting itself as the
  deployed one. The checkpoint is now tracked and the monitor refuses to run
  without it unless told `--allow-stage1-only`.
* **`iudbms-capture` pointed at a module that has never existed.** Any install
  created a command that crashed on launch. Replaced by `iudbms-monitor` and
  `iudbms-api`, which the systemd units call.
* **The dashboard 404'd once installed.** It was found via `parents[3]` of
  `app.py` — the repo root from a checkout, nothing in site-packages. It is
  package data now.
* **CORS was `allow_origins=["*"]` on an API with state-changing endpoints.** Any
  page the operator visited could POST `/devices/{id}/authorize`. The dashboard
  is same-origin and needs none; cross-origin access is now opt-in, and a test
  confirms a `text/plain` cross-site POST (which needs no preflight) is refused.

`deploy/` holds two hardened systemd units (monitor as root in group `iudbms`;
API unprivileged, loopback only), the env file, and install/verify/uninstall
scripts; `docs/DEPLOYMENT.md` documents them. **Verified** from a wheel installed
into a clean virtualenv outside the repo: both entry points, persistence, the
pseudonymised trace, every API endpoint, no CORS grant. **Not verified until the
VM:** the units themselves, the setgid state directory shared between root and
`iudbms`, SIGTERM under systemd, and live debugfs capture.

The monitor's dev default database is `out/live.db`, not `out/iudbms.db`:
`make eval` deletes the latter before rewriting it, and would have wiped live
data and mixed live devices with evaluation sessions on the dashboard.

### Finding 26 — first real kernel traffic: the parser holds, the monitor had two bugs (06 Oct)

**The first real usbmon output this project has parsed.** A UTM/QEMU guest (Ubuntu
24.04.5 aarch64, kernel 6.8.0-146) has `usbmon` as a module, debugfs mounted and a
real xHCI controller with a QEMU keyboard, mouse, tablet, hub and USB CD-ROM.
`scripts/vm_capture.sh` recorded 20 s while `lsusb -v` read every descriptor
(control transfers) and the kernel polled the empty CD-ROM (SCSI bulk-only
transport — the payloads start `USBC`/`USBS`). Committed as
`tests/fixtures/usbmon/real_qemu_vm_{capture.txt,descriptors.json}`;
`tests/test_real_capture.py` pins what follows.

**What held.** 338 of 338 lines parsed — **0.0000% loss against the plan's < 0.1%
criterion, which had never been measurable.** 0 of 338 events fail
`URBEvent.validate()`; all 108 control submissions carry their SETUP packet;
timestamps are monotone; the polling interval is its own field rather than the
length. Those were the line forms and the length defect the Sprint-0 parser got
wrong, and the parser handles them on genuine kernel output.

**What real traffic found that the simulator could not — a device with no
descriptor.** `SysfsScanner` skips root hubs on purpose (`usbN`), and the kernel
emits traffic for devnum 1 on every bus: 158 of the 338 events. The monitor holds
a not-yet-enumerated device's events and retries — and re-held them *appended* to
each window's new events, so the held list grew by that device's traffic every
window, without bound. Every simulated device has a descriptor, so nothing
exercised it. Fixed with `max_defer_windows` (30): held events are counted in
`events_undescribed` and dropped, still present in the raw trace which records at
drain. A second effect of the same cause: an 18 s capture took **34 windows** to
replay, because held events kept `run()` alive for the whole cap after the source
ended. Once nothing more can arrive there is nothing to wait for, so held events
now get `end_grace_windows` (3) more windows.

**What this does and does not show.** All five described devices scored, zero
alerts, Stage 2 between 0.23 and 0.46 against a 0.9916 threshold. That is **not**
a false-positive rate: the traffic is 34 s of mostly idle devices plus an
enumeration-style descriptor read, from *virtual* devices, with nobody typing. It
shows the capture→parse→window→score→persist→trace path runs end to end on real
kernel text. It says nothing about real device behaviour, and the models were
trained on simulated traffic so their scores here are out of distribution.

### Finding 27 — the install script's PyTorch advice was wrong, and the default wheel is a trap (06 Oct)

`deploy/install.sh` said `--cpu-torch` was "unnecessary on ARM64, where the default
wheel is CPU-only". That was true when written and is not now. PyPI's torch 2.14.1
declares `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
`nvidia-nvshmem-cu13` and `triton` for **every** `platform_system == "Linux"` —
ARM64 included. The aarch64 wheel alone is 454 MB, plus several GB of NVIDIA
libraries, for a 36 KB LSTM that only ever runs on CPU. Found on the first real
install: ~150 KB/s made it a multi-hour download. I stopped it, and the CPU-only
index serves the same torch 2.14.1 for aarch64 at 159 MB.

CPU-only is now the default on every architecture (`--default-torch` opts out), and
it is installed first from the CPU index so the project install sees `torch>=2.2`
already satisfied and does not replace it. `docs/DEPLOYMENT.md` says the same for
anyone installing by hand. **Lesson:** a dependency claim about someone else's
packaging is a claim about a moving target; I verified neither the wheel's metadata
nor its size when I wrote it.

### Finding 28 — the live service scored nothing; found only by running it (06 Oct)

The first systemd service on a real kernel passed `verify.sh` on every structural
check and **scored nothing**: heartbeat `devices=0 late=174 urb_recorded=174` —
every event late. usbmon timestamps are relative to the *first event the parser
sees*, but the window clock started when the *monitor* started. If the first URB
arrives after startup the stream sits permanently behind the clock (both advance
at one second per second, so it never catches up) and the whole thing is dropped
as late. A foreground run had worked only because its first URB arrived within a
fraction of a second. Every simulated or replayed source delivers its first event
immediately, so nothing could have seen it, and 383 tests passed throughout.

Fixed by anchoring the window clock to the first event's arrival
(`origin = now - first_event.ts`), so the offset is zero by construction however
long the bus was idle first. After: `devices=1 late=0`, windows persisted every
second from live kernel output. The monitor now waits for the first event rather
than emitting empty windows, so an idle bus produces no windows and no heartbeat.

**What a green `verify.sh` meant:** the install is structurally sound. It did not
mean the system worked. The check that found this was reading the heartbeat. The
script now also checks that windows advance; it did warn about exactly this and the
warning was the real signal.

**Library versions differ and scores do not.** The VM resolved scikit-learn 1.9.1,
torch 2.14.1, numpy 2.5.3; the models and `requirements-lock.txt` are 1.6.1 /
2.11.0 / 1.26.4. scikit-learn warns that unpickling across versions is
unsupported. The same real capture replayed on the Mac and on the VM gives
**identical Stage-1 and Stage-2 scores to six decimals on all 83 common windows**
(max |Δ| = 0.000000). So nothing is corrupted here, but it is an unsupported path
that happens to hold for this model and these inputs, `install.sh` does not use the
lock file, and a different sklearn could change that. Not fixed; recorded.

Smaller: `verify.sh` *sourced* the env file as shell, so `IUDBMS_MONITOR_ARGS=--bus 0`
ran `0` as a command (it is systemd syntax, not shell — now read as data, and the
value is quoted); and an unreadable models directory gave a raw `PermissionError`
traceback instead of a message.

### Live monitoring — wired 10 Sep, never yet run on real traffic

Until now **nothing called `UsbmonSource.live()`.** It existed and was
referenced only in its own docstring; `live_replay.py` read a recorded file or
a simulated session. The streaming detector was correct and had no live feed,
so "live monitoring" was aspirational.

`iudbms.live.LiveMonitor` closes it: a reader thread drains an `EventSource`
into a queue while the main loop wakes on window boundaries, discovers devices
via `SysfsScanner` as they enumerate, scores each device's window when it
closes, and detaches a device after a long silence so the next device on the
same bus address cannot inherit its payload history.

**Windows are cut on the kernel's timestamps, not on arrival time.** The reader
thread has scheduling jitter and the queue adds latency, so bucketing by
dequeue order would smear traffic across windows and make the same capture
score differently under load. Events arriving after their window has closed are
counted (`events_late`) rather than folded into a later window, which would
distort that window's rate.

`scripts/live_monitor.py` is the only entry point that touches live traffic;
`make monitor`. It refuses to run off Linux with a pointer to the replay script.

**Still untested against real traffic** — the monitor is exercised by nine unit
tests against a fake source, which covers windowing, discovery, deferral,
detach/re-attach and error propagation, but not debugfs itself. That needs the
VM, and it is now the same 30 minutes that the capture always was: run
`make monitor` there and it monitors for real.

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
