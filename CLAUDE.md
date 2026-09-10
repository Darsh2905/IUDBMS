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
Run the dashboard with `uvicorn iudbms.api.app:app` and open http://localhost:8000.
`iudbms.live.StreamingDetector` is the long-lived detector a deployment runs:
per-device extractor, baseline, sequence window and gate, advanced one window at
a time. `scripts/live_replay.py` drives it at wall-clock speed
(`make demo` / `make demo-hard`).
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
| IUDBMS (S1+S2), 28 of 29 features | 0.9945 | 0.929 | **100.0%** |
| Stage 1 only | 0.9363 | 0.094 | 11.7% |
| identity whitelist | — | — | 0.0% |

Benign session FP **0.6%** (1 of 178 benign test sessions). Median latency 3.6 s,
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

`dashboard/index.html`, served by FastAPI at `/`. **React from cdnjs, no build
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
