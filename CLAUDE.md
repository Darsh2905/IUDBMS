# IUDBMS — working notes for Claude Code

Read this before touching code. It records why things are shaped the way they
are, what is already known to be broken, and what has been decided so future
sessions do not re-litigate it.

Specification: `docs/IUDBMS_Implementation_Plan.md` (authoritative for scope,
schedule and acceptance criteria). This file is authoritative for *decisions
taken since* the plan was written.

---

## 1. Current state

Milestone 1 (repo skeleton + frozen data contract) is done. Milestones 2–10 are
not started. `prototype/` is the Sprint-0 vertical slice: it runs end to end and
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

1. **The parser silently drops four real `usbmon` line forms.** Tested against
   `Documentation/usb/usbmon.rst`: control submissions with a setup packet
   (`s 80 06 ...`), *every* submission line (`<` data tag), every line where
   the payload was not captured (`-` tag), and iso completions carrying
   `status:interval:start_frame:errcount`. Dropped with no counter, so the
   plan's "< 0.1 % loss" criterion could not be evaluated at all.
2. **`length` read the polling interval.** Given `0:4` with a true data length
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
