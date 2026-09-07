#!/usr/bin/env python3
"""
IUDBMS end-to-end prototype run.

    python3 run_demo.py

Pipeline: simulate/capture -> usbmon parse -> multi-modal features ->
Stage 1 (adaptive baseline + isolation forest) -> Stage 2 (LSTM) ->
closed-loop enforcement -> SQLite + metrics + figures + report.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from iudbms.simulator import Simulator, ATTACK_PROFILES
from iudbms.usbmon_parser import parse
from iudbms.features import sessionise, FEATURE_NAMES
from iudbms.stage1 import Stage1, DeviceBaseline, class_group
from iudbms.stage2_lstm import LSTMClassifier, build_sequences, SEQ_LEN
from iudbms.storage import Store
from iudbms.enforcement import PolicyEngine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEED = 7
TARGET_FPR = 0.001          # per-window; session-level FP compounds over ~70 windows
rng = np.random.default_rng(SEED)


def banner(msg):
    print(f"\n{'=' * 68}\n{msg}\n{'=' * 68}", flush=True)


# ---------------------------------------------------------------- 1. capture
banner("[1/7] CAPTURE  simulating device sessions -> usbmon text")
t0 = time.time()
sim = Simulator(seed=SEED)
os.makedirs(OUT, exist_ok=True)

# Each session is generated, parsed and featurised in turn, and its raw usbmon
# text is released immediately. A live deployment behaves the same way (bounded
# ring buffer drained by the feature extractor), so peak memory stays flat
# instead of growing with the number of URBs observed.
sessions, data = [], {}
total_lines = total_events = 0
first_line, sample_written = None, False

for sid, prof in sim.plan(n_benign=210, n_attack=140):
    s = sim.session(sid, prof)
    total_lines += len(s.lines)
    if first_line is None:
        first_line = s.lines[0]
    if not sample_written and s.profile == "staged_delay":
        with open(os.path.join(OUT, "sample_usbmon_capture.txt"), "w") as fh:
            fh.write(f"# device={s.descriptor['bus']}-{s.descriptor['devnum']} "
                     f"vid=0x{s.descriptor['vid']:04x} pid=0x{s.descriptor['pid']:04x} "
                     f"profile={s.profile} onset={s.onset:.1f}s\n")
            fh.write("\n".join(s.lines[:40]) + "\n")
        sample_written = True
    ev = parse(s.lines)
    total_events += len(ev)
    X, T, y = sessionise(s, ev)
    s.lines = []                       # release raw capture
    del ev
    data[s.sid] = (X, T, y)
    sessions.append(s)

rng.shuffle(sessions)
print(f"  sessions        : {len(sessions)}  "
      f"({sum(1 for s in sessions if s.label == 0)} benign / "
      f"{sum(1 for s in sessions if s.label == 1)} malicious)")
print(f"  usbmon lines    : {total_lines:,}")
print(f"  example line    : {first_line[:78]}")

# ---------------------------------------------------- 2. parse + features
banner("[2/7] PARSE + FEATURES  usbmon -> 28-dim multi-modal windows")
n_windows = sum(len(v[0]) for v in data.values())
print(f"  URB events parsed: {total_events:,}")
print(f"  windows (1.0 s)  : {n_windows:,}   features/window: {len(FEATURE_NAMES)}")
print(f"  modalities       : identity | timing | payload | protocol")

# ---------------------------------------------------------------- 3. split
by_profile = defaultdict(list)
for s in sessions:
    by_profile[s.profile].append(s)
train, test = [], []
for prof, group in by_profile.items():
    idx = rng.permutation(len(group))
    cut = int(0.7 * len(group))
    train += [group[i] for i in idx[:cut]]
    test += [group[i] for i in idx[cut:]]
fit_idx = rng.permutation(len(train))
val = [train[i] for i in fit_idx[int(0.8 * len(train)):]]
fit = [train[i] for i in fit_idx[:int(0.8 * len(train))]]
print(f"  split            : fit={len(fit)}  val={len(val)}  test={len(test)} sessions")

# -------------------------------------------------------------- 4. stage 1
banner("[3/7] STAGE 1  per-device adaptive baselines + isolation forest")
Xb = np.vstack([data[s.sid][0] for s in fit if s.label == 0])
Gb = [class_group(s.descriptor) for s in fit if s.label == 0
      for _ in range(len(data[s.sid][0]))]
s1model = Stage1(seed=SEED).fit(Xb, Gb)
print(f"  benign windows   : {len(Xb):,}   class-conditioned forests: "
      f"{sorted(s1model.forests)}")

dev_raw_all = []
for s in fit:
    if s.label == 0:
        _, _, dr = s1model.score_session(data[s.sid][0], s.descriptor, DeviceBaseline())
        dev_raw_all.append(dr)
s1model.calibrate_dev(np.concatenate(dev_raw_all))
print(f"  calibration (p99): iforest={s1model.scale['if']:.3f}  "
      f"device-dev={s1model.scale['dev']:.3f}")

S1, EXPL, BASELINES = {}, {}, {}
for s in sessions:
    bl = DeviceBaseline()
    sc, ex_, _ = s1model.score_session(data[s.sid][0], s.descriptor, bl)
    S1[s.sid], EXPL[s.sid], BASELINES[s.sid] = sc, ex_, bl
mal = np.concatenate([S1[s.sid][data[s.sid][2] == 1] for s in test if s.label == 1])
ben = np.concatenate([S1[s.sid][data[s.sid][2] == 0] for s in test])
print(f"  mean score       : benign={ben.mean():.3f}   post-onset attack={mal.mean():.3f}")

# -------------------------------------------------------------- 5. stage 2
banner("[4/7] STAGE 2  LSTM over 10-window sequences (NumPy BPTT)")


def seqset(slist):
    Xs, ys, meta = [], [], []
    for s in slist:
        X, T, y = data[s.sid]
        aug = np.hstack([X, S1[s.sid].reshape(-1, 1)])
        Xs.append(build_sequences(aug))
        ys.append(y)
        meta += [(s.sid, t) for t in T]
    return np.concatenate(Xs), np.concatenate(ys), meta


Xtr, ytr, _ = seqset(fit)
Xva, yva, _ = seqset(val)
print(f"  sequences        : train={Xtr.shape}  val={Xva.shape}  (T={SEQ_LEN})")
lstm = LSTMClassifier(d_in=Xtr.shape[2], hidden=32, seed=SEED)
lstm.fit(Xtr, ytr.astype(float), epochs=14, batch=128, lr=0.02, val=(Xva, yva))

S2 = {}
for s in sessions:
    X, T, y = data[s.sid]
    aug = np.hstack([X, S1[s.sid].reshape(-1, 1)])
    S2[s.sid] = lstm.predict_proba(build_sequences(aug))

# threshold learned from validation benign windows at target FPR
val_benign = np.concatenate([S2[s.sid][data[s.sid][2] == 0] for s in val])
THRESH = float(np.quantile(val_benign, 1 - TARGET_FPR))
val_benign_s1 = np.concatenate([S1[s.sid][data[s.sid][2] == 0] for s in val])
THRESH_S1 = float(np.quantile(val_benign_s1, 1 - TARGET_FPR))
print(f"  learned threshold: stage2={THRESH:.4f}  (stage1-only ablation={THRESH_S1:.4f})"
      f"  @ FPR<={TARGET_FPR:.1%}")

# ------------------------------------------------------------ 6. evaluation
banner("[5/7] EVALUATION  window-level and session-level, on held-out test")
yte = np.concatenate([data[s.sid][2] for s in test])
p1 = np.concatenate([S1[s.sid] for s in test])
p2 = np.concatenate([S2[s.sid] for s in test])
auc1, auc2 = roc_auc_score(yte, p1), roc_auc_score(yte, p2)
pr, rc, f1, _ = precision_recall_fscore_support(yte, (p2 >= THRESH).astype(int),
                                                average="binary", zero_division=0)
pr1, rc1, f11, _ = precision_recall_fscore_support(yte, (p1 >= THRESH_S1).astype(int),
                                                   average="binary", zero_division=0)
print(f"  window ROC-AUC   : stage1={auc1:.4f}   stage1+stage2={auc2:.4f}")
print(f"  window P/R/F1    : stage2 {pr:.3f}/{rc:.3f}/{f1:.3f}"
      f"    stage1-only {pr1:.3f}/{rc1:.3f}/{f11:.3f}")


def run_policy(slist, scores, thresh, store=None):
    eng = PolicyEngine(threshold=thresh, k=4, n=6)
    res = {}
    for s in slist:
        X, T, y = data[s.sid]
        ctx = eng.new_device()
        first, acts = None, []
        for i, t in enumerate(T):
            a = eng.step(ctx, s.sid, t, S1[s.sid][i], scores[s.sid][i],
                         EXPL[s.sid][i], s.descriptor)
            if a:
                acts.append(a)
                if first is None:
                    first = t + 1.0
                if store is not None:
                    aid = store.add_alert(s.sid, t, S1[s.sid][i], scores[s.sid][i],
                                          a["severity"], EXPL[s.sid][i])
                    store.add_action(aid, s.sid, t, a["action"], a["rule"])
        if store is not None:
            store.set_trust(s.sid, ctx["state"])
        res[s.sid] = {"first": first, "actions": acts, "state": ctx["state"]}
    return res


pol2 = run_policy(test, S2, THRESH)
pol1 = run_policy(test, S1, THRESH_S1)

# identity-only baseline (USBGuard-style VID:PID whitelist learned from fit set)
whitelist = {(s.descriptor["vid"], s.descriptor["pid"]) for s in fit if s.label == 0}
id_detect = {s.sid: (s.descriptor["vid"], s.descriptor["pid"]) not in whitelist for s in test}


def summarise(pol, slist):
    out = {}
    per = defaultdict(lambda: [0, 0, []])
    for s in slist:
        hit = pol[s.sid]["first"] is not None
        per[s.profile][1] += 1
        per[s.profile][0] += int(hit)
        if hit and s.label == 1:
            per[s.profile][2].append(pol[s.sid]["first"] - s.onset)
    atk = [s for s in slist if s.label == 1]
    ben = [s for s in slist if s.label == 0]
    out["attack_detect"] = sum(pol[s.sid]["first"] is not None for s in atk) / len(atk)
    out["benign_fp"] = sum(pol[s.sid]["first"] is not None for s in ben) / len(ben)
    lat = [pol[s.sid]["first"] - s.onset for s in atk if pol[s.sid]["first"] is not None]
    out["median_latency"] = float(np.median(lat)) if lat else float("nan")
    out["p90_latency"] = float(np.percentile(lat, 90)) if lat else float("nan")
    out["revoked"] = sum(pol[s.sid]["state"] == "revoked" for s in atk) / len(atk)
    out["per_profile"] = {k: (v[0] / v[1], v[1],
                              float(np.median(v[2])) if v[2] else None) for k, v in per.items()}
    ben_hours = sum(s.duration for s in ben) / 3600
    out["fp_per_hour"] = sum(pol[s.sid]["first"] is not None for s in ben) / max(ben_hours, 1e-9)
    return out


m2, m1 = summarise(pol2, test), summarise(pol1, test)
id_atk = [s for s in test if s.label == 1]
id_rate = sum(id_detect[s.sid] for s in id_atk) / len(id_atk)
print(f"  session detection: IUDBMS={m2['attack_detect']:.1%}   "
      f"stage1-only={m1['attack_detect']:.1%}   identity-whitelist={id_rate:.1%}")
print(f"  benign FP rate   : {m2['benign_fp']:.1%} of sessions "
      f"({m2['fp_per_hour']:.2f} alerts/benign-hour)")
print(f"  detect latency   : median={m2['median_latency']:.1f}s  p90={m2['p90_latency']:.1f}s")
for p in ATTACK_PROFILES:
    r2 = m2["per_profile"].get(p)
    r1 = m1["per_profile"].get(p)
    if r2:
        print(f"    {p:16s} n={r2[1]:2d}  IUDBMS={r2[0]:6.1%}  stage1-only={r1[0]:6.1%}  "
              f"latency={r2[2]:.1f}s")

# --------------------------------------------------------------- 7. persist
banner("[6/7] PERSIST + ENFORCE  SQLite store and closed-loop actions")
db_path = os.path.join(OUT, "iudbms.db")
if os.path.exists(db_path):
    os.remove(db_path)
store = Store(db_path)
for s in sessions:
    store.add_device(s.sid, s.descriptor, class_group(s.descriptor))
    X, T, y = data[s.sid]
    store.add_windows(s.sid, T, S1[s.sid], S2[s.sid], y, X, FEATURE_NAMES)
    store.add_baseline(s.sid, FEATURE_NAMES, BASELINES[s.sid])
run_policy(sessions, S2, THRESH, store=store)
store.commit()
print("  sqlite rows      :", store.counts())

ex_rule = store.cx.execute(
    "SELECT device_id, action, rule FROM actions WHERE action='DEAUTHORIZE' LIMIT 1").fetchone()
if ex_rule:
    r = json.loads(ex_rule[2])
    print("  example generated rule:")
    print("   ", json.dumps(r, indent=2)[:640].replace("\n", "\n    "))
    with open(os.path.join(OUT, "generated_rule_example.json"), "w") as fh:
        json.dump(r, fh, indent=2)

# --------------------------------------------------------------- 8. figures
banner("[7/7] FIGURES + REPORT")
demo = next(s for s in test if s.profile == "staged_delay" and pol2[s.sid]["first"])
X, T, y = data[demo.sid]
fig, ax = plt.subplots(3, 1, figsize=(11, 10.5))

ax[0].plot(T, S1[demo.sid], lw=1.4, label="Stage 1 (memoryless anomaly)", color="#888")
ax[0].plot(T, S2[demo.sid], lw=2.0, label="Stage 2 (LSTM P(malicious))", color="#c0392b")
ax[0].axhline(THRESH, ls="--", c="k", lw=1, label=f"learned threshold {THRESH:.2f}")
ax[0].axvline(demo.onset, ls=":", c="#2c7", lw=2, label=f"attack onset {demo.onset:.0f}s")
ax[0].axvline(pol2[demo.sid]["first"], ls="-", c="#2980b9", lw=2,
              label=f"enforcement {pol2[demo.sid]['first']:.0f}s")
ax[0].set_title(f"Staged BadUSB session {demo.sid}: 40 s of human-like typing, then injection")
ax[0].set_xlabel("time since enumeration (s)"); ax[0].set_ylabel("score")
ax[0].legend(fontsize=8); ax[0].set_ylim(-0.02, 1.02)

i_iat = FEATURE_NAMES.index("mean_iat")
i_ent = FEATURE_NAMES.index("iat_entropy")
ax[1].plot(T, X[:, i_iat] * 1000, lw=1.5, color="#8e44ad", label="mean inter-arrival (ms)")
ax[1].set_ylabel("mean IAT (ms)"); ax[1].set_xlabel("time (s)")
ax2 = ax[1].twinx()
ax2.plot(T, X[:, i_ent], lw=1.5, color="#16a085", label="IAT entropy")
ax2.set_ylabel("IAT entropy")
ax[1].axvline(demo.onset, ls=":", c="#2c7", lw=2)
ax[1].set_title("Why: human typing has slow, high-entropy timing; injection is fast and regular")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

profs = [p for p in ATTACK_PROFILES if p in m2["per_profile"]]
xpos = np.arange(len(profs))
ax[2].bar(xpos - 0.27, [m2["per_profile"][p][0] for p in profs], 0.27, label="IUDBMS (S1+S2)", color="#c0392b")
ax[2].bar(xpos, [m1["per_profile"][p][0] for p in profs], 0.27, label="Stage 1 only", color="#e67e22")
ax[2].bar(xpos + 0.27, [0.0] * len(profs), 0.27, label="Identity whitelist", color="#7f8c8d")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(profs, fontsize=9)
ax[2].set_ylabel("session detection rate"); ax[2].set_ylim(0, 1.08)
ax[2].set_title("Detection rate by attack class (held-out test sessions)")
ax[2].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "iudbms_detection_results.png"), dpi=150)
print("  wrote iudbms_detection_results.png")

# ROC
from sklearn.metrics import roc_curve
fig2, axr = plt.subplots(figsize=(6, 5.2))
for name, p, auc, c in (("Stage 1 only", p1, auc1, "#e67e22"),
                        ("Stage 1 + Stage 2 (IUDBMS)", p2, auc2, "#c0392b")):
    fpr, tpr, _ = roc_curve(yte, p)
    axr.plot(fpr, tpr, lw=2, color=c, label=f"{name}  AUC={auc:.3f}")
axr.plot([0, 1], [0, 1], "k--", lw=0.8, label="identity-based defence (chance)")
axr.set_xlabel("false positive rate (benign windows)")
axr.set_ylabel("true positive rate (malicious windows)")
axr.set_title("Window-level ROC, held-out test set")
axr.legend(fontsize=8); plt.tight_layout()
fig2.savefig(os.path.join(OUT, "iudbms_roc.png"), dpi=150)
print("  wrote iudbms_roc.png")

metrics = {
    "corpus": {"sessions": len(sessions), "usbmon_lines": total_lines,
               "windows": n_windows, "features": len(FEATURE_NAMES)},
    "split": {"fit": len(fit), "val": len(val), "test": len(test)},
    "thresholds": {"stage2": THRESH, "stage1": THRESH_S1, "target_fpr": TARGET_FPR},
    "window_level": {"auc_stage1": auc1, "auc_stage2": auc2,
                     "precision": pr, "recall": rc, "f1": f1,
                     "stage1_only": {"precision": pr1, "recall": rc1, "f1": f11}},
    "session_level": {"iudbms": m2, "stage1_only": m1, "identity_whitelist": id_rate},
    "runtime_sec": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2, default=float)
print(f"  wrote metrics.json     total runtime {metrics['runtime_sec']}s")
