"""
Generate the model card: data, features, models, decision rule, terminology.

    python3 -m eval.model_card          # -> docs/MODEL_CARD.md, out/model_card.json

Answers reviewer points 7, 10, 11 and 13 from the code and the trained artefacts
themselves rather than from prose that can drift: every setting is read from the
loaded objects or the constants the code uses, every count from the corpora, and
the traffic examples are real simulator output at a fixed seed. Feature units and
definitions live in FEATURE_DOC below; tests/test_model_card.py asserts it covers
every contract feature, so adding a feature without documenting it fails CI.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from iudbms.capture.simulator import (ATTACK_PROFILES, BENIGN_PROFILES, DURATION_RANGE,
                                      PAYLOAD_KEYS, SimulatedSource)
from iudbms.contract import CONTRACT_VERSION, FEATURE_NAMES, FEATURE_SCHEMA
from iudbms.stage1.model import GATE, MIN_GROUP_WINDOWS, SD_FLOOR, STD_CLIP, VAR_FLOOR, WARMUP
from iudbms.stage2.lstm import TrainConfig

from .analysis_common import CACHE, MODELS, ORIGINAL, OUT, deployed
from .analysis_idle_retrain import IDLE_MIX, QUIET_CLASSES
from .analysis_realistic import (CACHE as CACHE_B, HOLD_MEDIAN, HOLD_SIGMA, SHANNON_DIGRAM_BITS,
                                 SPEED_RANGE, RealisticSim)
from .corpus import load_corpus, stratified_split

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "MODEL_CARD.md"

W = "per 1 s window"
#: name -> (unit, definition). Computed over completion and error events only (C/E),
#: never submissions (S); see features/extractor.py.
FEATURE_DOC: dict[str, tuple[str, str]] = {
    "n_events": ("count", f"completed URBs {W}"),
    "ev_rate": ("events/s", "n_events / window length"),
    "mean_iat": ("s", "mean inter-arrival time between consecutive completions; = 1.0 if < 2 events"),
    "std_iat": ("s", "standard deviation of inter-arrival times; 0 if < 2 events"),
    "cv_iat": ("ratio", "std_iat / mean_iat; 0 if < 2 events (a degenerate value: see finding 36)"),
    "min_iat": ("s", "smallest inter-arrival time; = 1.0 if < 2 events"),
    "p10_iat": ("s", "10th percentile inter-arrival time; = 1.0 if < 2 events"),
    "iat_entropy": ("0-1", "Shannon entropy of the log10 inter-arrival histogram, 12 bins over "
                           "1e-4..10 s, divided by ln 12; 0 if < 2 inter-arrivals"),
    "burstiness_fano": ("ratio", "variance / mean of event counts in ten 0.1 s sub-windows "
                                 "(1 = Poisson, ~0 = regular, >1 = bursty)"),
    "mean_len": ("bytes", "mean URB data length"),
    "std_len": ("bytes", "standard deviation of URB data length"),
    "max_len": ("bytes", "largest URB data length"),
    "bytes_in_rate": ("bytes/s", "sum of device-to-host data lengths / window length"),
    "bytes_out_rate": ("bytes/s", "sum of host-to-device data lengths / window length"),
    "n_endpoints": ("count", "distinct endpoints that carried traffic"),
    "frac_intr": ("0-1", "fraction of events that are interrupt transfers"),
    "frac_bulk": ("0-1", "fraction of events that are bulk transfers"),
    "frac_iso": ("0-1", "fraction of events that are isochronous transfers"),
    "frac_ctrl": ("0-1", "fraction of events that are control transfers"),
    "frac_in": ("0-1", "fraction of events in the device-to-host direction"),
    "payload_novelty": ("0-1", "distinct captured payloads / captured payloads"),
    "repeat_ratio": ("0-1", "count of the most frequent payload / captured payloads"),
    "payload_seq_novelty": ("0-1", "fraction of this window's payload transitions (consecutive "
                                   "pairs) never seen earlier in the device's session; 1.0 if "
                                   "< 2 payloads"),
    "n_interfaces": ("count", "interfaces the device has declared, as visible at window end"),
    "has_hid": ("0/1", "a HID interface (class 0x03) is declared"),
    "has_storage": ("0/1", "a mass-storage interface (class 0x08) is declared"),
    "class_behaviour_mismatch": ("0-1", "fraction of events whose transfer type is outside the "
                                        "union expected for the declared interface classes"),
    "late_iface_flag": ("0/1", "an interface that appeared after enumeration is visible"),
    "log_time_since_enum": ("ln(1+s)", "time since the device enumerated (EXCLUDED: label leak, "
                                       "finding 8)"),
    "concurrent_device_rate": ("ln(1+events/s)", "event rate of every other device on the host "
                                                 "(dropped automatically: constant on a "
                                                 "single-device corpus)"),
}

GLOSSARY = [
    ("window", "one device's traffic in one 1-second interval; the unit the models score"),
    ("window-level recall", "of post-onset attack windows, the fraction scored at or above "
                            "the threshold"),
    ("window-level false-positive rate", "of benign windows, the fraction scored at or above "
                                         "the threshold (the calibration target, 1e-4)"),
    ("session", "one device from attachment to removal (one simulated device run)"),
    ("session-level detection rate", "of attack sessions, the fraction in which the 4-of-6 "
                                     "gate fires at least once after onset"),
    ("session-level false-positive rate", "of benign sessions, the fraction in which the gate "
                                          "fires at least once"),
    ("false alarms per device-hour", "gate firings on benign devices divided by the hours those "
                                     "devices were observed; the unit for long-running use"),
    ("latency", "time from attack onset to the end of the window that completed the gate"),
]


def corpus_counts(sessions, seed: int = 7) -> dict:
    fit, val, test = stratified_split(sessions, seed=seed)
    out = {}
    for name, part in (("fit (train)", fit), ("validation", val), ("test", test)):
        c = Counter(s.profile for s in part)
        out[name] = {"sessions": len(part),
                     "benign": sum(1 for s in part if s.label == 0),
                     "attack": sum(1 for s in part if s.label == 1),
                     "windows": int(sum(len(s.X) for s in part)),
                     "post_onset_windows": int(sum(int(s.y.sum()) for s in part)),
                     "hours": round(sum(s.duration for s in part) / 3600, 2),
                     "per_profile": dict(sorted(c.items()))}
    return out


def traffic_examples(n: int = 10) -> dict:
    """usbmon lines from fixed-seed sessions: what the detector actually reads."""
    sim = RealisticSim(seed=11)
    sim._idle, sim._active, sim._fully_idle = (0.0, 0.0), (1e9, 1e9), False
    out = {}
    for prof in ("keyboard", "hid_injection", "mimicry"):
        s = sim.session("ex", prof, duration=60.0)
        start = (s.onset or 5.0) + 2.0
        lines = [ln for ln in s.lines if int(ln.split()[1]) / 1e6 >= start and " C " in ln]
        out[prof] = {"onset_s": s.onset, "from_s": start, "lines": lines[:n]}
    return out


def build() -> dict:
    cfg, s1, lstm = deployed(MODELS)
    idx = list(cfg["feature_idx"])
    tc = TrainConfig()
    B = load_corpus(CACHE_B).sessions
    A = load_corpus(CACHE).sessions
    fitB, _, _ = stratified_split(B, seed=7)
    y = np.concatenate([s.y for s in fitB])
    features = []
    for i, (name, mod) in enumerate(FEATURE_SCHEMA):
        unit, defn = FEATURE_DOC[name]
        features.append({"index": i, "name": name, "modality": mod, "unit": unit,
                         "definition": defn, "used_by_deployed_model": i in idx})
    card = {
        "contract_version": CONTRACT_VERSION,
        "deployed_model": {"path": str(MODELS.relative_to(ROOT)), "trained_on": "Corpus B",
                           "threshold": cfg["threshold"], "target_window_fpr": cfg["target_fpr"],
                           "gate": {"k": cfg["k"], "n": cfg["n"]}, "seq_len": cfg["seq_len"],
                           "n_features": len(idx), "canary": cfg.get("canary")},
        "features": features,
        "normalisation": {
            "stage1": f"per-feature z-score with the mean and SD of benign training windows; SD "
                      f"floored at {SD_FLOOR} (SD_FLOOR) and the result clipped to +/-{STD_CLIP} "
                      f"(STD_CLIP)",
            "stage2": "per-input z-score with the mean and SD of all training sequence rows "
                      "(SD clamped at 1e-6), stored in the checkpoint and applied in forward()",
        },
        "stage1": {
            "model": "Isolation Forest per declared class group (hid, storage, av, composite), "
                     "plus a generic forest over all benign windows used for groups with fewer "
                     f"than {MIN_GROUP_WINDOWS} training windows ('other')",
            "groups_fitted": sorted(s1.forests),
            "n_estimators": s1.n_estimators, "contamination": s1.contamination,
            "max_samples": "min(512, windows in the group)", "random_state": s1.seed,
            "device_baseline": {"type": "per-device EWMA of mean and variance", "alpha": 0.15,
                                "warmup_windows": WARMUP, "update_gate": GATE,
                                "variance_floor": VAR_FLOOR},
            "fusion_weights": {"forest": s1.weights[0], "device_baseline": s1.weights[1]},
            "output": "anomaly score in [0, 1]; forest term scaled by its benign 99th percentile, "
                      "device term by the benign 99th percentile of the deviation",
            "trained_on": "benign windows of the fit split only (no labels)",
        },
        "stage2": {
            "input": f"the {len(idx)} deployed features + the Stage-1 score = {lstm.d_in} inputs "
                     f"per window, as sequences of the last {cfg['seq_len']} windows, left-padded "
                     "by repeating the earliest row (strictly causal)",
            "architecture": f"1-layer LSTM, hidden size {lstm.hidden}, forget-gate bias 1; "
                            f"Linear({lstm.hidden} -> 1) on the last time step; sigmoid",
            "output_layer": "one logit -> sigmoid = probability the current window is post-onset",
            "loss": "binary cross-entropy with logits (BCEWithLogitsLoss)",
            "class_weighting": "pos_weight = negative windows / positive windows in the training "
                               f"set ({(len(y) - y.sum()) / max(1, y.sum()):.2f} for Corpus B's "
                               "fit split)",
            "optimiser": f"Adam, learning rate {tc.lr}, weight decay {tc.weight_decay}, gradient "
                         f"norm clipped at {tc.grad_clip}",
            "batch_size": tc.batch_size, "epochs": tc.epochs,
            "stopping_rule": f"fixed {tc.epochs} epochs; no early stopping. Validation recall is "
                             "logged per epoch but does not select the checkpoint",
            "seed": 7, "labels": "window y = 1 iff the window ends after the attack onset",
        },
        "decision": {
            "window_threshold": cfg.get("threshold_rule") or (
                "the (1 - target_fpr) quantile of Stage-2 scores on validation windows labelled "
                "benign (benign sessions and pre-onset windows of attack sessions)"),
            "threshold_history": cfg.get("threshold_history", []),
            "gate": f"alert when {cfg['k']} of the last {cfg['n']} windows of one device are at or "
                    "above the threshold; the alert is timed at the end of the completing window",
            "session_alarm": "a session counts as alarmed if the gate fires at least once",
            "from_window_to_session": "not analytic: windows of one device are strongly "
                                      "autocorrelated (consecutive Stage-2 inputs share 9 of 10 "
                                      "rows), so session-level false alarms are MEASURED on held-out "
                                      "benign sessions and long sessions (out/analysis_session_fp.json)",
            "enforcement": "THROTTLE then DEAUTHORIZE are generated, logged and stored but NOT "
                           "applied: ENFORCEMENT_REVIEWED is False (dry-run)",
        },
        "data": {
            "corpus_A": {"description": "original simulator: every benign device always active",
                         "seed": 7, "splits": corpus_counts(A)},
            "corpus_B": {"description": "realistic simulator: duty cycles, HID key-release and "
                                        "rollover reports, language-like text, per-device speed; "
                                        "the deployed model's training data",
                         "seed": 7, "splits": corpus_counts(B)},
            "corpus_C": {"description": "fresh realistic set, 100 sessions per attack family + "
                                        "3,000 benign; never used to train, calibrate or select",
                         "seed": 2025},
            "split_rule": "by session, never by window; stratified by profile for the 30% test "
                          "split, then 20% of the remainder drawn at random for validation; "
                          "eval.corpus.stratified_split asserts no session crosses splits",
            "generator": {
                "session_duration_s": list(DURATION_RANGE),
                "benign_profiles": list(BENIGN_PROFILES), "attack_profiles": list(ATTACK_PROFILES),
                "duty_cycle_mix": [{"name": n, "p": p, **{k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in kw.items()}}
                                   for n, p, kw in IDLE_MIX],
                "duty_cycled_classes": list(QUIET_CLASSES),
                "typing_speed_multiplier": list(SPEED_RANGE),
                "key_hold_lognormal": {"median_s": HOLD_MEDIAN, "sigma": HOLD_SIGMA},
                "text_source_bits_per_symbol": SHANNON_DIGRAM_BITS,
                "single_shot_payload_keys": list(PAYLOAD_KEYS),
            },
        },
        "traffic_examples": traffic_examples(),
        "glossary": [{"term": t, "meaning": m} for t, m in GLOSSARY],
        "reproduce": {
            "deployed model": "make train-deployed   (Corpus B, seed 7)",
            "Corpus-A headline": "make eval   (seed 7)",
            "this card": "python3 -m eval.model_card",
        },
    }
    _ = ORIGINAL
    return card


def to_markdown(c: dict) -> str:
    L = ["# IUDBMS model card", "",
         "Generated by `python3 -m eval.model_card` from the trained artefacts and the code's own "
         "constants. Do not edit by hand.", "",
         f"Contract {c['contract_version']}. Deployed model: `{c['deployed_model']['path']}`, "
         f"trained on {c['deployed_model']['trained_on']}; window threshold "
         f"{c['deployed_model']['threshold']:.6f} (rule below); gate {c['deployed_model']['gate']['k']}-of-"
         f"{c['deployed_model']['gate']['n']}.", "",
         "## Terminology", "", "| term | meaning |", "|---|---|"]
    L += [f"| {g['term']} | {g['meaning']} |" for g in c["glossary"]]
    L += ["", "## Input features (order is frozen; index = column)", "",
          "| # | name | modality | unit | definition | in deployed model |", "|---|---|---|---|---|---|"]
    L += [f"| {f['index']} | `{f['name']}` | {f['modality']} | {f['unit']} | {f['definition']} | "
          f"{'yes' if f['used_by_deployed_model'] else 'no'} |" for f in c["features"]]
    n = c["normalisation"]
    L += ["", "## Normalisation", "", f"* **Stage 1:** {n['stage1']}.", f"* **Stage 2:** {n['stage2']}.",
          "", "## Stage 1 (unsupervised, benign-only)", ""]
    for k, v in c["stage1"].items():
        L.append(f"* **{k}:** {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    L += ["", "## Stage 2 (supervised)", ""]
    for k, v in c["stage2"].items():
        L.append(f"* **{k}:** {v}")
    L += ["", "## Decision rule", ""]
    for k, v in c["decision"].items():
        L.append(f"* **{k}:** {v}")
    L += ["", "## Data", "", f"Split rule: {c['data']['split_rule']}.", ""]
    for corp in ("corpus_A", "corpus_B"):
        d = c["data"][corp]
        L += [f"### {corp.replace('_', ' ').title()} (seed {d['seed']}): {d['description']}", "",
              "| split | sessions | benign | attack | windows | post-onset windows | hours |",
              "|---|---|---|---|---|---|---|"]
        for sp, v in d["splits"].items():
            L.append(f"| {sp} | {v['sessions']} | {v['benign']} | {v['attack']} | {v['windows']} | "
                     f"{v['post_onset_windows']} | {v['hours']} |")
        L += ["", "Per profile (fit / validation / test): " + ", ".join(
            f"{p} {d['splits']['fit (train)']['per_profile'].get(p, 0)}/"
            f"{d['splits']['validation']['per_profile'].get(p, 0)}/"
            f"{d['splits']['test']['per_profile'].get(p, 0)}"
            for p in sorted(d["splits"]["test"]["per_profile"])), ""]
    L += [f"### Corpus C (seed {c['data']['corpus_C']['seed']}): {c['data']['corpus_C']['description']}", "",
          "### Generator settings", "", "```json", json.dumps(c["data"]["generator"], indent=1), "```", "",
          "## Example traffic (usbmon text, completions only, fixed seed)", ""]
    for prof, ex in c["traffic_examples"].items():
        L += [f"**{prof}** (onset {ex['onset_s']}; from t = {ex['from_s']:.1f} s):", "", "```"]
        L += ex["lines"] + ["```", ""]
    L += ["## Reproduce", ""] + [f"* {k}: `{v}`" for k, v in c["reproduce"].items()]
    return "\n".join(L) + "\n"


def main() -> int:
    card = build()
    (OUT / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    DOC.write_text(to_markdown(card))
    print(f"wrote {DOC} and out/model_card.json  ({len(card['features'])} features documented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
