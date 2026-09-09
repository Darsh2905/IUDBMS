"""
IUDBMS - Layer 2 feature extraction.

Novelty pillar 1: multi-modal fusion. Every window vector mixes four modalities
that prior work uses in isolation:

  identity/descriptor : declared class, interface count, late-interface event
  timing              : IAT moments, entropy, burstiness (Fano factor)
  payload             : transfer sizes, novelty, repetition
  protocol/endpoint    : endpoint fan-out, transfer-type mix, class-behaviour mismatch

The class-behaviour mismatch feature is what makes descriptor spoofing useless:
a device may *claim* HID, but if its URBs are bulk OUT, the mismatch term fires.
"""

import math
import numpy as np

WINDOW = 1.0        # seconds
STRIDE = 1.0

FEATURE_NAMES = [
    "n_events", "ev_rate", "mean_iat", "std_iat", "cv_iat", "min_iat", "p10_iat",
    "iat_entropy", "burstiness_fano",
    "mean_len", "std_len", "max_len", "bytes_in_rate", "bytes_out_rate",
    "n_endpoints", "frac_intr", "frac_bulk", "frac_iso", "frac_ctrl", "frac_in",
    "payload_novelty", "repeat_ratio",
    "n_interfaces", "has_hid", "has_storage",
    "class_behaviour_mismatch", "late_iface_flag", "log_time_since_enum",
]
N_FEATURES = len(FEATURE_NAMES)

EXPECTED_TTYPE = {
    0x03: {"intr", "control"},          # HID
    0x08: {"bulk", "control"},          # mass storage
    0x0E: {"iso", "control", "intr"},   # video
    0x01: {"iso", "control"},           # audio
    0x02: {"bulk", "intr", "control"},  # CDC
}


def _entropy(values, bins, lo, hi):
    if len(values) < 2:
        return 0.0
    h, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(bins))


def _fano(ts, t0, sub=0.1):
    if len(ts) < 2:
        return 0.0
    counts, _ = np.histogram(ts, bins=int(WINDOW / sub), range=(t0, t0 + WINDOW))
    m = counts.mean()
    return float(counts.var() / m) if m > 0 else 0.0


def window_features(events, t0, descriptor):
    """events: URBEvent list falling inside [t0, t0+WINDOW)."""
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    ifaces = descriptor["interfaces"]
    late_ts = descriptor.get("late_iface_ts")

    # --- identity modality (available even with zero traffic) ---
    visible = [i for i in ifaces]
    if late_ts is not None and t0 + WINDOW < late_ts:
        visible = ifaces[:-1]
    f["n_interfaces"] = float(len(visible))
    f["has_hid"] = float(any(i["bInterfaceClass"] == 0x03 for i in visible))
    f["has_storage"] = float(any(i["bInterfaceClass"] == 0x08 for i in visible))
    f["late_iface_flag"] = float(late_ts is not None and t0 + WINDOW >= late_ts)
    f["log_time_since_enum"] = float(math.log1p(t0))

    if not events:
        f["mean_iat"] = f["min_iat"] = f["p10_iat"] = WINDOW
        return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)

    ts = np.array([e.ts for e in events])
    lens = np.array([e.length for e in events], dtype=np.float64)
    n = len(events)

    # --- timing modality ---
    f["n_events"] = float(n)
    f["ev_rate"] = n / WINDOW
    if n >= 2:
        iat = np.diff(ts)
        f["mean_iat"] = float(iat.mean())
        f["std_iat"] = float(iat.std())
        f["cv_iat"] = float(iat.std() / iat.mean()) if iat.mean() > 0 else 0.0
        f["min_iat"] = float(iat.min())
        f["p10_iat"] = float(np.percentile(iat, 10))
        f["iat_entropy"] = _entropy(np.log10(np.maximum(iat, 1e-6)), 12, -4.0, 1.0)
    else:
        f["mean_iat"] = f["min_iat"] = f["p10_iat"] = WINDOW
    f["burstiness_fano"] = _fano(ts, t0)

    # --- payload modality ---
    f["mean_len"] = float(lens.mean())
    f["std_len"] = float(lens.std())
    f["max_len"] = float(lens.max())
    f["bytes_in_rate"] = float(sum(e.length for e in events if e.direction == "in") / WINDOW)
    f["bytes_out_rate"] = float(sum(e.length for e in events if e.direction == "out") / WINDOW)
    payloads = [e.data for e in events if e.data]
    if payloads:
        uniq = {}
        for p in payloads:
            uniq[p] = uniq.get(p, 0) + 1
        f["payload_novelty"] = len(uniq) / len(payloads)
        f["repeat_ratio"] = max(uniq.values()) / len(payloads)

    # --- protocol / endpoint modality ---
    f["n_endpoints"] = float(len({e.ep for e in events}))
    for key, tt in (("frac_intr", "intr"), ("frac_bulk", "bulk"),
                    ("frac_iso", "iso"), ("frac_ctrl", "control")):
        f[key] = sum(1 for e in events if e.ttype == tt) / n
    f["frac_in"] = sum(1 for e in events if e.direction == "in") / n

    allowed = set()
    for i in visible:
        allowed |= EXPECTED_TTYPE.get(i["bInterfaceClass"], {"control"})
    f["class_behaviour_mismatch"] = sum(1 for e in events if e.ttype not in allowed) / n

    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)


def sessionise(session, events):
    """Return (X [W,F], t_starts [W], y [W]) for one device session."""
    if events:
        t_end = max(e.ts for e in events)
    else:
        t_end = session.duration
    t_end = max(t_end, session.duration)
    n_win = max(1, int(t_end // STRIDE))

    buckets = [[] for _ in range(n_win)]
    for e in events:
        idx = int(e.ts // STRIDE)
        if 0 <= idx < n_win:
            buckets[idx].append(e)

    X, T, y = [], [], []
    for w in range(n_win):
        t0 = w * STRIDE
        X.append(window_features(buckets[w], t0, session.descriptor))
        T.append(t0)
        malicious = session.label == 1 and session.onset is not None and (t0 + WINDOW) > session.onset
        y.append(int(malicious))
    return np.vstack(X), np.array(T), np.array(y)
