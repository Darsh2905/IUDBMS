"""
L2 — multi-modal feature extraction (novelty pillar N1).

Turns a 1.0 s window of URB events plus a descriptor snapshot into the frozen
30-dimensional vector (contract 3.0.0). Modalities that prior work uses in isolation are
fused here:

    identity/descriptor  what the device claims to be, and when it changed
    timing               inter-arrival moments, entropy, burstiness
    payload              transfer sizes, novelty, repetition
    protocol/endpoint    endpoint fan-out, transfer-type mix, class mismatch

``class_behaviour_mismatch`` is what makes descriptor spoofing useless: a
device may *claim* HID, but if its URBs are bulk OUT the mismatch term fires.

Which events count
------------------
Features are computed over **completions and errors only** (``C`` and ``E``),
never submissions (``S``). Every submission has a matching completion, so
counting both merely doubles every rate while adding no information, and it
makes the result depend on kernel scheduling rather than device behaviour. The
completion is also the event that actually moved data — a submission's
``length`` is only the *requested* size.

This is load-bearing for two reasons beyond tidiness:

* It is why the Sprint-0 corpus's missing ``S`` events (CLAUDE.md §4 defect 3)
  were never load-bearing. Adding them to the simulator makes the trace
  realistic *without* shifting any baseline, and
  ``test_submissions_do_not_change_the_vector`` asserts exactly that.
* It is what lets the usbmon and eBPF backends agree on event counts (plan
  §4.1) regardless of whether a backend reports submissions at all.

Empty windows are legitimate input, not an error: a zero-traffic second is a
real observation about a device, and every feature must still be finite
(plan §4.2).
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np

from ..contract import (
    CLASS_AUDIO,
    CLASS_CDC,
    CLASS_HID,
    CLASS_MASS_STORAGE,
    CLASS_VIDEO,
    FEATURE_DTYPE,
    FEATURE_NAMES,
    N_FEATURES,
    DeviceDescriptor,
    FeatureExtractor,
    URBEvent,
)

__all__ = ["WindowFeatureExtractor", "EXPECTED_TTYPE", "window_events",
           "COUNTED_KINDS", "HostContext"]


class HostContext:
    """Activity of the OTHER devices attached to the same host.

    Pillar N4's bound (CLAUDE.md finding 11) is that within a single device's
    URB stream a single-shot injection is statistically the benign typing
    process: a per-feature KS scan separates ``mimicry_adaptive`` on zero of the
    28 deployed features. The named next defence is cross-device correlation --
    a human typing is usually also moving a mouse; a BadUSB typing alone is not.

    This carries that one piece of host-level state. It is deliberately a
    separate object rather than extra arguments, because a live deployment has
    exactly one of these per host and shares it across every device's extractor,
    while a single-device capture has none at all (and the feature is then 0.0,
    which is also what a genuinely lone device sees).
    """

    __slots__ = ("_rate",)

    def __init__(self, rate_by_window: dict[tuple[str, int], float] | None = None):
        #: (device_key, window_index) -> events/sec on every OTHER device
        self._rate = rate_by_window or {}

    def other_rate(self, device_key: str, t0: float, window: float) -> float:
        return self._rate.get((device_key, int(round(t0 / window))), 0.0)

#: Only these event kinds contribute to traffic-derived features. See module docstring.
COUNTED_KINDS = frozenset({"C", "E"})

#: Transfer types each interface class is expected to use. A device whose URBs
#: fall outside the union of its declared classes' expectations is anomalous by
#: construction, independent of any learned model.
EXPECTED_TTYPE: dict[int, frozenset[str]] = {
    CLASS_AUDIO:         frozenset({"iso", "control"}),
    CLASS_CDC:           frozenset({"bulk", "intr", "control"}),
    CLASS_HID:           frozenset({"intr", "control"}),
    CLASS_MASS_STORAGE:  frozenset({"bulk", "control"}),
    CLASS_VIDEO:         frozenset({"iso", "control", "intr"}),
}

#: Fallback for classes we do not model: control traffic is always legitimate.
_DEFAULT_EXPECTED = frozenset({"control"})

_IAT_ENTROPY_BINS = 12
_IAT_LOG_LO, _IAT_LOG_HI = -4.0, 1.0     # log10 seconds: 100 us .. 10 s
_FANO_SUBWINDOW = 0.1                     # seconds


def _entropy(values: np.ndarray, bins: int, lo: float, hi: float) -> float:
    """Shannon entropy of a histogram, normalised to [0, 1]."""
    if len(values) < 2:
        return 0.0
    h, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    total = h.sum()
    if total == 0:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(bins))


def _fano(ts: np.ndarray, t0: float, window: float) -> float:
    """Fano factor (variance/mean of sub-window counts): burstiness.

    1.0 for a Poisson stream, ~0 for a perfectly regular one, large for bursty.
    """
    if len(ts) < 2:
        return 0.0
    n_bins = max(1, int(round(window / _FANO_SUBWINDOW)))
    counts, _ = np.histogram(ts, bins=n_bins, range=(t0, t0 + window))
    m = counts.mean()
    return float(counts.var() / m) if m > 0 else 0.0


class WindowFeatureExtractor(FeatureExtractor):
    """The 29-feature extractor.

    Holds per-session payload history for ``payload_seq_novelty`` (contract
    2.0.0), so **one instance belongs to one device session**. Call
    :meth:`reset` between sessions.
    """

    #: Cap on remembered payload transitions. A bulk device emits thousands of
    #: distinct random payloads per second and would otherwise grow this set
    #: without bound. Once full, novelty keeps reporting ~1.0, which is the
    #: correct answer for genuinely random data.
    MAX_REMEMBERED_BIGRAMS = 8192

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_bigrams: set[tuple[bytes, bytes]] = set()
        self._prev_payload: bytes | None = None

    def _seq_novelty(self, payloads: list[bytes]) -> float:
        """Fraction of this window's payload transitions not seen earlier.

        This is the cross-window signal the per-window features cannot carry.
        A replayed keystroke script cycles a fixed set of transitions, so its
        novelty decays toward zero as the script repeats; a human types novel
        pairs indefinitely. Measured causally -- each transition is judged
        against history from *before* it, then folded in -- so training and
        live inference see identical values.

        Defaults to 1.0 when a window holds fewer than two payloads, including
        idle windows: "no evidence of repetition" is the benign reading, so
        silence cannot manufacture an alert.
        """
        seq = ([self._prev_payload] if self._prev_payload is not None else []) + payloads
        if len(seq) < 2:
            if payloads:
                self._prev_payload = payloads[-1]
            return 1.0

        novel = 0
        total = 0
        for a, b in zip(seq, seq[1:]):
            total += 1
            bigram = (a, b)
            if bigram not in self._seen_bigrams:
                novel += 1
                if len(self._seen_bigrams) < self.MAX_REMEMBERED_BIGRAMS:
                    self._seen_bigrams.add(bigram)
        self._prev_payload = seq[-1]
        return novel / total if total else 1.0

    def extract(
        self,
        events: Sequence[URBEvent],
        t0: float,
        descriptor: DeviceDescriptor,
        host: "HostContext | None" = None,
    ) -> np.ndarray:
        f: dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)
        t_end = t0 + self.WINDOW

        # ---------------- host modality (contract 3.0.0) ----------------
        # log1p because the quantity is a rate spanning orders of magnitude
        # (a mouse emits ~100/s, a keyboard ~2/s) and the models standardise
        # linearly. Absent context is 0.0, which is also what a genuinely lone
        # device observes -- the two are indistinguishable by design, and that
        # is the honest limit of the feature.
        if host is not None:
            f["concurrent_device_rate"] = math.log1p(
                host.other_rate(descriptor.device_key, t0, self.WINDOW))

        # ---------------- identity modality ----------------
        # Available even with zero traffic, which is the point: a device that
        # goes silent still has an identity, and one that grows an interface
        # mid-session is anomalous before it sends anything.
        visible = descriptor.visible_at(t_end)
        classes = {i.bInterfaceClass for i in visible}
        f["n_interfaces"] = float(len(visible))
        f["has_hid"] = float(CLASS_HID in classes)
        f["has_storage"] = float(CLASS_MASS_STORAGE in classes)
        f["late_iface_flag"] = float(
            any(i.first_seen_ts > 0.0 and t_end >= i.first_seen_ts for i in descriptor.interfaces)
        )
        f["log_time_since_enum"] = float(math.log1p(max(t0, 0.0)))

        counted = [e for e in events if e.evt in COUNTED_KINDS]
        n = len(counted)
        if n == 0:
            # Still advance session state: an idle second is history too.
            f["payload_seq_novelty"] = self._seq_novelty([])
            # An idle second. IATs are undefined; report them as a full window
            # rather than zero, so "silent" does not masquerade as "infinitely
            # fast" to any downstream scaler.
            f["mean_iat"] = f["min_iat"] = f["p10_iat"] = self.WINDOW
            return self._pack(f)

        ts = np.fromiter((e.ts for e in counted), dtype=np.float64, count=n)
        lens = np.fromiter((e.length for e in counted), dtype=np.float64, count=n)

        # ---------------- timing modality ----------------
        f["n_events"] = float(n)
        f["ev_rate"] = n / self.WINDOW
        if n >= 2:
            iat = np.diff(ts)
            mean_iat = float(iat.mean())
            f["mean_iat"] = mean_iat
            f["std_iat"] = float(iat.std())
            f["cv_iat"] = float(iat.std() / mean_iat) if mean_iat > 0 else 0.0
            f["min_iat"] = float(iat.min())
            f["p10_iat"] = float(np.percentile(iat, 10))
            f["iat_entropy"] = _entropy(
                np.log10(np.maximum(iat, 1e-6)), _IAT_ENTROPY_BINS, _IAT_LOG_LO, _IAT_LOG_HI
            )
        else:
            f["mean_iat"] = f["min_iat"] = f["p10_iat"] = self.WINDOW
        f["burstiness_fano"] = _fano(ts, t0, self.WINDOW)

        # ---------------- payload modality ----------------
        f["mean_len"] = float(lens.mean())
        f["std_len"] = float(lens.std())
        f["max_len"] = float(lens.max())
        inbound = np.fromiter((e.direction == "in" for e in counted), dtype=bool, count=n)
        f["bytes_in_rate"] = float(lens[inbound].sum() / self.WINDOW)
        f["bytes_out_rate"] = float(lens[~inbound].sum() / self.WINDOW)

        # Only events with a captured payload participate. Events the kernel
        # declined to capture carry data == b"", and counting those would make
        # an uncaptured bulk stream look like a perfectly repeating one — the
        # exact conflation URBEvent.data_truncated exists to prevent.
        payloads = [e.data for e in counted if e.data]
        if payloads:
            uniq: dict[bytes, int] = {}
            for p in payloads:
                uniq[p] = uniq.get(p, 0) + 1
            f["payload_novelty"] = len(uniq) / len(payloads)
            f["repeat_ratio"] = max(uniq.values()) / len(payloads)
        f["payload_seq_novelty"] = self._seq_novelty(payloads)

        # ---------------- protocol / endpoint modality ----------------
        f["n_endpoints"] = float(len({e.ep for e in counted}))
        tt_counts: dict[str, int] = {}
        for e in counted:
            tt_counts[e.ttype] = tt_counts.get(e.ttype, 0) + 1
        f["frac_intr"] = tt_counts.get("intr", 0) / n
        f["frac_bulk"] = tt_counts.get("bulk", 0) / n
        f["frac_iso"] = tt_counts.get("iso", 0) / n
        f["frac_ctrl"] = tt_counts.get("control", 0) / n
        f["frac_in"] = float(inbound.sum()) / n

        allowed: set[str] = set()
        for i in visible:
            allowed |= EXPECTED_TTYPE.get(i.bInterfaceClass, _DEFAULT_EXPECTED)
        if not allowed:
            allowed = set(_DEFAULT_EXPECTED)
        f["class_behaviour_mismatch"] = sum(1 for e in counted if e.ttype not in allowed) / n

        return self._pack(f)

    @staticmethod
    def _pack(f: dict[str, float]) -> np.ndarray:
        return np.fromiter((f[k] for k in FEATURE_NAMES), dtype=FEATURE_DTYPE,
                           count=N_FEATURES)


def window_events(
    events: Sequence[URBEvent],
    *,
    window: float = 1.0,
    stride: float = 1.0,
    duration: float | None = None,
) -> Iterator[tuple[float, list[URBEvent]]]:
    """Bucket events into tumbling windows, yielding ``(t0, events_in_window)``.

    Empty windows are yielded, not skipped: an idle second is an observation.
    ``duration`` extends the window count beyond the last event, so a device
    that goes silent for the rest of a session still produces windows.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    # Two independent lower bounds on the window count, because they round
    # differently: an event at exactly t=4.0 belongs to window 4 (so it needs
    # five windows), whereas a 4.0 s session needs only four to cover [0, 4).
    by_events = int(max((e.ts for e in events), default=-1.0) // stride) + 1
    by_duration = math.ceil((duration or 0.0) / stride)
    n_win = max(1, by_events, by_duration)

    buckets: list[list[URBEvent]] = [[] for _ in range(n_win)]
    for e in events:
        idx = int(e.ts // stride)
        if 0 <= idx < n_win:
            buckets[idx].append(e)

    for w in range(n_win):
        yield w * stride, buckets[w]
