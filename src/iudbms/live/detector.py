"""
The streaming detector — the thing a real deployment actually runs.

Everything else in this repo scores a *stored* session: the evaluation harness
builds a feature extractor and a device baseline per session, scores the whole
array at once, and throws the state away. That is fine for measurement and
useless as a monitor, and it is why the SQLite ``baselines`` table has always
persisted empty.

This class holds the per-device state a live deployment needs, and closes the
loop the thesis rests on: capture -> features -> Stage 1 -> Stage 2 -> policy ->
action, incrementally, one window at a time, with no future information.

Per device it owns:

    WindowFeatureExtractor   payload history for payload_seq_novelty
    DeviceBaseline           the online EWMA baseline (novelty pillar N2)
    deque of recent vectors  the LSTM's K-window sequence
    DeviceContext            the k-of-n persistence gate

Causality is structural here rather than promised: the detector cannot see a
future window because it has not been given one. Scores produced live are
therefore identical to the offline ones, which
``test_streaming_matches_offline_scoring`` asserts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..contract import FEATURE_NAMES, DeviceDescriptor, URBEvent
from ..features.extractor import WindowFeatureExtractor
from ..policy import Action, DeviceContext, PolicyEngine, is_keyboard
from ..stage1 import DeviceBaseline, Stage1, class_group
from ..stage2 import SEQ_LEN, LSTMDetector

__all__ = ["StreamingDetector", "DeviceState", "WindowVerdict"]


@dataclass
class WindowVerdict:
    """What the detector concluded about one window of one device."""

    device_id: str
    t_start: float
    features: np.ndarray
    stage1: float
    stage2: float
    evidence: list[tuple[str, float]]
    gate_hits: int
    gate_n: int
    trust_state: str
    action: Action | None = None

    @property
    def over_threshold(self) -> bool:
        return self.gate_hits > 0


@dataclass
class DeviceState:
    device_id: str
    descriptor: DeviceDescriptor
    extractor: WindowFeatureExtractor
    baseline: DeviceBaseline
    ctx: DeviceContext
    history: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    n_windows: int = 0


class StreamingDetector:
    """Scores devices incrementally as their traffic arrives.

    ``feature_idx`` selects the deployed feature subset (28 of 29 — see the
    ``log_time_since_enum`` finding); it must match what the models were
    trained on.
    """

    def __init__(
        self,
        stage1: Stage1,
        stage2: LSTMDetector | None,
        policy: PolicyEngine,
        *,
        feature_idx: np.ndarray | None = None,
        seq_len: int = SEQ_LEN,
        human_max_rps: float | None = None,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.policy = policy
        self.seq_len = seq_len
        self.feature_idx = (np.arange(len(FEATURE_NAMES)) if feature_idx is None
                            else np.asarray(feature_idx))
        self.devices: dict[str, DeviceState] = {}
        #: Human-capability rule (eval.analysis_hybrid): a boot-protocol keyboard sending
        #: at least this many completed URBs in a 1 s window is faster than any of 1,972
        #: real typists sustained. None disables it (a model set that did not adopt it).
        self.human_max_rps = human_max_rps

    # ------------------------------------------------------------------

    def attach(self, descriptor: DeviceDescriptor, device_id: str | None = None) -> DeviceState:
        """Register a device at enumeration. Idempotent."""
        did = device_id or descriptor.device_key
        if did not in self.devices:
            names = tuple(FEATURE_NAMES[i] for i in self.feature_idx)
            self.devices[did] = DeviceState(
                device_id=did, descriptor=descriptor,
                extractor=WindowFeatureExtractor(),
                baseline=DeviceBaseline(feature_names=names),
                ctx=self.policy.new_device(did),
            )
        return self.devices[did]

    def detach(self, device_id: str) -> None:
        """Forget a device on unplug, so its history cannot leak into the next
        device that lands on the same bus address."""
        self.devices.pop(device_id, None)

    # ------------------------------------------------------------------

    def observe_window(
        self, device_id: str, t_start: float, events: list[URBEvent],
    ) -> WindowVerdict:
        """Feed one window of events and get the verdict for it."""
        st = self.devices.get(device_id)
        if st is None:
            raise KeyError(f"{device_id} was never attached; call attach() at enumeration")

        x_full = st.extractor.extract_checked(events, t_start, st.descriptor)
        x = x_full[self.feature_idx].astype(np.float64)

        s1, evidence = self._stage1_score(st, x)
        st.history.append(np.append(x, s1))
        s2 = self._stage2_score(st) if self.stage2 is not None else s1

        rule_hit = (self.human_max_rps is not None and is_keyboard(st.descriptor)
                    and x_full[0] >= self.human_max_rps)            # column 0: n_events
        action = self.policy.step(st.ctx, t_start, s1, s2, evidence, st.descriptor,
                                  rule_hit=bool(rule_hit))
        st.n_windows += 1

        return WindowVerdict(
            device_id=device_id, t_start=t_start, features=x_full,
            stage1=s1, stage2=s2, evidence=evidence,
            gate_hits=st.ctx.hits_in_window, gate_n=self.policy.n,
            trust_state=st.ctx.state, action=action,
        )

    def _stage1_score(self, st: DeviceState, x: np.ndarray) -> tuple[float, list]:
        """One window through the population model and the device baseline."""
        m = self.stage1
        g = class_group(st.descriptor)
        z = m._std(x.reshape(1, -1))[0]
        a = float(m._raw_if(x.reshape(1, -1), g)[0]) / (m.scale["if"] + 1e-6)
        d, top = st.baseline.deviation(z)
        b = d / (m.scale["dev"] + 1e-6)
        w_if, w_dev = m.weights
        s = float(np.clip(w_if * np.clip(a, 0, 3) / 3
                          + w_dev * np.clip(b, 0, 3) / 3, 0, 1))
        # Update-gate, exactly as offline: an already-anomalous window must not
        # teach the baseline that it is normal.
        from ..stage1 import GATE, WARMUP
        if s < GATE or st.baseline.n < WARMUP:
            st.baseline.update(z)
        return s, top

    def _stage2_score(self, st: DeviceState) -> float:
        """Left-pad by repetition, matching build_sequences exactly."""
        hist = list(st.history)
        if len(hist) < self.seq_len:
            hist = [hist[0]] * (self.seq_len - len(hist)) + hist
        seq = np.asarray(hist, dtype=np.float32)[None, :, :]
        return float(self.stage2.predict_proba(seq)[0])
