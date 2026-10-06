"""
Persist the live monitor's verdicts into the store the API and dashboard read.

Before this, the live path wrote **nothing**: it printed to a terminal, and the
SQLite store was only ever filled by ``eval.run_eval --persist`` from a batch
run. A deployed monitor's alerts therefore never reached the dashboard, and the
operator API served a historical evaluation rather than the machine it was
running on.

The semantics mirror ``eval/persist.py`` so the dashboard reads live and batch
data the same way: one ``windows`` row per scored window, and an ``alerts`` row
plus an ``actions`` row only when the policy engine escalates (THROTTLE or
DEAUTHORIZE) -- not on every window above threshold, which the 4-of-6 gate
exists to suppress.

Two things differ from batch, deliberately:

* **Labels are NULL.** Live traffic has no ground truth; writing 0 would claim
  every window was known benign.
* **An escalation pins its evidence.** The raw URBs from the windows the model
  actually saw -- the last ``seq_len`` windows -- are exempted from trace
  eviction, so every action keeps the traffic that justified it.
"""

from __future__ import annotations

from collections.abc import Callable

from ..contract import FEATURE_NAMES, DeviceDescriptor
from ..stage1 import class_group
from .detector import WindowVerdict

__all__ = ["LivePersister"]


class LivePersister:
    def __init__(
        self,
        store,
        *,
        recorder=None,
        seq_len: int = 10,
        window: float = 1.0,
        commit_every: int = 1,
    ):
        self.store = store
        self.recorder = recorder
        self.seq_len = seq_len
        self.window = window
        self.commit_every = max(1, commit_every)
        self._known: set[str] = set()
        self._state: dict[str, str] = {}
        self._batches = 0
        self.n_windows = 0
        self.n_actions = 0

    def observe(self, verdicts: list[WindowVerdict],
                descriptor_of: Callable[[str], DeviceDescriptor | None]) -> None:
        s = self.store
        for v in verdicts:
            if v.device_id not in self._known:
                d = descriptor_of(v.device_id)
                if d is not None:
                    s.add_device(v.device_id, d, class_group(d), first_seen=v.t_start)
                self._known.add(v.device_id)
            s.add_windows(v.device_id, [v.t_start], [v.stage1], [v.stage2], [None],
                          [v.features], names=FEATURE_NAMES)
            self.n_windows += 1

            if v.action is not None:
                rule = v.action.rule
                aid = s.add_alert(v.device_id, v.t_start, rule["stage1"], rule["stage2"],
                                  v.action.severity, v.evidence)
                s.add_action(aid, v.device_id, v.t_start, v.action.action, rule)
                self.n_actions += 1
                if self.recorder is not None:
                    # the windows the LSTM saw when it produced this verdict
                    t0 = v.t_start - (self.seq_len - 1) * self.window
                    self.recorder.pin(v.device_id, t0, v.t_start + self.window)

            if self._state.get(v.device_id) != v.trust_state:
                s.set_trust(v.device_id, v.trust_state)
                self._state[v.device_id] = v.trust_state

        self._batches += 1
        if self._batches % self.commit_every == 0:
            s.commit()

    def close(self) -> None:
        self.store.commit()
