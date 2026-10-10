"""The critical-threshold shortcut equals running the real gate."""

import numpy as np

from eval.analysis_calibration import critical_threshold
from iudbms.contract import DeviceDescriptor
from iudbms.policy import PolicyEngine

DESC = DeviceDescriptor(bus=1, devnum=2, vid=1, pid=1)


def fires(scores, t, k=4, n=6):
    eng = PolicyEngine(threshold=t, k=k, n=n)
    ctx = eng.new_device("d")
    return any(eng.step(ctx, float(i), 0.0, float(s), [], DESC) for i, s in enumerate(scores))


def test_critical_threshold_matches_the_policy_engine():
    rng = np.random.default_rng(0)
    for trial in range(300):
        L = int(rng.integers(1, 40))
        s = rng.beta(0.5, 0.5, L) if trial % 2 else rng.random(L) ** 4
        tau = critical_threshold(s, 4, 6)
        for t in np.concatenate([s, [0.0, 1.0, 0.5]]):
            assert fires(s, t) == (t <= tau), (trial, L, t, tau)
