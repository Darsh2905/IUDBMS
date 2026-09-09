"""
IUDBMS - Stage 1: instantaneous anomaly scoring.

Novelty pillar 2: per-device adaptive baselines.

Two components are fused per window:

  (a) population model - an Isolation Forest trained on benign windows only,
      conditioned on the device's *declared* class group. Answers: "is this
      window unusual for anything of this claimed type?"

  (b) device model - an EWMA mean/variance baseline learned online from the
      device's own first windows. Answers: "is this unusual for *this* device?"
      Deviation is the mean of the three largest |z| scores, which also gives
      a free explanation (the three features are reported with every alert).

The device baseline is update-gated: windows already scoring above `gate` are
not folded back in, so a slow-ramp attack cannot drag its own baseline along.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from .features import FEATURE_NAMES, N_FEATURES

WARMUP = 4          # windows before the device baseline is trusted
GATE = 0.55         # windows scoring above this do not update the baseline
EPS = 1e-6


def class_group(descriptor):
    classes = {i["bInterfaceClass"] for i in descriptor["interfaces"]}
    if 0x03 in classes and 0x08 in classes:
        return "composite"
    if 0x03 in classes:
        return "hid"
    if 0x08 in classes:
        return "storage"
    if classes & {0x01, 0x0E}:
        return "av"
    return "other"


class DeviceBaseline:
    """Online EWMA mean/variance per feature for a single physical device."""

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.mean = np.zeros(N_FEATURES)
        self.var = np.ones(N_FEATURES)
        self.n = 0

    def deviation(self, x, k=3):
        if self.n < WARMUP:
            return 0.0, []
        z = np.abs(x - self.mean) / np.sqrt(self.var + EPS)
        top = np.argsort(z)[::-1][:k]
        return float(z[top].mean()), [(FEATURE_NAMES[i], round(float(z[i]), 2)) for i in top]

    def update(self, x):
        if self.n == 0:
            self.mean = x.copy()
            self.var = np.full(N_FEATURES, 0.25)
        else:
            d = x - self.mean
            self.mean += self.alpha * d
            self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)
        self.n += 1


class Stage1:
    def __init__(self, seed=0):
        self.forests = {}
        self.mu = None
        self.sd = None
        self.scale = {"if": 1.0, "dev": 1.0}
        self.seed = seed

    def _std(self, X):
        return (X - self.mu) / (self.sd + EPS)

    def fit(self, X_benign, groups_benign):
        groups_benign = np.asarray(groups_benign)
        self.mu = X_benign.mean(axis=0)
        self.sd = X_benign.std(axis=0)
        Z = self._std(X_benign)
        for g in sorted(set(groups_benign)):
            m = np.array([gg == g for gg in groups_benign])
            if m.sum() < 30:                      # too few - fold into 'other'
                continue
            self.forests[g] = IsolationForest(
                n_estimators=200, max_samples=min(512, int(m.sum())),
                contamination=0.02, random_state=self.seed).fit(Z[m])
        self.forests["other"] = IsolationForest(
            n_estimators=200, max_samples=min(512, len(Z)),
            contamination=0.02, random_state=self.seed).fit(Z)

        raw = np.empty(len(Z))
        for g in set(groups_benign):
            m = groups_benign == g
            raw[m] = self._raw_if(X_benign[m], g)
        self.scale["if"] = float(np.percentile(raw, 99)) or 1.0
        return self

    def _raw_if(self, X, group):
        """Batched: one forest call per session instead of per window."""
        f = self.forests.get(group, self.forests["other"])
        return -f.score_samples(self._std(np.atleast_2d(X)))

    def calibrate_dev(self, dev_raw_benign):
        v = np.percentile(dev_raw_benign, 99) if len(dev_raw_benign) else 1.0
        self.scale["dev"] = float(v) if v > 0 else 1.0

    def score_session(self, X, descriptor, baseline=None):
        """Returns (scores [W], explanations [W]) for one device session."""
        g = class_group(descriptor)
        bl = baseline or DeviceBaseline()
        Z = self._std(X)                      # baselines live in standardised space
        raw_if = self._raw_if(X, g)
        out, expl, dev_raw = [], [], []
        for x, a_raw in zip(Z, raw_if):
            a = a_raw / (self.scale["if"] + EPS)
            d, top = bl.deviation(x)
            dev_raw.append(d)
            b = d / (self.scale["dev"] + EPS)
            s = float(np.clip(0.55 * np.clip(a, 0, 3) / 3 + 0.45 * np.clip(b, 0, 3) / 3, 0, 1))
            if s < GATE or bl.n < WARMUP:
                bl.update(x)
            out.append(s)
            expl.append(top)
        return np.array(out), expl, np.array(dev_raw)
