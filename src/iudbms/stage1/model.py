"""
L3 Stage 1 — memoryless anomaly scoring (novelty pillar N2).

Two components are fused per window:

  (a) **Population model** — an Isolation Forest trained on benign windows only,
      conditioned on the device's *declared* class group. Answers: "is this
      window unusual for anything of this claimed type?" A single global model
      conflates a 125 Hz mouse with a human typist, so the class conditioning is
      doing real work.

  (b) **Device model** — an EWMA mean/variance baseline learned online from the
      device's own early windows. Answers: "is this unusual for *this* device?"
      Deviation is the mean of the three largest |z| scores, which also yields a
      free explanation: those three feature names ship with every alert and are
      what makes the generated rule auditable (pillar N4).

The device baseline is **update-gated**: a window already scoring above ``GATE``
is not folded back in, so a slow-ramp attack cannot drag its own baseline along
behind it.

Class group is taken from the interfaces present **at enumeration** (t=0), not
from whatever the device eventually exposes. That is deliberate: it is what a
defender knows at authorisation time, and it means an ``identity_switch``
device that later grows an HID interface is still scored against the storage
population — which is exactly what makes its injected keystrokes anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ..contract import (
    CLASS_AUDIO,
    CLASS_HID,
    CLASS_MASS_STORAGE,
    CLASS_VIDEO,
    FEATURE_NAMES,
    N_FEATURES,
    DeviceDescriptor,
)

__all__ = ["Stage1", "DeviceBaseline", "class_group", "WARMUP", "GATE",
           "VAR_FLOOR", "SD_FLOOR", "STD_CLIP"]

WARMUP = 4        # windows before the device baseline is trusted at all
GATE = 0.55       # windows scoring above this do not update the baseline
EPS = 1e-6

#: Lower bound on a feature's EWMA variance, in *standardised* (population)
#: units. Without it the per-device baseline is effectively inert.
#:
#: Many features are near-constant for a given device — a keyboard's frac_iso
#: is always 0, has_hid always 1, frac_ctrl is 0 except during enumeration.
#: Their EWMA variance collapses toward zero, so any change at all produces an
#: enormous |z|. Measured on a 200-session corpus, the benign device-deviation
#: had median 0.8 but a 99th percentile of 90.8 and a maximum of 1267. Since
#: calibrate_dev() divides by that 99th percentile, the 0.45-weighted device
#: term contributed ~0.009 to a score in [0, 1] — novelty pillar N2 was
#: present in the architecture and absent from the arithmetic.
#:
#: 0.05 says: never claim a device is more stable than 5% of the population
#: variance for a feature. Swept over {0, 0.01, 0.05, 0.1, 0.25, 0.5}; 0.05
#: maximised Stage-1 AUC (0.8922 vs 0.8900 unfloored) and cut the benign
#: window FPR at the operating threshold from 0.26% to 0.07%.
VAR_FLOOR = 0.05

#: Floor on the population standard deviation used to standardise features,
#: and a clip on the result. Same failure as VAR_FLOOR, one level up.
#:
#: Several features are *exactly* constant across benign traffic --
#: class_behaviour_mismatch, late_iface_flag and n_interfaces are all zero or
#: one for every benign device, by construction. Dividing by ``sd + 1e-6`` then
#: turns a single 0->1 flip into a standardised value of ~1e6, which swamps all
#: 29 dimensions in both the isolation forest and the device-baseline z-scores.
#: It surfaced as alert evidence reading "late_iface_flag 4472091.23".
#:
#: A feature that is constant for benign traffic IS informative when it changes
#: -- that is the whole point of class_behaviour_mismatch -- so the fix is to
#: bound its influence, not remove it. With the floor, a binary flip is ~100
#: sigma, clipped to 10: still decisively anomalous, no longer the only thing
#: the model can see.
SD_FLOOR = 1e-2
STD_CLIP = 10.0
MIN_GROUP_WINDOWS = 30    # below this a group folds into the generic forest


def class_group(descriptor: DeviceDescriptor) -> str:
    """Coarse device family from the interfaces declared at enumeration."""
    classes = descriptor.classes_at(0.0)
    if CLASS_HID in classes and CLASS_MASS_STORAGE in classes:
        return "composite"
    if CLASS_HID in classes:
        return "hid"
    if CLASS_MASS_STORAGE in classes:
        return "storage"
    if classes & {CLASS_AUDIO, CLASS_VIDEO}:
        return "av"
    return "other"


class DeviceBaseline:
    """Online EWMA mean/variance per feature for one physical device.

    ``feature_names`` must match the width of the vectors being scored. It is
    not cosmetic: the modality ablations score 6-9 column subsets, and indexing
    the full 28-name list would attach the wrong feature name to every piece of
    alert evidence.
    """

    __slots__ = ("alpha", "mean", "var", "n", "names")

    def __init__(self, alpha: float = 0.15, feature_names: tuple[str, ...] | None = None):
        self.alpha = alpha
        self.names = tuple(feature_names) if feature_names is not None else FEATURE_NAMES
        d = len(self.names)
        self.mean = np.zeros(d)
        self.var = np.ones(d)
        self.n = 0

    def deviation(self, x: np.ndarray, k: int = 3) -> tuple[float, list[tuple[str, float]]]:
        """Mean of the ``k`` largest |z| scores, plus which features they were."""
        if self.n < WARMUP:
            return 0.0, []
        z = np.abs(x - self.mean) / np.sqrt(np.maximum(self.var, VAR_FLOOR) + EPS)
        top = np.argsort(z)[::-1][:k]
        return float(z[top].mean()), [(self.names[i], round(float(z[i]), 2)) for i in top]

    def update(self, x: np.ndarray) -> None:
        if self.n == 0:
            self.mean = x.astype(np.float64, copy=True)
            self.var = np.full(len(self.names), 0.25)
        else:
            d = x - self.mean
            self.mean += self.alpha * d
            self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)
        self.n += 1


class Stage1:
    """Class-conditioned Isolation Forests fused with a per-device baseline."""

    def __init__(self, seed: int = 0, n_estimators: int = 200, contamination: float = 0.02,
                 feature_names: tuple[str, ...] | None = None):
        self.forests: dict[str, IsolationForest] = {}
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.scale = {"if": 1.0, "dev": 1.0}
        self.seed = seed
        self.n_estimators = n_estimators
        self.contamination = contamination
        # None until fit(); the modality ablations fit on column subsets, and
        # the baseline needs the matching names to label alert evidence.
        self.feature_names: tuple[str, ...] | None = (
            tuple(feature_names) if feature_names is not None else None
        )

    def _std(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mu) / self.sd, -STD_CLIP, STD_CLIP)

    def fit(self, X_benign: np.ndarray, groups_benign) -> Stage1:
        """Fit on benign windows only — Stage 1 needs no attack labels."""
        groups = np.asarray(groups_benign)
        X_benign = np.asarray(X_benign, dtype=np.float64)
        if self.feature_names is None:
            self.feature_names = (
                FEATURE_NAMES if X_benign.shape[1] == N_FEATURES
                else tuple(f"f{i}" for i in range(X_benign.shape[1]))
            )
        elif len(self.feature_names) != X_benign.shape[1]:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but X has "
                f"{X_benign.shape[1]} columns"
            )
        self.mu = X_benign.mean(axis=0)
        self.sd = np.maximum(X_benign.std(axis=0), SD_FLOOR)
        Z = self._std(X_benign)

        for g in sorted(set(groups.tolist())):
            m = groups == g
            if m.sum() < MIN_GROUP_WINDOWS:
                continue
            self.forests[g] = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=min(512, int(m.sum())),
                contamination=self.contamination,
                random_state=self.seed,
            ).fit(Z[m])

        # Generic fallback for unseen or under-represented groups.
        self.forests["other"] = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(512, len(Z)),
            contamination=self.contamination,
            random_state=self.seed,
        ).fit(Z)

        raw = np.empty(len(Z))
        for g in set(groups.tolist()):
            m = groups == g
            raw[m] = self._raw_if(X_benign[m], g)
        p99 = float(np.percentile(raw, 99))
        self.scale["if"] = p99 if p99 > 0 else 1.0
        return self

    def _raw_if(self, X: np.ndarray, group: str) -> np.ndarray:
        """Batched: one forest call per session rather than per window."""
        f = self.forests.get(group, self.forests["other"])
        return -f.score_samples(self._std(np.atleast_2d(X)))

    def calibrate_dev(self, dev_raw_benign: np.ndarray) -> None:
        """Scale the device-deviation term by its benign 99th percentile."""
        v = float(np.percentile(dev_raw_benign, 99)) if len(dev_raw_benign) else 1.0
        self.scale["dev"] = v if v > 0 else 1.0

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        import pickle
        from pathlib import Path

        from ..contract import CONTRACT_VERSION
        Path(path).write_bytes(pickle.dumps({
            "contract_version": CONTRACT_VERSION,
            "forests": self.forests, "mu": self.mu, "sd": self.sd,
            "scale": self.scale, "seed": self.seed,
            "feature_names": self.feature_names,
        }))

    @classmethod
    def load(cls, path) -> "Stage1":
        import pickle
        from pathlib import Path

        from ..contract import CONTRACT_VERSION
        blob = pickle.loads(Path(path).read_bytes())
        got = blob.get("contract_version")
        if got and got.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise RuntimeError(
                f"Stage 1 was fit against contract {got}, running {CONTRACT_VERSION}. "
                f"The feature vector changed width; retrain rather than load."
            )
        m = cls(seed=blob["seed"], feature_names=blob["feature_names"])
        m.forests, m.mu, m.sd, m.scale = (
            blob["forests"], blob["mu"], blob["sd"], blob["scale"])
        return m

    def score_session(
        self,
        X: np.ndarray,
        descriptor: DeviceDescriptor,
        baseline: DeviceBaseline | None = None,
    ) -> tuple[np.ndarray, list[list[tuple[str, float]]], np.ndarray]:
        """Score one device session window by window.

        Returns ``(scores, explanations, raw_device_deviations)``. Scoring is
        strictly causal: window *i* is scored using a baseline built from
        windows < *i* only, so this is valid for live use as well as offline
        evaluation.
        """
        if self.mu is None:
            raise RuntimeError("Stage1.fit() must be called before scoring")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.mu):
            raise ValueError(
                f"model was fit on {len(self.mu)} features, got {X.shape[1]}"
            )
        g = class_group(descriptor)
        bl = baseline if baseline is not None else DeviceBaseline(
            feature_names=self.feature_names)
        Z = self._std(X)                       # baselines live in standardised space
        raw_if = self._raw_if(X, g)

        out = np.empty(len(X))
        expl: list[list[tuple[str, float]]] = []
        dev_raw = np.empty(len(X))

        for i, (x, a_raw) in enumerate(zip(Z, raw_if)):
            a = a_raw / (self.scale["if"] + EPS)
            d, top = bl.deviation(x)
            dev_raw[i] = d
            b = d / (self.scale["dev"] + EPS)
            s = float(np.clip(0.55 * np.clip(a, 0, 3) / 3 + 0.45 * np.clip(b, 0, 3) / 3, 0, 1))
            # Update-gate: an already-anomalous window must not teach the
            # baseline that it is normal.
            if s < GATE or bl.n < WARMUP:
                bl.update(x)
            out[i] = s
            expl.append(top)
        return out, expl, dev_raw
