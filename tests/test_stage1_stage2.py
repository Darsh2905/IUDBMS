"""Milestones 4 and 5 — Stage 1 scoring and the PyTorch LSTM."""

import numpy as np
import pytest

from iudbms.capture.simulator import Simulator
from iudbms.contract import FEATURE_NAMES, N_FEATURES, DeviceDescriptor, Interface
from iudbms.stage1 import GATE, WARMUP, DeviceBaseline, Stage1, class_group
from iudbms.stage2 import SEQ_LEN, LSTMDetector, TrainConfig, build_sequences


def desc(*classes, late=None):
    ifaces = [Interface(c, first_seen_ts=0.0) for c in classes]
    if late is not None:
        ifaces.append(Interface(late, first_seen_ts=20.0))
    return DeviceDescriptor(bus=1, devnum=3, vid=1, pid=2, interfaces=tuple(ifaces))


# --------------------------------------------------------------------------
# class grouping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("classes,group", [
    ((0x03,), "hid"), ((0x08,), "storage"), ((0x01,), "av"),
    ((0x0E,), "av"), ((0x03, 0x08), "composite"), ((0xFF,), "other"),
])
def test_class_group(classes, group):
    assert class_group(desc(*classes)) == group


def test_class_group_uses_enumeration_time_identity():
    """identity_switch must still be scored against the storage population
    after it grows an HID interface — that is what makes its keystrokes
    anomalous rather than expected."""
    assert class_group(desc(0x08, late=0x03)) == "storage"


# --------------------------------------------------------------------------
# device baseline
# --------------------------------------------------------------------------

def test_baseline_is_silent_during_warmup():
    bl = DeviceBaseline()
    for _ in range(WARMUP - 1):
        bl.update(np.zeros(N_FEATURES))
    assert bl.deviation(np.full(N_FEATURES, 99.0)) == (0.0, [])


def test_baseline_flags_a_departure_after_warmup():
    bl = DeviceBaseline()
    rng = np.random.default_rng(0)
    for _ in range(30):
        bl.update(rng.normal(0, 0.1, N_FEATURES))
    score, top = bl.deviation(np.full(N_FEATURES, 10.0))
    assert score > 3.0
    assert len(top) == 3 and all(isinstance(n, str) for n, _ in top)


def test_baseline_explanation_names_the_deviating_feature():
    bl = DeviceBaseline()
    for _ in range(30):
        bl.update(np.zeros(N_FEATURES))
    x = np.zeros(N_FEATURES)
    x[FEATURE_NAMES.index("cv_iat")] = 50.0
    _, top = bl.deviation(x)
    assert top[0][0] == "cv_iat"


def test_baseline_respects_a_narrower_feature_set():
    """The modality ablations score column subsets; indexing the full 28-name
    list would attach the wrong name to every piece of alert evidence."""
    names = ("mean_iat", "cv_iat", "iat_entropy")
    bl = DeviceBaseline(feature_names=names)
    for _ in range(30):
        bl.update(np.zeros(3))
    x = np.array([0.0, 0.0, 9.0])
    _, top = bl.deviation(x)
    assert top[0][0] == "iat_entropy"


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, N_FEATURES))
    groups = ["hid"] * 200 + ["storage"] * 200
    return Stage1(seed=0).fit(X, groups)


def test_fit_builds_one_forest_per_populated_group(fitted):
    assert {"hid", "storage", "other"} <= set(fitted.forests)


def test_scoring_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        Stage1().score_session(np.zeros((5, N_FEATURES)), desc(0x03))


def test_scores_are_bounded(fitted):
    rng = np.random.default_rng(1)
    s, _, _ = fitted.score_session(rng.normal(0, 1, (60, N_FEATURES)), desc(0x03))
    assert np.all((s >= 0) & (s <= 1))


def test_wrong_feature_width_is_rejected(fitted):
    with pytest.raises(ValueError, match="features"):
        fitted.score_session(np.zeros((5, 9)), desc(0x03))


def test_update_gate_stops_an_attack_teaching_its_own_baseline(fitted):
    """A slow-ramp attack must not drag its baseline along behind it."""
    bl = DeviceBaseline()
    X = np.vstack([np.zeros((20, N_FEATURES)), np.full((40, N_FEATURES), 60.0)])
    scores, _, _ = fitted.score_session(X, desc(0x03), bl)
    assert bl.n <= 20 + WARMUP, "anomalous windows were folded into the baseline"
    assert scores[-1] >= GATE


def test_scoring_is_causal(fitted):
    """Window i must be scored from windows < i only, so offline evaluation
    and live operation see the same thing."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (40, N_FEATURES))
    full, _, _ = fitted.score_session(X, desc(0x03))
    prefix, _, _ = fitted.score_session(X[:20], desc(0x03))
    np.testing.assert_allclose(full[:20], prefix, rtol=1e-9)


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------

def test_sequences_are_causal_and_left_padded():
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    S = build_sequences(X)
    assert S.shape == (20, SEQ_LEN, 3)
    np.testing.assert_array_equal(S[5, -1], X[5])
    np.testing.assert_array_equal(S[0], np.repeat(X[:1], SEQ_LEN, axis=0))
    np.testing.assert_array_equal(S[15], X[6:16])


def test_padding_repeats_rather_than_zeroing():
    """Zero-padding would make a device's first seconds look like a
    discontinuity rather than a short history."""
    X = np.full((3, 4), 7.0, dtype=np.float32)
    assert np.all(build_sequences(X)[0] == 7.0)


def test_lstm_learns_a_separable_transition():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (300, 5)), rng.normal(4, 1, (300, 5))]).astype(np.float32)
    y = np.concatenate([np.zeros(300), np.ones(300)]).astype(np.float32)
    S = build_sequences(X)
    m = LSTMDetector(d_in=5, seed=0)
    hist = m.fit(S, y, cfg=TrainConfig(epochs=6, seed=0), verbose=False)
    assert hist[-1]["loss"] < hist[0]["loss"]
    p = m.predict_proba(S)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3


def test_predictions_are_probabilities():
    S = build_sequences(np.zeros((10, 4), dtype=np.float32))
    p = LSTMDetector(d_in=4, seed=0).predict_proba(S)
    assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))


def test_training_is_deterministic_under_seed():
    rng = np.random.default_rng(3)
    S = build_sequences(rng.normal(0, 1, (120, 4)).astype(np.float32))
    y = (np.arange(120) > 60).astype(np.float32)
    out = []
    for _ in range(2):
        m = LSTMDetector(d_in=4, seed=11)
        m.fit(S, y, cfg=TrainConfig(epochs=3, seed=11), verbose=False)
        out.append(m.predict_proba(S))
    np.testing.assert_allclose(out[0], out[1], rtol=1e-6)


def test_checkpoint_round_trips(tmp_path):
    rng = np.random.default_rng(4)
    S = build_sequences(rng.normal(0, 1, (40, 6)).astype(np.float32))
    m = LSTMDetector(d_in=6, seed=0)
    m.fit(S, (np.arange(40) > 20).astype(np.float32),
          cfg=TrainConfig(epochs=2, seed=0), verbose=False)
    p = tmp_path / "m.pt"
    m.save(p)
    np.testing.assert_allclose(LSTMDetector.load(p).predict_proba(S), m.predict_proba(S),
                               rtol=1e-6)


def test_checkpoint_refuses_a_mismatched_contract_major(tmp_path, monkeypatch):
    m = LSTMDetector(d_in=6, seed=0)
    m.contract_version = "99.0.0"
    p = tmp_path / "m.pt"
    m.save(p)
    with pytest.raises(RuntimeError, match="retrain"):
        LSTMDetector.load(p)


def test_real_session_scores_end_to_end():
    """Smoke test through the actual pipeline types, not synthetic arrays."""
    from iudbms.capture.simulator import SimulatedSource
    from iudbms.features.extractor import WindowFeatureExtractor, window_events

    s = Simulator(seed=5).session("A0001", "hid_injection")
    ex = WindowFeatureExtractor()
    X = np.vstack([ex.extract(b, t0, s.descriptor)
                   for t0, b in window_events(list(SimulatedSource(s).events()),
                                              duration=s.duration)])
    m1 = Stage1(seed=0).fit(X[:20], ["hid"] * 20)
    scores, expl, _ = m1.score_session(X, s.descriptor)
    assert len(scores) == len(X) and np.all(np.isfinite(scores))
    assert len(expl) == len(X)


def test_variance_floor_stops_near_constant_features_exploding():
    """Without a floor, a feature that is constant for a device collapses its
    EWMA variance, so any change produces an enormous |z| and swamps the
    top-3 statistic. That made the device term ~0 after calibration, leaving
    novelty pillar N2 present in the architecture and absent from the
    arithmetic."""
    from iudbms.stage1 import VAR_FLOOR

    bl = DeviceBaseline()
    for _ in range(40):
        bl.update(np.zeros(N_FEATURES))       # perfectly constant device
    assert bl.var.min() < VAR_FLOOR, "test premise: variance really does collapse"

    x = np.zeros(N_FEATURES)
    x[0] = 1.0                                 # a one-unit move in one feature
    score, _ = bl.deviation(x)
    assert score <= 1.0 / np.sqrt(VAR_FLOOR) + 1e-6, (
        f"deviation {score} exceeds the floor-implied bound"
    )


def test_constant_benign_feature_does_not_swamp_the_vector():
    """Several features are exactly constant across benign traffic
    (class_behaviour_mismatch, late_iface_flag, n_interfaces). Dividing by their
    ~zero standard deviation turned a single 0->1 flip into a standardised value
    of ~1e6, which swamped all other dimensions in both the isolation forest and
    the device-baseline z-scores. It surfaced as alert evidence reading
    'late_iface_flag 4472091.23'."""
    from iudbms.stage1 import STD_CLIP

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, N_FEATURES))
    const = FEATURE_NAMES.index("late_iface_flag")
    X[:, const] = 0.0                      # constant across benign traffic
    m = Stage1(seed=0).fit(X, ["hid"] * 300)

    flipped = np.zeros((1, N_FEATURES))
    flipped[0, const] = 1.0
    z = m._std(flipped)
    assert np.all(np.abs(z) <= STD_CLIP + 1e-9), f"standardised value {z.max()} exceeds the clip"
    assert abs(z[0, const]) == pytest.approx(STD_CLIP), (
        "a flip on a benign-constant feature should still read as maximally anomalous"
    )


def test_alert_evidence_z_scores_stay_interpretable():
    """Evidence is shown to an operator; a z of 4.5 million is not a number
    anyone can act on."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, N_FEATURES))
    X[:, FEATURE_NAMES.index("late_iface_flag")] = 0.0
    m = Stage1(seed=0).fit(X, ["hid"] * 300)

    probe = rng.normal(0, 1, (40, N_FEATURES))
    probe[:, FEATURE_NAMES.index("late_iface_flag")] = 0.0
    probe[20:, FEATURE_NAMES.index("late_iface_flag")] = 1.0
    _, expl, _ = m.score_session(probe, desc(0x03))
    worst = max((v for row in expl for _, v in row), default=0.0)
    assert worst < 1000, f"largest reported deviation was {worst}"
