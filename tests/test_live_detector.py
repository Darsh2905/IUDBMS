"""The streaming detector.

The load-bearing test is ``test_streaming_matches_offline_scoring``: if live
scores differ from the offline ones, then either the reported numbers do not
describe the deployed system or the demo is a mock. They must be identical.
"""

import numpy as np
import pytest

from eval.corpus import build_corpus, stratified_split
from iudbms.capture.simulator import SimulatedSource, Simulator
from iudbms.contract import FEATURE_NAMES, N_FEATURES, feature_index
from iudbms.features.extractor import WindowFeatureExtractor, window_events
from iudbms.live import StreamingDetector
from iudbms.policy import PolicyEngine
from iudbms.stage1 import DeviceBaseline, Stage1, class_group
from iudbms.stage2 import LSTMDetector, TrainConfig, build_sequences

DEPLOYED = np.array([i for i in range(N_FEATURES)
                     if i != feature_index("log_time_since_enum")])


@pytest.fixture(scope="module")
def trained():
    """A small but real pipeline: Stage 1 + LSTM fit on simulated benign data."""
    corpus = build_corpus(n_benign=24, n_attack=16, seed=7, verbose=False)
    fit, val, test = stratified_split(corpus.sessions, seed=7)
    names = tuple(FEATURE_NAMES[i] for i in DEPLOYED)

    Xb = np.vstack([s.X[:, DEPLOYED] for s in fit if s.label == 0])
    groups = [class_group(s.descriptor) for s in fit if s.label == 0
              for _ in range(len(s.X))]
    s1 = Stage1(seed=7, feature_names=names).fit(Xb, groups)
    s1.calibrate_dev(np.concatenate(
        [s1.score_session(s.X[:, DEPLOYED], s.descriptor)[2]
         for s in fit if s.label == 0]))

    seqs, ys = [], []
    for s in fit:
        sc, _, _ = s1.score_session(s.X[:, DEPLOYED], s.descriptor)
        seqs.append(build_sequences(np.hstack([s.X[:, DEPLOYED], sc.reshape(-1, 1)])))
        ys.append(s.y)
    lstm = LSTMDetector(d_in=len(DEPLOYED) + 1, seed=7)
    lstm.fit(np.concatenate(seqs), np.concatenate(ys).astype(np.float32),
             cfg=TrainConfig(epochs=3, seed=7), verbose=False)
    return s1, lstm, test


def _offline(s1, lstm, session):
    """Score a session the way eval/pipeline.py does."""
    sc, _, _ = s1.score_session(session.X[:, DEPLOYED], session.descriptor)
    seq = build_sequences(np.hstack([session.X[:, DEPLOYED], sc.reshape(-1, 1)]))
    return sc, lstm.predict_proba(seq)


def _stream(s1, lstm, session, sim_session):
    """Score the same session incrementally, one window at a time."""
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.9, k=4, n=6),
                            feature_idx=DEPLOYED)
    det.attach(session.descriptor, session.sid)
    events = list(SimulatedSource(sim_session).events())
    out1, out2 = [], []
    for t0, bucket in window_events(events, duration=sim_session.duration):
        v = det.observe_window(session.sid, t0, bucket)
        out1.append(v.stage1); out2.append(v.stage2)
    return np.array(out1), np.array(out2)


def test_streaming_matches_offline_scoring(trained):
    """If these diverge, the reported numbers do not describe the live system."""
    s1, lstm, test = trained
    sim = Simulator(seed=7)
    # rebuild the exact session the corpus produced, so features match
    sess = test[0]
    sim_sess = None
    probe = Simulator(seed=7)
    for sid, prof in probe.plan(n_benign=24, n_attack=16):
        s = probe.session(sid, prof)
        if sid == sess.sid:
            sim_sess = s
            break
    assert sim_sess is not None, "could not rebuild the session"

    off1, off2 = _offline(s1, lstm, sess)
    liv1, liv2 = _stream(s1, lstm, sess, sim_sess)

    n = min(len(off1), len(liv1))
    np.testing.assert_allclose(liv1[:n], off1[:n], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(liv2[:n], off2[:n], rtol=1e-5, atol=1e-5)


def test_attach_is_required(trained):
    s1, lstm, _ = trained
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.9), feature_idx=DEPLOYED)
    with pytest.raises(KeyError, match="attach"):
        det.observe_window("nope", 0.0, [])


def test_attach_is_idempotent(trained):
    s1, lstm, test = trained
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.9), feature_idx=DEPLOYED)
    a = det.attach(test[0].descriptor, "d1")
    b = det.attach(test[0].descriptor, "d1")
    assert a is b and len(det.devices) == 1


def test_detach_forgets_device_history(trained):
    """A new device on the same bus address must not inherit the old one's
    payload history, or it reads as suspiciously familiar traffic."""
    s1, lstm, test = trained
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.9), feature_idx=DEPLOYED)
    det.attach(test[0].descriptor, "1-2")
    det.observe_window("1-2", 0.0, [])
    det.detach("1-2")
    assert "1-2" not in det.devices
    det.attach(test[0].descriptor, "1-2")
    assert det.devices["1-2"].n_windows == 0
    assert not det.devices["1-2"].extractor._seen_bigrams


def test_verdict_reports_gate_progress(trained):
    """The operator view needs to show the gate filling, not just its result."""
    s1, lstm, test = trained
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.0, k=4, n=6),
                            feature_idx=DEPLOYED)
    det.attach(test[0].descriptor, "d1")
    hits = [det.observe_window("d1", float(i), []).gate_hits for i in range(8)]
    assert hits[0] == 1 and max(hits) == 6, hits


def test_devices_are_scored_independently(trained):
    """Two devices monitored at once must not share baselines or history."""
    s1, lstm, test = trained
    det = StreamingDetector(s1, lstm, PolicyEngine(threshold=0.9), feature_idx=DEPLOYED)
    det.attach(test[0].descriptor, "a")
    det.attach(test[1].descriptor, "b")
    for i in range(6):
        det.observe_window("a", float(i), [])
    assert det.devices["a"].n_windows == 6
    assert det.devices["b"].n_windows == 0


def test_stage1_only_mode(trained):
    """With no LSTM the detector falls back to Stage 1, for the ablation path."""
    s1, _, test = trained
    det = StreamingDetector(s1, None, PolicyEngine(threshold=0.9), feature_idx=DEPLOYED)
    det.attach(test[0].descriptor, "d1")
    v = det.observe_window("d1", 0.0, [])
    assert v.stage2 == v.stage1


# --------------------------------------------------------------------------
# human-capability rule: live == the evaluated OR of two separate gates
# --------------------------------------------------------------------------

def test_capability_rule_live_matches_the_offline_or_of_two_gates():
    """eval.analysis_hybrid scores model OR rule as two separate 4-of-6 gates; the live
    detector must fire at exactly the same window, or the shipped rule is not the
    measured one."""
    import json
    from pathlib import Path

    import numpy as np

    from eval.analysis_common import deployed, featurize, run_gate, score_with
    from eval.analysis_hybrid import combine, rule_scores
    from eval.analysis_realistic import RealisticSim
    from iudbms.capture.simulator import SimulatedSource
    from iudbms.features.extractor import window_events
    from iudbms.live import StreamingDetector
    from iudbms.policy import PolicyEngine

    cfg, s1, lstm = deployed()
    idx = np.array(cfg["feature_idx"])
    sim = RealisticSim(seed=21)
    for prof in ("hid_injection", "keyboard", "mimicry", "identity_switch"):
        ss = sim.session("x", prof, duration=60.0)
        sess = featurize(ss, "x", prof)
        _, S2, _ = score_with(s1, lstm, [sess], idx, cfg["seq_len"])
        offline = combine(run_gate(S2, [sess], cfg["threshold"]),
                          run_gate(rule_scores([sess]), [sess], 0.5))["x"]
        det = StreamingDetector(s1, lstm, PolicyEngine(cfg["threshold"], cfg["k"], cfg["n"]),
                                feature_idx=idx, seq_len=cfg["seq_len"], human_max_rps=30)
        det.attach(ss.descriptor, "x")
        live = None
        for t0, b in window_events(list(SimulatedSource(ss).events()), duration=ss.duration):
            v = det.observe_window("x", t0, b)
            if v.action is not None and live is None:
                live = t0 + 1.0
        assert live == offline, (prof, live, offline)


def test_rule_flags_alone_need_their_own_k_of_n():
    from iudbms.contract import DeviceDescriptor, Endpoint, Interface
    from iudbms.policy import PolicyEngine
    kbd = DeviceDescriptor(bus=1, devnum=3, vid=1, pid=1,
                           interfaces=(Interface(3, 1, 1, (Endpoint(0x81, "intr", 10),)),))
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("k")
    # two model windows and two rule windows interleaved: neither gate reaches 4
    seq = [(0.95, False), (0.1, True), (0.95, False), (0.1, True), (0.1, False), (0.1, False)]
    assert all(eng.step(ctx, float(i), 0, s, [], kbd, rule_hit=r) is None
               for i, (s, r) in enumerate(seq))
    ctx = eng.new_device("k2")
    fired = [eng.step(ctx, float(i), 0, 0.1, [], kbd, rule_hit=True) for i in range(4)]
    assert fired[-1] is not None and fired[-1].rule["fired_by"] == "capability rule"
