"""The checkpoint set in out/models/ must belong together.

``deploy.json`` carries the decision threshold, which was calibrated on one
specific Stage-2 model's validation scores. The first deployment shipped a
Stage-2 file from a *different* run beside that threshold: it loaded, scored real
traffic, passed every structural check -- and detected 0 of 120 simulated attacks,
because its scores never reach a threshold calibrated for another model's scale.
Retraining with the documented command reproduced the reported 89.2% exactly, so
the evaluation was sound and the shipped file was not the evaluated one.

Nothing could tell, because every file was individually valid. These tests make
the *set* checkable without needing the corpus.
"""

import json
from pathlib import Path

import pytest

from iudbms.live.cli import verify_canary
from iudbms.stage1 import Stage1
from iudbms.stage2 import LSTMDetector

MODELS = Path(__file__).resolve().parent.parent / "out" / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "deploy.json").exists(),
                                reason="no trained models (run `make eval`)")


def _load():
    cfg = json.loads((MODELS / "deploy.json").read_text())
    return cfg, Stage1.load(MODELS / "stage1.pkl"), LSTMDetector.load(MODELS / "stage2.pt")


def test_shipped_models_match_their_deploy_json():
    cfg, s1, s2 = _load()
    assert cfg.get("canary"), (
        "deploy.json carries no canary, so nothing ties these weights to the "
        "threshold beside them. Regenerate the set with `make eval`.")
    verify_canary(cfg, s1, s2, MODELS)        # raises SystemExit on a mismatch


def test_canary_is_deterministic():
    _, s1, s2 = _load()
    assert s2.canary(10) == s2.canary(10)
    assert s1.canary() == s1.canary()


def test_a_stage2_from_another_run_is_refused():
    """The bug itself: same shape, different weights."""
    cfg, s1, s2 = _load()
    other = LSTMDetector(d_in=s2.d_in, hidden=s2.hidden, seed=999)
    assert other.canary(10) != s2.canary(10)
    with pytest.raises(SystemExit, match="does not match deploy.json"):
        verify_canary(cfg, s1, other, MODELS)


def test_a_stage1_from_another_run_is_refused():
    import numpy as np
    cfg, s1, s2 = _load()
    other = Stage1(seed=1, feature_names=s1.feature_names)
    X = np.random.default_rng(3).normal(size=(400, len(s1.mu)))
    other.fit(X, ["hid"] * 400)
    with pytest.raises(SystemExit, match="does not match deploy.json"):
        verify_canary(cfg, other, s2, MODELS)


# --------------------------------------------------------------------------
# the property that actually broke: can the shipped set FIRE?
# --------------------------------------------------------------------------

def _stream(profile: str, seed: int):
    """One simulated session through the real streaming detector, shipped models."""
    import numpy as np

    from iudbms.capture.simulator import SimulatedSource, Simulator
    from iudbms.features.extractor import window_events
    from iudbms.live import StreamingDetector
    from iudbms.policy import PolicyEngine

    cfg, s1, s2 = _load()
    det = StreamingDetector(
        s1, s2, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"]),
        feature_idx=np.array(cfg["feature_idx"]), seq_len=cfg["seq_len"])
    sess = Simulator(seed=seed).session("T", profile)
    det.attach(sess.descriptor, "T")
    events = list(SimulatedSource(sess).events())
    actions = []
    for t0, bucket in window_events(events, duration=sess.duration):
        v = det.observe_window("T", t0, bucket)
        if v.action is not None:
            actions.append((t0, v.action.action))
    return sess, actions


@pytest.mark.parametrize("profile", ["hid_injection", "class_exfil", "staged_delay"])
def test_shipped_models_alert_on_a_simulated_attack(profile):
    """End to end, through the streaming detector, with the shipped checkpoint.

    The bug this guards: a Stage-2 file whose scores never reached the threshold
    beside it. It loaded, scored, persisted and reported healthy -- and could not
    raise a single alert. These three classes are detected 100% by the evaluated
    system, so the shipped set must catch them; the seed is fixed so a failure is
    a regression, not noise.
    """
    sess, actions = _stream(profile, seed=11)
    assert actions, (f"{profile}: the shipped models raised no alert on a simulated attack "
                     f"(onset {sess.onset:.1f}s) -- the deployed system cannot fire")
    assert actions[0][0] >= sess.onset - 1.0, "alerted before the attack began"


def test_shipped_models_stay_quiet_on_simulated_benign_devices():
    for profile in ("mouse", "webcam", "mass_storage", "audio"):
        _, actions = _stream(profile, seed=11)
        assert not actions, f"{profile}: false alarm {actions}"


# --------------------------------------------------------------------------
# the property finding 32 is about: an IDLE keyboard must not be an attack
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,kw", [
    ("idle after enumeration", dict(fully_idle=True)),
    ("gaps 8-20 s", dict(idle=(8.0, 20.0))),
    ("gaps 20-60 s", dict(idle=(20.0, 60.0))),
])
def test_shipped_models_do_not_flag_idle_keyboards_or_mice(name, kw):
    """Corpus A's benign devices were always active, so the model it produced flagged
    90-100% of keyboards and mice the moment they had realistic idle gaps. The
    deployed set is trained on duty-cycled traffic; this keeps it that way."""
    import numpy as np

    from eval.analysis_idle import DutyCycleSim
    from iudbms.capture.simulator import SimulatedSource
    from iudbms.features.extractor import window_events
    from iudbms.live import StreamingDetector
    from iudbms.policy import PolicyEngine

    cfg, s1, s2 = _load()
    flagged = 0
    for i, prof in enumerate(("keyboard", "mouse") * 6):
        det = StreamingDetector(
            s1, s2, PolicyEngine(threshold=cfg["threshold"], k=cfg["k"], n=cfg["n"]),
            feature_idx=np.array(cfg["feature_idx"]), seq_len=cfg["seq_len"])
        sim = DutyCycleSim(seed=500 + i, **kw)
        sess = sim.session(f"I{i}", prof)
        det.attach(sess.descriptor, f"I{i}")
        for t0, b in window_events(list(SimulatedSource(sess).events()), duration=sess.duration):
            if det.observe_window(f"I{i}", t0, b).action is not None:
                flagged += 1
                break
    assert flagged == 0, f"{flagged} of 12 idle ({name}) keyboards/mice were flagged"
