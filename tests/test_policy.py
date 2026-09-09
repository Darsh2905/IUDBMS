"""Milestone 6 — policy engine and its safety interlocks.

The interlock tests matter more than the ladder tests: writing 0 to a device's
``authorized`` node can detach the keyboard being typed on, so every one of
these is a guard against bricking a machine.
"""

import pytest

from iudbms.capture.simulator import Simulator
from iudbms.policy import ENFORCEMENT_REVIEWED, KILL_SWITCH_ENV, EnforcementBlocked, PolicyEngine

DESC = Simulator(seed=1).session("A0001", "hid_injection").descriptor
EV = [("cv_iat", 1.8)]


def drive(eng, ctx, scores):
    return [a for i, s in enumerate(scores)
            if (a := eng.step(ctx, float(i), 0.2, s, EV, DESC)) is not None]


def test_single_anomalous_window_never_acts():
    """The k-of-n gate is the main false-positive control."""
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    assert drive(eng, ctx, [0.99] + [0.1] * 20) == []
    assert ctx.state == "observe"


def test_k_of_n_requires_k_hits_inside_the_window():
    """Four hits spread across more than n windows must not trigger."""
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    spread = [0.99, 0.1, 0.1, 0.99, 0.1, 0.1, 0.99, 0.1, 0.1, 0.99]
    assert drive(eng, ctx, spread) == []


def test_escalation_ladder_observe_throttle_deauthorize():
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    acts = drive(eng, ctx, [0.99] * 20)
    assert [a.action for a in acts] == ["THROTTLE", "DEAUTHORIZE"]
    assert [a.severity for a in acts] == ["high", "critical"]
    assert ctx.state == "revoked"


def test_deauthorize_is_emitted_once():
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    acts = drive(eng, ctx, [0.99] * 60)
    assert sum(a.action == "DEAUTHORIZE" for a in acts) == 1


def test_invalid_gate_rejected():
    with pytest.raises(ValueError):
        PolicyEngine(threshold=0.5, k=7, n=6)
    with pytest.raises(ValueError):
        PolicyEngine(threshold=0.5, k=0, n=6)


# --------------------------------------------------------------------------
# generated rule — the pillar N4 artefact
# --------------------------------------------------------------------------

def test_rule_carries_learned_trigger_and_evidence():
    """What makes enforcement *learned* rather than a static table entry."""
    eng = PolicyEngine(threshold=0.942, k=4, n=6)
    ctx = eng.new_device("A0001")
    rule = drive(eng, ctx, [0.99] * 10)[0].rule
    assert rule["trigger"] == "stage2>=0.942 in 4/6 windows"
    assert rule["evidence_features"] == EV
    assert rule["declared_class"] == ["0x3"]
    assert rule["dry_run"] is True
    assert rule["enforcement_reviewed"] is False


def test_deauthorize_rule_names_the_sysfs_node():
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    deauth = [a for a in drive(eng, ctx, [0.99] * 20) if a.action == "DEAUTHORIZE"][0]
    assert deauth.rule["kernel_cmd"] == (
        f"echo 0 > /sys/bus/usb/devices/{DESC.bus}-{DESC.devnum}/authorized")


# --------------------------------------------------------------------------
# safety interlocks
# --------------------------------------------------------------------------

def test_interlock_1_dry_run_is_the_default():
    assert PolicyEngine(threshold=0.9).dry_run is True


def test_interlock_2_allowlisted_device_is_never_acted_on():
    """Revoking the boot keyboard locks the operator out of the machine."""
    eng = PolicyEngine(threshold=0.9, k=4, n=6, allowlist={"boot-kbd"})
    ctx = eng.new_device("boot-kbd")
    assert drive(eng, ctx, [0.99] * 40) == []
    assert ctx.state == "observe"


def test_interlock_3_kill_switch_suppresses_all_actions(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    assert drive(eng, eng.new_device("A0001"), [0.99] * 40) == []


@pytest.mark.parametrize("value,engaged", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("no", False),
])
def test_kill_switch_accepts_the_usual_spellings(monkeypatch, value, engaged):
    monkeypatch.setenv(KILL_SWITCH_ENV, value)
    assert PolicyEngine.kill_switch_engaged() is engaged


def test_interlock_4_real_enforcement_is_disabled():
    """MUST stay disabled until the safety checks have been reviewed."""
    assert ENFORCEMENT_REVIEWED is False, (
        "Real sysfs writes must not be enabled until the interlocks are reviewed "
        "and the target is a snapshotted VM."
    )


def test_execute_refuses_to_write_even_when_dry_run_is_off():
    eng = PolicyEngine(threshold=0.9, k=4, n=6, dry_run=False)
    ctx = eng.new_device("A0001")
    act = drive(eng, ctx, [0.99] * 20)[-1]
    with pytest.raises(EnforcementBlocked, match="ENFORCEMENT_REVIEWED"):
        eng.execute(act)


def test_execute_in_dry_run_only_describes_the_command():
    eng = PolicyEngine(threshold=0.9, k=4, n=6)
    ctx = eng.new_device("A0001")
    act = drive(eng, ctx, [0.99] * 20)[-1]
    out = eng.execute(act)
    assert out.startswith("DRY-RUN:")
    assert "/authorized" in out


def test_execute_reports_the_kill_switch_before_anything_else(monkeypatch):
    eng = PolicyEngine(threshold=0.9, k=4, n=6, dry_run=False)
    ctx = eng.new_device("A0001")
    act = drive(eng, ctx, [0.99] * 20)[-1]
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    with pytest.raises(EnforcementBlocked, match="kill switch"):
        eng.execute(act)
