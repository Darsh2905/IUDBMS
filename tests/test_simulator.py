"""The simulator, and the leakage audit that polices it.

The most important test here is ``test_audit_catches_a_reintroduced_duration_leak``:
an audit that cannot fail is worse than no audit, because it manufactures
confidence. It re-introduces the exact Sprint-0 defect and asserts the audit
flags it.
"""

import collections

import numpy as np
import pytest

from eval.leakage_audit import audit, ks_statistic
from iudbms.capture.simulator import (
    ALL_PROFILES,
    ATTACK_PROFILES,
    BENIGN_PROFILES,
    DURATION_RANGE,
    SimulatedSource,
    Simulator,
)
from iudbms.contract import EventSource
from iudbms.features.extractor import WindowFeatureExtractor, window_events

EX = WindowFeatureExtractor()


# --------------------------------------------------------------------------
# usbmon fidelity — the simulator must emit what the kernel emits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_every_profile_round_trips_through_the_real_parser(profile):
    """Zero unparsed lines: the simulator writes real usbmon, not a dialect."""
    s = Simulator(seed=3).session("T0001", profile)
    src = SimulatedSource(s)
    evs = list(src.events())
    assert src.stats.lines_unparsed == 0, src.stats.unparsed_samples
    assert len(evs) == len(s.lines)
    assert src.stats.loss_rate == 0.0


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_every_event_satisfies_the_contract(profile):
    for e in SimulatedSource(Simulator(seed=4).session("T0001", profile)).events():
        e.validate()


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_submissions_and_completions_are_paired(profile):
    """CLAUDE.md §4 defect 3: the old simulator emitted completions only."""
    evs = list(SimulatedSource(Simulator(seed=5).session("T0001", profile)).events())
    kinds = collections.Counter(e.evt for e in evs)
    assert kinds["S"] > 0, "real usbmon emits a submission for every URB"
    assert kinds["S"] == kinds["C"]


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_control_traffic_exists(profile):
    """CLAUDE.md §4 defect 3: frac_ctrl was a constant-zero feature because the
    old corpus contained no control transfers at all."""
    evs = list(SimulatedSource(Simulator(seed=6).session("T0001", profile)).events())
    assert sum(1 for e in evs if e.ttype == "control") > 0


def test_control_submissions_carry_a_setup_packet():
    evs = list(SimulatedSource(Simulator(seed=7).session("T0001", "keyboard")).events())
    setups = [e for e in evs if e.ttype == "control" and e.evt == "S"]
    assert setups
    assert all(len(e.setup) == 8 for e in setups)


def test_frac_ctrl_is_nonzero_somewhere_in_a_session():
    s = Simulator(seed=8).session("T0001", "keyboard")
    evs = list(SimulatedSource(s).events())
    from iudbms.contract import feature_index
    vals = [EX.extract(b, t0, s.descriptor)[feature_index("frac_ctrl")]
            for t0, b in window_events(evs, duration=s.duration)]
    assert max(vals) > 0.0, "control traffic must reach the feature vector"


def test_timestamps_are_sorted():
    s = Simulator(seed=9).session("T0001", "mass_storage")
    ts = [e.ts for e in SimulatedSource(s).events()]
    assert ts == sorted(ts)


def test_source_satisfies_the_protocol():
    src = SimulatedSource(Simulator(seed=1).session("T0001", "mouse"))
    assert isinstance(src, EventSource)
    assert src.descriptor(src._session.descriptor.device_key) is not None
    with pytest.raises(KeyError):
        src.descriptor("99-99")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        Simulator(seed=1).session("T0001", "not_a_profile")


def test_determinism_under_seed():
    a = Simulator(seed=42).session("T0001", "staged_delay")
    b = Simulator(seed=42).session("T0001", "staged_delay")
    assert a.lines == b.lines
    assert (a.duration, a.onset) == (b.duration, b.onset)


# --------------------------------------------------------------------------
# attack semantics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", BENIGN_PROFILES)
def test_benign_profiles_have_no_onset(profile):
    s = Simulator(seed=2).session("T0001", profile)
    assert s.label == 0 and s.onset is None


@pytest.mark.parametrize("profile", ATTACK_PROFILES)
def test_attack_profiles_have_an_onset_inside_the_session(profile):
    s = Simulator(seed=2).session("T0001", profile)
    assert s.label == 1
    assert s.onset is not None
    assert 0 < s.onset < s.duration, "an onset past the end means no attack occurs"


def test_identity_switch_exposes_its_hid_interface_at_onset():
    s = Simulator(seed=11).session("T0001", "identity_switch")
    assert s.descriptor.classes_at(s.onset - 0.01) == frozenset({0x08})
    assert s.descriptor.classes_at(s.onset + 0.01) == frozenset({0x08, 0x03})


def test_attack_descriptors_are_indistinguishable_from_benign_ones():
    """The whole premise: identity checks cannot see these attacks."""
    sim = Simulator(seed=12)
    benign_ids = {(sim.session(f"S{i}", p).descriptor.vid,
                   sim.session(f"S{i}", p).descriptor.pid)
                  for i, p in enumerate(BENIGN_PROFILES * 3)}
    for p in ("hid_injection", "staged_delay", "mimicry", "class_exfil"):
        d = sim.session("A0001", p).descriptor
        assert (d.vid, d.pid) in benign_ids


def test_class_exfil_declares_hid_but_sends_bulk_out():
    s = Simulator(seed=13).session("T0001", "class_exfil")
    evs = list(SimulatedSource(s).events())
    assert s.descriptor.classes_at(1e9) == frozenset({0x03})
    post = [e for e in evs if e.ts > s.onset and e.evt == "C"]
    assert sum(1 for e in post if e.ttype == "bulk" and e.direction == "out") > 100


# --------------------------------------------------------------------------
# the leak, and the audit that catches it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_every_profile_draws_from_the_shared_duration_range(profile):
    """CLAUDE.md §4 defect 4. staged_delay used to redraw uniform(65, 85)."""
    sim = Simulator(seed=21)
    durs = [sim.session(f"T{i:04d}", profile).duration for i in range(40)]
    lo, hi = DURATION_RANGE
    assert min(durs) >= lo and max(durs) <= hi
    # and it must actually span the range, not cluster in the top half
    assert min(durs) < lo + 0.25 * (hi - lo), (
        f"{profile} never produces a short session; it is drawing from a "
        f"narrower range than the shared one"
    )


def test_staged_delay_keeps_a_long_dwell_without_a_longer_session():
    """The attack needs to dwell before firing, which is why it used to draw a
    longer duration. A proportional onset keeps the semantics, drops the leak."""
    sim = Simulator(seed=22)
    for i in range(30):
        s = sim.session(f"T{i:04d}", "staged_delay")
        frac = s.onset / s.duration
        assert 0.40 <= frac <= 0.67, f"dwell fraction {frac:.2f} outside design range"
        assert s.onset > 15.0, "dwell must still be long in absolute terms"


@pytest.mark.slow
def test_leakage_audit_passes_on_the_current_simulator():
    assert audit(n_benign=120, n_attack=80)["findings"] == []


@pytest.mark.slow
def test_audit_catches_a_reintroduced_duration_leak(monkeypatch):
    """An audit that cannot fail manufactures confidence rather than providing
    it. Re-introduce the exact Sprint-0 defect and assert it is caught."""
    real_session = Simulator.session

    def leaky(self, sid, profile, duration=None):
        if profile == "staged_delay":
            duration = self.rng.uniform(65.0, 85.0)     # the original bug
        return real_session(self, sid, profile, duration)

    monkeypatch.setattr(Simulator, "session", leaky)
    findings = audit(n_benign=120, n_attack=80)["findings"]
    assert any("staged_delay" in f for f in findings), (
        f"audit failed to catch a re-introduced duration leak; got {findings}"
    )


def test_ks_statistic_separates_obviously_different_samples():
    rng = np.random.default_rng(0)
    same_a, same_b = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    d_same, crit = ks_statistic(same_a, same_b)
    d_diff, _ = ks_statistic(same_a, rng.normal(4, 1, 200))
    assert d_same < crit < d_diff


def test_mimicry_timing_matches_human_typing():
    """The mimicry profile exists to be timing-indistinguishable. An earlier
    version drew think-pauses from uniform(0.35, 1.1) against the human's
    uniform(0.4, 3.0), which left a blatant timing signature — a timing-only
    model detected it at 100%, defeating the point of the profile and
    inflating the apparent case for multi-modal fusion (pillar N1).

    Compares post-onset mimicry inter-arrivals against benign keyboard ones.
    """
    sim = Simulator(seed=31)
    human = []
    for i in range(6):
        s = sim.session(f"S{i:04d}", "keyboard")
        ts = [e.ts for e in SimulatedSource(s).events()
              if e.evt == "C" and e.ttype == "intr"]
        human.extend(np.diff(ts))

    mim = []
    for i in range(6):
        s = sim.session(f"A{i:04d}", "mimicry")
        ts = [e.ts for e in SimulatedSource(s).events()
              if e.evt == "C" and e.ttype == "intr" and e.ts > s.onset]
        mim.extend(np.diff(ts))

    d, crit = ks_statistic(np.array(human), np.array(mim))
    assert d < crit, (
        f"mimicry inter-arrivals are separable from human typing "
        f"(KS D={d:.4f} > critical {crit:.4f}); the profile is not doing its job"
    )


def test_mimicry_still_carries_a_structural_tell():
    """Timing is matched, so detection must rest on payload structure: mimicry
    replays a fixed script, benign typing does not."""
    sim = Simulator(seed=32)
    def repeat_ratio(profile, post_onset):
        s = sim.session("T0001", profile)
        evs = [e for e in SimulatedSource(s).events()
               if e.evt == "C" and e.data and (not post_onset or e.ts > s.onset)]
        payloads = [e.data for e in evs]
        counts = collections.Counter(payloads)
        return max(counts.values()) / len(payloads)

    assert repeat_ratio("mimicry", True) > repeat_ratio("keyboard", False)
