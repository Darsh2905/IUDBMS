"""Milestone 3 — the 28-feature extractor.

Includes the plan §4.2 acceptance criteria (empty-window safety, throughput)
and the invariant that makes the Sprint-0 corpus's missing submission events
harmless: features are computed over completions only, so adding submissions to
a stream must not move the vector.
"""

import math
import time

import numpy as np
import pytest

from iudbms.contract import (
    FEATURE_DTYPE,
    N_FEATURES,
    ContractError,
    DeviceDescriptor,
    Endpoint,
    Interface,
    URBEvent,
    feature_index,
)
from iudbms.features.extractor import WindowFeatureExtractor, window_events

EX = WindowFeatureExtractor()


def feat(vec, name):
    return float(vec[feature_index(name)])


def ev(ts, *, ttype="intr", direction="in", ep=1, length=8, data=b"", evt="C",
       status=0, interval=None, truncated=False):
    return URBEvent(ts=ts, evt=evt, ttype=ttype, direction=direction, bus=1, dev=3,
                    ep=ep, status=status, length=length, data=data,
                    interval=interval, data_truncated=truncated)


def kbd(late_hid_at=None):
    ifaces = [Interface(0x03, 0x01, 0x01, (Endpoint(0x81, "intr", 10),), first_seen_ts=0.0)]
    if late_hid_at is not None:
        ifaces = [Interface(0x08, 0x06, 0x50, (Endpoint(0x81, "bulk"),), first_seen_ts=0.0),
                  Interface(0x03, 0x01, 0x01, (Endpoint(0x83, "intr", 10),),
                            first_seen_ts=late_hid_at)]
    return DeviceDescriptor(bus=1, devnum=3, vid=0x046D, pid=0xC31C,
                            interfaces=tuple(ifaces))


# --------------------------------------------------------------------------
# plan §4.2 acceptance: empty windows
# --------------------------------------------------------------------------

def test_empty_window_is_valid_and_finite():
    v = EX.extract_checked([], 0.0, kbd())
    assert np.all(np.isfinite(v))
    assert feat(v, "n_events") == 0.0
    assert feat(v, "ev_rate") == 0.0


def test_empty_window_reports_full_window_iats_not_zero():
    """Silent must not masquerade as infinitely fast to a downstream scaler."""
    v = EX.extract_checked([], 0.0, kbd())
    for name in ("mean_iat", "min_iat", "p10_iat"):
        assert feat(v, name) == pytest.approx(EX.WINDOW)


def test_empty_window_still_carries_identity():
    v = EX.extract_checked([], 5.0, kbd())
    assert feat(v, "n_interfaces") == 1.0
    assert feat(v, "has_hid") == 1.0
    assert feat(v, "log_time_since_enum") == pytest.approx(math.log1p(5.0))


def test_single_event_window_is_finite():
    """n == 1 means no inter-arrival exists at all."""
    v = EX.extract_checked([ev(0.5)], 0.0, kbd())
    assert np.all(np.isfinite(v))
    assert feat(v, "n_events") == 1.0
    assert feat(v, "std_iat") == 0.0
    assert feat(v, "mean_iat") == pytest.approx(EX.WINDOW)


def test_output_is_float32_of_the_contracted_width():
    v = EX.extract([], 0.0, kbd())
    assert v.shape == (N_FEATURES,) and v.dtype == FEATURE_DTYPE


# --------------------------------------------------------------------------
# submissions must not change anything
# --------------------------------------------------------------------------

def test_submissions_do_not_change_the_vector():
    """Every S has a matching C, so counting both would double every rate while
    adding no information. This is why the Sprint-0 corpus's missing S events
    (CLAUDE.md §4 defect 3) were never load-bearing: adding them to the
    simulator makes the trace realistic without shifting any baseline."""
    completions = [ev(0.1 * i) for i in range(1, 10)]
    with_subs = []
    for c in completions:
        with_subs.append(ev(c.ts - 0.001, evt="S", status=-115))
        with_subs.append(c)
    a = EX.extract_checked(completions, 0.0, kbd())
    b = EX.extract_checked(with_subs, 0.0, kbd())
    np.testing.assert_array_equal(a, b)


def test_error_events_are_counted():
    """An E event is a real observation about a misbehaving device."""
    v = EX.extract_checked([ev(0.1), ev(0.2, evt="E", status=-71)], 0.0, kbd())
    assert feat(v, "n_events") == 2.0


# --------------------------------------------------------------------------
# timing modality
# --------------------------------------------------------------------------

def test_regular_injection_has_low_cv_and_low_entropy():
    """Programmatic HID injection: near-constant 4 ms inter-arrivals."""
    inj = [ev(0.004 * i) for i in range(1, 200)]
    v = EX.extract_checked(inj, 0.0, kbd())
    assert feat(v, "cv_iat") < 0.05
    assert feat(v, "iat_entropy") < 0.2
    assert feat(v, "ev_rate") > 150


def test_human_typing_has_higher_cv_than_injection():
    rng = np.random.default_rng(0)
    human = [ev(t) for t in np.cumsum(rng.lognormal(-1.9, 0.55, 12)) if t < 1.0]
    inj = [ev(0.004 * i) for i in range(1, 200)]
    hv = EX.extract_checked(human, 0.0, kbd())
    iv = EX.extract_checked(inj, 0.0, kbd())
    assert feat(hv, "cv_iat") > feat(iv, "cv_iat")
    assert feat(hv, "mean_iat") > feat(iv, "mean_iat")


def test_fano_separates_bursty_from_regular():
    regular = [ev(0.01 * i) for i in range(1, 100)]
    bursty = [ev(0.001 * i) for i in range(1, 50)] + [ev(0.9 + 0.001 * i) for i in range(50)]
    assert feat(EX.extract_checked(bursty, 0.0, kbd()), "burstiness_fano") > \
           feat(EX.extract_checked(regular, 0.0, kbd()), "burstiness_fano")


def test_iat_entropy_is_normalised_to_unit_range():
    rng = np.random.default_rng(1)
    for scale in (0.001, 0.01, 0.1):
        evs = [ev(t) for t in np.cumsum(rng.exponential(scale, 60)) if t < 1.0]
        if len(evs) >= 2:
            assert 0.0 <= feat(EX.extract_checked(evs, 0.0, kbd()), "iat_entropy") <= 1.0


# --------------------------------------------------------------------------
# payload modality
# --------------------------------------------------------------------------

def test_uncaptured_payloads_do_not_look_like_perfect_repetition():
    """CLAUDE.md §3 rationale for data_truncated: a bulk stream the kernel
    declined to capture must not read as a perfectly repeating one."""
    evs = [ev(0.01 * i, ttype="bulk", length=512, data=b"", truncated=True)
           for i in range(1, 50)]
    v = EX.extract_checked(evs, 0.0, kbd())
    assert feat(v, "repeat_ratio") == 0.0
    assert feat(v, "payload_novelty") == 0.0


def test_scripted_payloads_repeat_more_than_random_ones():
    script = [bytes([0, 0, k, 0, 0, 0, 0, 0]) for k in (4, 5, 6)]
    scripted = [ev(0.01 * i, data=script[i % 3]) for i in range(30)]
    rng = np.random.default_rng(2)
    varied = [ev(0.01 * i, data=bytes(rng.integers(0, 255, 8))) for i in range(30)]
    assert feat(EX.extract_checked(scripted, 0.0, kbd()), "payload_novelty") < \
           feat(EX.extract_checked(varied, 0.0, kbd()), "payload_novelty")
    assert feat(EX.extract_checked(scripted, 0.0, kbd()), "repeat_ratio") > \
           feat(EX.extract_checked(varied, 0.0, kbd()), "repeat_ratio")


def test_byte_rates_use_transfer_length_not_captured_prefix():
    """usbmon captures a prefix; the transfer really moved `length` bytes."""
    evs = [ev(0.1 * i, ttype="bulk", direction="out", length=4096,
              data=b"\x00" * 16, truncated=True) for i in range(1, 6)]
    v = EX.extract_checked(evs, 0.0, kbd())
    assert feat(v, "bytes_out_rate") == pytest.approx(5 * 4096)
    assert feat(v, "bytes_in_rate") == 0.0


def test_direction_split_is_exhaustive():
    evs = [ev(0.1, direction="in", length=100), ev(0.2, direction="out", length=300)]
    v = EX.extract_checked(evs, 0.0, kbd())
    assert feat(v, "bytes_in_rate") == pytest.approx(100)
    assert feat(v, "bytes_out_rate") == pytest.approx(300)
    assert feat(v, "frac_in") == pytest.approx(0.5)


# --------------------------------------------------------------------------
# protocol modality
# --------------------------------------------------------------------------

def test_transfer_type_fractions_sum_to_one():
    evs = [ev(0.1, ttype="intr"), ev(0.2, ttype="bulk"),
           ev(0.3, ttype="iso"), ev(0.4, ttype="control")]
    v = EX.extract_checked(evs, 0.0, kbd())
    total = sum(feat(v, k) for k in ("frac_intr", "frac_bulk", "frac_iso", "frac_ctrl"))
    assert total == pytest.approx(1.0)
    assert feat(v, "frac_ctrl") == pytest.approx(0.25)


def test_frac_ctrl_is_live_not_constant_zero():
    """CLAUDE.md §4 defect 3: the Sprint-0 corpus contained no control traffic
    at all, so this feature was constant zero across every window."""
    v = EX.extract_checked([ev(0.1, ttype="control", ep=0)], 0.0, kbd())
    assert feat(v, "frac_ctrl") == 1.0


def test_endpoint_fanout():
    evs = [ev(0.1, ep=1), ev(0.2, ep=2), ev(0.3, ep=1), ev(0.4, ep=3)]
    assert feat(EX.extract_checked(evs, 0.0, kbd()), "n_endpoints") == 3.0


# --------------------------------------------------------------------------
# identity modality — the core of the thesis
# --------------------------------------------------------------------------

def test_hid_declared_device_doing_bulk_out_trips_the_mismatch():
    """class_exfil: the descriptor says keyboard, the behaviour says storage."""
    evs = [ev(0.01 * i, ttype="bulk", direction="out", ep=2, length=2048)
           for i in range(1, 50)]
    v = EX.extract_checked(evs, 0.0, kbd())
    assert feat(v, "class_behaviour_mismatch") == 1.0
    assert feat(v, "has_hid") == 1.0, "descriptor still claims HID"


def test_hid_doing_interrupt_traffic_has_no_mismatch():
    evs = [ev(0.01 * i, ttype="intr") for i in range(1, 50)]
    assert feat(EX.extract_checked(evs, 0.0, kbd()), "class_behaviour_mismatch") == 0.0


def test_control_traffic_never_counts_as_mismatch():
    """Every device does control transfers; enumeration is not an attack."""
    evs = [ev(0.01 * i, ttype="control", ep=0) for i in range(1, 10)]
    assert feat(EX.extract_checked(evs, 0.0, kbd()), "class_behaviour_mismatch") == 0.0


def test_late_interface_is_invisible_before_it_arrives():
    """identity_switch: storage that grows an HID interface at t=20s."""
    d = kbd(late_hid_at=20.0)
    before = EX.extract_checked([], 10.0, d)
    after = EX.extract_checked([], 25.0, d)
    assert feat(before, "has_hid") == 0.0
    assert feat(before, "n_interfaces") == 1.0
    assert feat(before, "late_iface_flag") == 0.0
    assert feat(after, "has_hid") == 1.0
    assert feat(after, "n_interfaces") == 2.0
    assert feat(after, "late_iface_flag") == 1.0


def test_mismatch_uses_only_interfaces_visible_in_this_window():
    """Interrupt traffic from a storage-only device is a mismatch until the
    HID interface actually appears."""
    d = kbd(late_hid_at=20.0)
    evs = [ev(0.01 * i, ttype="intr") for i in range(1, 20)]
    assert feat(EX.extract_checked(evs, 10.0, d), "class_behaviour_mismatch") == 1.0
    assert feat(EX.extract_checked(evs, 25.0, d), "class_behaviour_mismatch") == 0.0


def test_log_time_since_enum_is_monotonic():
    vals = [feat(EX.extract_checked([], t, kbd()), "log_time_since_enum")
            for t in (0.0, 1.0, 10.0, 60.0)]
    assert vals == sorted(vals)
    assert vals[0] == 0.0


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------

def test_windows_are_tumbling_and_cover_every_event():
    evs = [ev(t) for t in (0.1, 0.5, 1.2, 3.7)]
    got = list(window_events(evs, duration=4.0))
    assert [t0 for t0, _ in got] == [0.0, 1.0, 2.0, 3.0]
    assert sum(len(b) for _, b in got) == len(evs)
    assert len(got[0][1]) == 2 and len(got[1][1]) == 1
    assert got[2][1] == [], "idle windows are yielded, not skipped"


def test_windows_extend_to_session_duration_when_device_goes_silent():
    got = list(window_events([ev(0.5)], duration=10.0))
    assert len(got) == 10
    assert all(b == [] for _, b in got[1:])


def test_at_least_one_window_even_with_no_events():
    assert len(list(window_events([], duration=0.0))) == 1


def test_every_window_of_a_session_passes_the_contract():
    rng = np.random.default_rng(3)
    evs = [ev(float(t)) for t in np.cumsum(rng.exponential(0.05, 400))]
    d = kbd()
    for t0, bucket in window_events(evs, duration=25.0):
        EX.extract_checked(bucket, t0, d)


# --------------------------------------------------------------------------
# plan §4.2 acceptance: throughput
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_throughput_meets_plan_requirement():
    """Plan §4.2: >= 500 windows/s."""
    d = kbd()
    bucket = [ev(0.005 * i) for i in range(1, 200)]
    n = 2000
    t = time.perf_counter()
    for i in range(n):
        EX.extract(bucket, float(i), d)
    rate = n / (time.perf_counter() - t)
    assert rate >= 500, f"only {rate:,.0f} windows/s"


def test_extractor_rejects_a_broken_subclass():
    class Broken(WindowFeatureExtractor):
        def extract(self, events, t0, descriptor):
            v = super().extract(events, t0, descriptor)
            out = v.copy()
            out[feature_index("cv_iat")] = np.nan
            return out

    with pytest.raises(ContractError, match="cv_iat"):
        Broken().extract_checked([], 0.0, kbd())


def test_event_exactly_on_a_window_boundary_is_not_dropped():
    """An event at t=4.0 belongs to window 4, so a 4.0 s session needs five
    windows to hold it even though [0, 4) needs only four."""
    got = list(window_events([ev(4.0)], duration=4.0))
    assert len(got) == 5
    assert got[4][1] and got[4][0] == 4.0
    assert sum(len(b) for _, b in got) == 1
