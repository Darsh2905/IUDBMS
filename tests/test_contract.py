"""Milestone 1 — the frozen data contract.

These tests exist to make the contract *enforced* rather than merely
documented. In particular ``test_feature_order_is_frozen`` will fail loudly if
anyone reorders or renames a feature: that order is baked into trained
checkpoints, the SQLite ``windows.features`` JSON and every published figure,
so a silent reorder would invalidate results without breaking anything visibly.
"""

import hashlib

import numpy as np
import pytest

from iudbms.contract import (
    CONTRACT_VERSION,
    FEATURE_DTYPE,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODALITIES,
    N_FEATURES,
    CaptureStats,
    ContractError,
    DeviceDescriptor,
    Endpoint,
    EventSource,
    FeatureExtractor,
    Interface,
    URBEvent,
    feature_index,
    modality_indices,
    modality_of,
)

# --------------------------------------------------------------------------
# feature schema
# --------------------------------------------------------------------------

def test_feature_count_matches_plan():
    """28 in contract 1.x; 29 from 2.0.0 (payload_seq_novelty); 30 from 3.0.0,
    which appended concurrent_device_rate — the first host-level feature."""
    assert N_FEATURES == 30
    assert len(FEATURE_NAMES) == 30


def test_feature_names_unique():
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_feature_order_is_frozen():
    """Guard against silent reordering. Update the digest ONLY together with a
    CONTRACT_VERSION bump and a retrain of every checkpoint."""
    digest = hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest()[:16]
    assert digest == "602d0ae16ce5132f", (
        f"feature order changed (digest {digest}). If this is deliberate, bump "
        f"CONTRACT_VERSION, retrain Stage 1 + Stage 2, and update this digest."
    )


def test_modalities_partition_the_vector():
    """Plan §4.2: timing 9, payload 7, protocol 6, identity 6 — plus the
    payload_seq_novelty appended in contract 2.0.0, making payload 8, and the
    host modality appended in 3.0.0."""
    expected = {"timing": 9, "payload": 8, "protocol": 6, "identity": 6, "host": 1}
    seen: list[int] = []
    for m in MODALITIES:
        idx = modality_indices(m)
        assert len(idx) == expected[m], f"{m}: expected {expected[m]}, got {len(idx)}"
        seen.extend(idx)
    assert sorted(seen) == list(range(N_FEATURES)), "modalities must partition exactly"


def test_modality_indices_is_a_union_not_a_concat():
    both = modality_indices("timing", "payload")
    assert both == tuple(sorted(set(modality_indices("timing")) | set(modality_indices("payload"))))
    assert modality_indices("timing", "timing") == modality_indices("timing")


def test_modality_lookup_helpers():
    assert modality_of("iat_entropy") == "timing"
    assert modality_of("frac_ctrl") == "protocol"
    assert modality_of("log_time_since_enum") == "identity"
    assert FEATURE_NAMES[feature_index("mean_iat")] == "mean_iat"
    assert FEATURE_SCHEMA[feature_index("has_hid")] == ("has_hid", "identity")


def test_unknown_modality_rejected():
    with pytest.raises(ContractError):
        modality_indices("timings")  # type: ignore[arg-type]


def test_contract_version_is_semver():
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


# --------------------------------------------------------------------------
# URBEvent
# --------------------------------------------------------------------------

def _ev(**kw) -> URBEvent:
    base = dict(ts=1.0, evt="C", ttype="intr", direction="in", bus=1, dev=3,
                ep=1, status=0, length=8, data=b"\x00" * 8, interval=8)
    base.update(kw)
    return URBEvent(**base)  # type: ignore[arg-type]


def test_urbevent_is_immutable():
    e = _ev()
    with pytest.raises(Exception):
        e.length = 99  # type: ignore[misc]


def test_urbevent_device_key():
    assert _ev(bus=2, dev=7).device_key == "2-7"


def test_valid_event_passes():
    _ev().validate()


def test_zero_length_transfer_is_valid():
    """A zero-length URB is a real observation, not an error."""
    _ev(length=0, data=b"").validate()


def test_length_is_data_length_not_interval():
    """Regression guard for the Sprint-0 defect: usbmon prints the polling
    interval adjacent to the status, and the old parser read it as the length.
    The two now live in separate fields and may legitimately differ."""
    e = _ev(length=8, data=b"\x00" * 8, interval=4)
    e.validate()
    assert e.length == 8
    assert e.interval == 4


def test_rejects_endpoint_address_instead_of_number():
    """0x81 is an endpoint *address*; the contract wants the number (1)."""
    with pytest.raises(ContractError, match="direction bit"):
        _ev(ep=0x81).validate()


def test_rejects_negative_length():
    with pytest.raises(ContractError, match="non-negative"):
        _ev(length=-1, data=b"").validate()


def test_rejects_payload_longer_than_length():
    with pytest.raises(ContractError, match="captured"):
        _ev(length=4, data=b"\x00" * 8).validate()


def test_short_payload_requires_truncation_flag():
    """Distinguishes 'zero-length transfer' from 'usbmon did not capture it'."""
    with pytest.raises(ContractError, match="data_truncated"):
        _ev(length=512, data=b"\x00" * 16, data_truncated=False).validate()
    _ev(length=512, data=b"\x00" * 16, data_truncated=True).validate()


def test_rejects_interval_on_bulk():
    with pytest.raises(ContractError, match="intr/iso"):
        _ev(ttype="bulk", interval=8).validate()


def test_rejects_non_finite_timestamp():
    with pytest.raises(ContractError, match="finite"):
        _ev(ts=float("nan")).validate()


@pytest.mark.parametrize("field,bad", [("evt", "X"), ("ttype", "isoc"), ("direction", "up")])
def test_rejects_bad_vocabulary(field, bad):
    with pytest.raises(ContractError):
        _ev(**{field: bad}).validate()


# --------------------------------------------------------------------------
# DeviceDescriptor
# --------------------------------------------------------------------------

def _desc(**kw) -> DeviceDescriptor:
    base = dict(
        bus=1, devnum=4, vid=0x046D, pid=0xC31C,
        interfaces=(
            Interface(0x08, 0x06, 0x50,
                      (Endpoint(0x81, "bulk"), Endpoint(0x02, "bulk")), first_seen_ts=0.0),
            Interface(0x03, 0x01, 0x01,
                      (Endpoint(0x83, "intr", interval=10),), first_seen_ts=22.5),
        ),
    )
    base.update(kw)
    return DeviceDescriptor(**base)  # type: ignore[arg-type]


def test_late_interface_is_invisible_before_it_arrives():
    """The identity_switch attack: storage device grows an HID interface."""
    d = _desc()
    assert len(d.visible_at(10.0)) == 1
    assert d.classes_at(10.0) == frozenset({0x08})
    assert len(d.visible_at(30.0)) == 2
    assert d.classes_at(30.0) == frozenset({0x08, 0x03})
    assert d.has_late_interface()


def test_late_interface_need_not_sort_last():
    """Sprint-0 reconstructed visibility as ``interfaces[:-1]``, so a late
    interface that is not last was mis-reported. Per-interface timestamps make
    the query total regardless of ordering."""
    d = _desc(interfaces=(
        Interface(0x03, 0x01, 0x01, (), first_seen_ts=30.0),   # late, but FIRST
        Interface(0x08, 0x06, 0x50, (), first_seen_ts=0.0),
    ))
    assert d.classes_at(10.0) == frozenset({0x08})
    assert d.classes_at(40.0) == frozenset({0x08, 0x03})


def test_endpoint_address_decoding():
    assert Endpoint(0x81, "intr").number == 1
    assert Endpoint(0x81, "intr").direction == "in"
    assert Endpoint(0x02, "bulk").number == 2
    assert Endpoint(0x02, "bulk").direction == "out"


# --------------------------------------------------------------------------
# CaptureStats
# --------------------------------------------------------------------------

def test_loss_rate_zero_when_nothing_seen():
    assert CaptureStats().loss_rate == 0.0


def test_loss_rate_counts_unparsed_and_overflow():
    s = CaptureStats(lines_seen=1000, events_dropped_overflow=10)
    for _ in range(5):
        s.note_unparsed("garbage")
    assert s.loss_rate == pytest.approx(15 / 1010)


def test_unparsed_samples_are_capped():
    s = CaptureStats()
    for i in range(100):
        s.note_unparsed(f"line {i}")
    assert s.lines_unparsed == 100
    assert len(s.unparsed_samples) == CaptureStats.MAX_SAMPLES


def test_duration_and_rate_track_events():
    s = CaptureStats()
    for i in range(5):
        s.note_event(_ev(ts=float(i)))
    assert s.events_emitted == 5
    assert s.duration == pytest.approx(4.0)
    assert s.event_rate == pytest.approx(1.25)


# --------------------------------------------------------------------------
# protocols / ABCs
# --------------------------------------------------------------------------

def test_event_source_protocol_is_structural():
    class Fake:
        def __init__(self):
            self._stats = CaptureStats()

        def events(self):
            yield _ev()

        def descriptor(self, device_key):
            return _desc()

        @property
        def stats(self):
            return self._stats

    assert isinstance(Fake(), EventSource)
    assert not isinstance(object(), EventSource)


class _Extractor(FeatureExtractor):
    def __init__(self, vec=None):
        self._vec = vec

    def extract(self, events, t0, descriptor):
        if self._vec is not None:
            return self._vec
        return np.zeros(N_FEATURES, dtype=FEATURE_DTYPE)


def test_extract_checked_accepts_valid_vector():
    v = _Extractor().extract_checked([], 0.0, _desc())
    assert v.shape == (N_FEATURES,) and v.dtype == FEATURE_DTYPE


def test_extract_checked_rejects_wrong_shape():
    with pytest.raises(ContractError, match="shape"):
        _Extractor(np.zeros(27, dtype=FEATURE_DTYPE)).extract_checked([], 0.0, _desc())


def test_extract_checked_rejects_wrong_dtype():
    with pytest.raises(ContractError, match="dtype"):
        _Extractor(np.zeros(N_FEATURES, dtype=np.float64)).extract_checked([], 0.0, _desc())


def test_extract_checked_names_the_non_finite_features():
    bad = np.zeros(N_FEATURES, dtype=FEATURE_DTYPE)
    bad[feature_index("cv_iat")] = np.nan
    bad[feature_index("mean_len")] = np.inf
    with pytest.raises(ContractError, match="cv_iat"):
        _Extractor(bad).extract_checked([], 0.0, _desc())


def test_window_defaults_match_plan():
    assert FeatureExtractor.WINDOW == 1.0
    assert FeatureExtractor.STRIDE == 1.0
