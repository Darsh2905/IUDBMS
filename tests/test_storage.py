"""Milestone 7 — SQLite persistence (plan §4.6 schema)."""

import numpy as np
import pytest

from iudbms.capture.simulator import Simulator
from iudbms.contract import FEATURE_NAMES, N_FEATURES
from iudbms.stage1 import DeviceBaseline, class_group
from iudbms.storage import Store

DESC = Simulator(seed=1).session("A0001", "hid_injection").descriptor


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        yield s


def test_schema_creates_every_table(store):
    assert set(store.counts()) == {"devices", "windows", "baselines", "alerts", "actions",
                                   "trace_runs", "trace_sessions", "urb_events"}
    assert all(v == 0 for v in store.counts().values())


def test_old_database_gains_trace_tables(tmp_path):
    """Databases written before the trace tables existed must gain them on the
    next open rather than fail -- the dashboard opens whatever DB it is given."""
    import sqlite3
    db = tmp_path / "old.db"
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE devices (device_id TEXT PRIMARY KEY, bus INTEGER)")
    cx.commit(); cx.close()
    with Store(db) as s:
        assert s.counts()["urb_events"] == 0


def test_device_round_trip(store):
    store.add_device("A0001", DESC, class_group(DESC), first_seen=1.5)
    d = store.devices()[0]
    assert d["device_id"] == "A0001"
    assert (d["bus"], d["devnum"]) == (DESC.bus, DESC.devnum)
    assert d["class_group"] == "hid"
    assert d["trust_state"] == "observe"


def test_add_device_is_idempotent(store):
    for _ in range(3):
        store.add_device("A0001", DESC, "hid")
    assert store.counts()["devices"] == 1


def test_windows_store_named_features(store):
    store.add_device("A0001", DESC, "hid")
    X = np.arange(2 * N_FEATURES, dtype=np.float32).reshape(2, N_FEATURES)
    store.add_windows("A0001", [0.0, 1.0], [0.1, 0.9], [0.2, 0.99], [0, 1], X)
    rows = store.timeline("A0001")
    assert [r["t_start"] for r in rows] == [0.0, 1.0]
    assert rows[1]["label"] == 1
    import json
    feats = json.loads(store.cx.execute(
        "SELECT features FROM windows ORDER BY t_start").fetchone()[0])
    assert set(feats) == set(FEATURE_NAMES)


def test_baseline_persists_one_row_per_feature(store):
    store.add_device("A0001", DESC, "hid")
    bl = DeviceBaseline()
    for _ in range(5):
        bl.update(np.zeros(N_FEATURES))
    store.add_baseline("A0001", bl)
    assert store.counts()["baselines"] == N_FEATURES


def test_baseline_respects_a_narrower_feature_set(store):
    """Modality ablations produce baselines over column subsets."""
    store.add_device("A0001", DESC, "hid")
    bl = DeviceBaseline(feature_names=("mean_iat", "cv_iat"))
    bl.update(np.zeros(2))
    store.add_baseline("A0001", bl)
    assert store.counts()["baselines"] == 2


def test_alert_and_action_are_linked(store):
    store.add_device("A0001", DESC, "hid")
    aid = store.add_alert("A0001", 12.0, 0.4, 0.98, "critical", [("cv_iat", 1.8)])
    store.add_action(aid, "A0001", 12.0, "DEAUTHORIZE", {"kernel_cmd": "echo 0 > ..."})
    a = store.alerts()[0]
    assert a["severity"] == "critical"
    assert a["evidence"] == [["cv_iat", 1.8]]
    act = store.actions()[0]
    assert act["alert_id"] == aid
    assert act["rule"]["kernel_cmd"].startswith("echo 0")


def test_alerts_filter_by_device(store):
    for did in ("A0001", "A0002"):
        store.add_device(did, DESC, "hid")
        store.add_alert(did, 1.0, 0.1, 0.9, "high", [])
    assert len(store.alerts()) == 2
    assert len(store.alerts("A0001")) == 1


def test_trust_state_transitions(store):
    store.add_device("A0001", DESC, "hid")
    store.set_trust("A0001", "revoked")
    assert store.devices()[0]["trust_state"] == "revoked"


def test_timeline_is_ordered_by_time(store):
    store.add_device("A0001", DESC, "hid")
    X = np.zeros((3, N_FEATURES), dtype=np.float32)
    store.add_windows("A0001", [2.0, 0.0, 1.0], [0] * 3, [0] * 3, [0] * 3, X)
    assert [r["t_start"] for r in store.timeline("A0001")] == [0.0, 1.0, 2.0]
