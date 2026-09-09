"""Milestone 7 — the control API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iudbms.api.app import app
from iudbms.capture.simulator import Simulator
from iudbms.contract import N_FEATURES
from iudbms.storage import Store

DESC = Simulator(seed=1).session("A0001", "hid_injection").descriptor


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    with Store(db) as s:
        s.add_device("A0001", DESC, "hid")
        s.add_windows("A0001", [0.0, 1.0, 2.0], [0.1, 0.2, 0.9], [0.1, 0.3, 0.99],
                      [0, 0, 1], np.zeros((3, N_FEATURES), dtype=np.float32))
        aid = s.add_alert("A0001", 2.0, 0.9, 0.99, "critical", [("cv_iat", 1.8)])
        s.add_action(aid, "A0001", 2.0, "DEAUTHORIZE", {"kernel_cmd": "echo 0 > x"})
    monkeypatch.setenv("IUDBMS_DB", str(db))
    return TestClient(app)


def test_health_reports_enforcement_is_gated(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["enforcement_reviewed"] is False


def test_devices(client):
    d = client.get("/devices").json()
    assert len(d) == 1 and d[0]["device_id"] == "A0001"


def test_timeline_is_ordered(client):
    body = client.get("/devices/A0001/timeline").json()
    assert [w["t_start"] for w in body["windows"]] == [0.0, 1.0, 2.0]


def test_timeline_404_for_unknown_device(client):
    assert client.get("/devices/nope/timeline").status_code == 404


def test_alerts_carry_evidence(client):
    a = client.get("/alerts").json()
    assert a[0]["severity"] == "critical"
    assert a[0]["evidence"] == [["cv_iat", 1.8]]


def test_actions_carry_the_generated_rule(client):
    assert client.get("/actions").json()[0]["rule"]["kernel_cmd"] == "echo 0 > x"


def test_authorize_records_the_decision_without_touching_the_kernel(client):
    body = client.post("/devices/A0001/authorize",
                       json={"authorized": False, "reason": "test"}).json()
    assert body["trust_state"] == "revoked"
    assert body["applied_to_kernel"] is False
    assert client.get("/devices").json()[0]["trust_state"] == "revoked"


def test_policy_rejects_k_greater_than_n(client):
    r = client.post("/policy", json={"threshold": 0.9, "k": 7, "n": 6})
    assert r.status_code == 422


def test_policy_rejects_out_of_range_threshold(client):
    assert client.post("/policy", json={"threshold": 1.5}).status_code == 422


def test_policy_echoes_effective_settings(client):
    body = client.post("/policy", json={"threshold": 0.94, "k": 4, "n": 6}).json()
    assert body["threshold"] == 0.94 and body["enforcement_reviewed"] is False


def test_stats(client):
    assert client.get("/stats").json()["devices"] == 1


def test_missing_database_is_a_clear_503(tmp_path, monkeypatch):
    monkeypatch.setenv("IUDBMS_DB", str(tmp_path / "absent.db"))
    r = TestClient(app).get("/devices")
    assert r.status_code == 503
    assert "run_eval" in r.json()["detail"]
