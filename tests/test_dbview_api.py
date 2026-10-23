"""The database explorer reads a store read-only and reports what is in it."""

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iudbms.api import dbview
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
                      [0, 1, 1], np.zeros((3, N_FEATURES), dtype=np.float32))
        aid = s.add_alert("A0001", 2.0, 0.9, 0.99, "critical", [("cv_iat", 1.8)])
        s.add_action(aid, "A0001", 2.0, "DEAUTHORIZE", {"kernel_cmd": "echo 0 > x"})
    monkeypatch.setenv("IUDBMS_DB", str(db))
    dbview._devices.cache_clear()
    return TestClient(dbview.app)


def test_overview_counts_and_labels(client):
    o = client.get("/db/overview").json()
    assert (o["sessions"], o["attack"], o["attack_alerted"]) == (1, 1, 1)
    assert o["tables"]["windows"] == 3 and o["windows"]["attack_windows"] == 2


def test_device_detail_and_window_features(client):
    d = client.get("/db/device/A0001").json()
    assert len(d["windows"]) == 3 and d["alerts"][0]["evidence"] == [["cv_iat", 1.8]]
    assert d["actions"][0]["rule"]["kernel_cmd"] == "echo 0 > x"
    w = client.get("/db/device/A0001/window", params={"t": 1.0}).json()
    assert w["label"] == 1 and w["features"]
    assert client.get("/db/device/nope").status_code == 404


def test_table_browser_whitelists_names(client):
    assert client.get("/db/table/devices").json()["total"] == 1
    assert client.get("/db/table/sqlite_master").status_code == 404
    assert client.get("/db/table/devices; DROP TABLE devices").status_code == 404


def test_database_is_read_only(client):
    with pytest.raises(sqlite3.OperationalError):
        dbview.cx().execute("DELETE FROM devices")
