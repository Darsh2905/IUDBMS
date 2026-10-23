"""The live lab's plumbing (no VM): hub fan-out, refusal before ready, page served."""

import queue

import pytest
from fastapi.testclient import TestClient

from iudbms.api import lab


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(lab, "LAB", lab.Lab("debian-13"))
    return TestClient(lab.app)            # no `with`: the lifespan (which boots a VM) is not run


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "live lab" in r.text


def test_actions_are_refused_until_the_guest_is_live(client):
    assert client.post("/lab/action", json={"name": "inject"}).status_code == 409
    assert client.post("/lab/action", json={"name": "rm-rf"}).status_code == 404


def test_hub_keeps_history_and_fans_out():
    h = lab.Hub()
    q = h.subscribe()
    w = {"device": "1-2", "t": 0.0}
    h.emit("window", w)
    h.emit("log", {"msg": "x"})
    assert q.get_nowait() == ("window", w) and q.get_nowait()[0] == "log"
    snap = h.snapshot()
    assert snap["history"]["1-2"] == [w] and snap["log"] == [{"msg": "x"}]
    h.unsubscribe(q)
    h.emit("log", {"msg": "y"})
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_hub_history_is_bounded():
    h = lab.Hub()
    for i in range(lab.HISTORY + 50):
        h.emit("window", {"device": "d", "t": float(i)})
    assert len(h.snapshot()["history"]["d"]) == lab.HISTORY
