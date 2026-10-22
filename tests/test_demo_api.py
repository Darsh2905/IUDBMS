"""The live demo streams the deployed pipeline's own verdicts, and says so honestly."""

import json

import pytest
from fastapi.testclient import TestClient

from iudbms.api import demo
from iudbms.api.app import app

pytestmark = pytest.mark.skipif(not (demo.models_dir() / "deploy.json").exists(),
                                reason="needs the deployed models in out/models")


def _stream(client, scenario, seed=11):
    text = client.get(f"/demo/stream?scenario={scenario}&speed=60&seed={seed}").text
    ev = {}
    for block in filter(None, text.split("\n\n")):
        name, data = block.split("\n", 1)
        ev.setdefault(name.removeprefix("event: "), []).append(json.loads(data.removeprefix("data: ")))
    return ev


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_page_and_scenarios_are_served(client):
    assert client.get("/demo").status_code == 200
    ids = {s["id"] for s in client.get("/demo/scenarios").json()}
    assert {"kbd", "inject", "slow", "replay"} <= ids
    assert client.get("/demo/stream?scenario=nope").status_code == 404


def test_every_stream_is_dry_run_and_matches_the_deployed_operating_point(client):
    cfg = json.loads((demo.models_dir() / "deploy.json").read_text())
    ev = _stream(client, "inject")
    init = ev["init"][0]
    assert init["dry_run"] is True
    assert (init["threshold"], init["k"], init["n"]) == (cfg["threshold"], cfg["k"], cfg["n"])
    assert len(ev["window"]) == init["windows"] and len(ev["end"]) == 1


def test_streamed_scores_equal_the_detectors_own(client):
    """Not a mock: replaying the same session through a fresh StreamingDetector gives the
    same Stage 2 scores the page received."""
    from iudbms.features.extractor import window_events
    ev = _stream(client, "inject", seed=5)
    det, _ = demo._detector()
    desc, events, duration, _ = demo._sim_session(demo.scenarios()["inject"], 5)
    det.attach(desc, desc.device_key)
    ours = [det.observe_window(desc.device_key, t, b).stage2
            for t, b in window_events(events, duration=duration)]
    assert [w["s2"] for w in ev["window"]] == pytest.approx(ours, abs=1e-9)


@pytest.mark.parametrize("scenario", ["kbd", "mouse", "drive"])
def test_benign_devices_stay_quiet(client, scenario):
    end = _stream(client, scenario)["end"][0]
    assert end["alarmed"] is False


@pytest.mark.parametrize("scenario", ["inject", "staged", "switch", "exfil"])
def test_injection_speed_attacks_are_caught_within_ten_seconds(client, scenario):
    end = _stream(client, scenario)["end"][0]
    assert end["alarmed"] and end["latency"] <= 10.0


def test_the_known_miss_is_shown_as_a_miss(client):
    """Human-speed single-shot payload: the demo must not quietly stop including it."""
    sc = demo.scenarios()["slow"]
    assert sc.expect == "miss"
    assert _stream(client, "slow")["end"][0]["alarmed"] is False
