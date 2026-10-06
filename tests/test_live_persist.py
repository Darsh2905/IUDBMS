"""Live persistence (live.persist) and the monitor CLI (live.cli).

The live monitor used to write nothing: it printed to a terminal, and the store
the dashboard reads was only ever filled from a batch evaluation. These tests
pin down what a deployed monitor now writes, and that the CLI refuses to run in
a configuration it would otherwise silently degrade to.
"""

import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pytest

from iudbms.capture.simulator import SimulatedSource, Simulator
from iudbms.contract import N_FEATURES, CaptureStats
from iudbms.live import LivePersister, StreamingDetector
from iudbms.live.monitor import LiveMonitor
from iudbms.policy import PolicyEngine
from iudbms.stage1 import Stage1
from iudbms.storage import Store, TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "out" / "models"
FIXTURE = ROOT / "tests" / "fixtures" / "usbmon"


class ListSource:
    def __init__(self, events):
        self._events = events
        self.stats = CaptureStats()

    def events(self):
        yield from self._events

    def descriptor(self, key):
        raise KeyError(key)


def _clock():
    t = {"v": 0.0}

    def tick():
        t["v"] += 1.0
        return t["v"]
    return tick


@pytest.fixture(scope="module")
def stage1():
    X = np.random.default_rng(0).normal(0, 1, (300, N_FEATURES))
    return Stage1(seed=0).fit(X, ["hid"] * 300)


def _run(store, stage1, threshold, *, recorder=None):
    """A keyboard session through the real monitor, persisted live."""
    s = Simulator(seed=4).session("K", "hid_injection", duration=15.0)
    desc = s.descriptor
    events = list(SimulatedSource(s).events())
    det = StreamingDetector(stage1, None, PolicyEngine(threshold=threshold, k=2, n=3))
    mon = LiveMonitor(det, ListSource(events), lambda k: desc, clock=_clock(),
                      recorder=recorder)
    per = LivePersister(store, recorder=recorder, seq_len=10)
    for batch in mon.run():
        per.observe(batch, lambda k: desc)
    per.close()
    return per, desc


def test_windows_devices_and_trust_are_persisted(tmp_path, stage1):
    with Store(tmp_path / "t.db") as store:
        per, desc = _run(store, stage1, threshold=2.0)        # never fires
        c = store.counts()
        assert c["devices"] == 1 and c["windows"] == per.n_windows > 0
        assert c["alerts"] == c["actions"] == 0
        assert store.devices()[0]["device_id"] == desc.device_key


def test_live_windows_have_no_ground_truth_label(tmp_path, stage1):
    """0 would claim every live window was known benign."""
    with Store(tmp_path / "t.db") as store:
        _run(store, stage1, threshold=2.0)
        labels = {r[0] for r in store.cx.execute("SELECT label FROM windows")}
        assert labels == {None}


def test_escalations_write_alerts_actions_and_pin_evidence(tmp_path, stage1):
    with Store(tmp_path / "t.db") as store:
        rec = TraceRecorder(store, source="test")
        per, desc = _run(store, stage1, threshold=0.0, recorder=rec)   # always fires
        store.commit()
        acts = store.actions()
        assert per.n_actions == len(acts) >= 2
        assert {a["action"] for a in acts} >= {"THROTTLE", "DEAUTHORIZE"}
        assert all(a["rule"]["dry_run"] for a in acts)
        assert store.devices()[0]["trust_state"] == "revoked"
        pinned = store.cx.execute("SELECT COUNT(*) FROM urb_events WHERE pinned=1"
                                  ).fetchone()[0]
        assert pinned > 0, "an escalation did not pin the traffic behind it"


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

needs_models = pytest.mark.skipif(not (MODELS / "stage2.pt").exists(),
                                  reason="trained models not present (run make eval)")


@needs_models
def test_cli_replay_populates_the_store(tmp_path):
    from iudbms.live.cli import main
    db = tmp_path / "live.db"
    rc = main(["--replay", str(FIXTURE / "demo_capture.txt"),
               "--descriptors", str(FIXTURE / "demo_descriptors.json"),
               "--models", str(MODELS), "--db", str(db),
               "--max-windows", "3", "--log", "actions", "--color", "never"])
    assert rc == 0
    with Store(db) as store:
        c = store.counts()
        assert c["windows"] > 0 and c["devices"] > 0
        assert c["trace_runs"] == 1 and c["urb_events"] > 0
        mode = store.cx.execute("SELECT payload_mode FROM trace_runs").fetchone()[0]
        assert mode == "pseudonymous", "the default must not store keystrokes"


@needs_models
def test_cli_refuses_to_run_without_stage2(tmp_path):
    """A missing stage2.pt used to mean Stage 1 alone, silently."""
    from iudbms.live.cli import main
    partial = tmp_path / "models"
    partial.mkdir()
    for f in ("deploy.json", "stage1.pkl"):
        shutil.copy(MODELS / f, partial / f)
    args = ["--replay", str(FIXTURE / "demo_capture.txt"), "--models", str(partial),
            "--no-persist", "--max-windows", "1", "--log", "actions"]
    with pytest.raises(SystemExit, match="stage2.pt is missing"):
        main(args)
    assert main(args + ["--allow-stage1-only"]) == 0


@needs_models
def test_cli_refuses_models_from_another_contract(tmp_path):
    from iudbms.live.cli import main
    other = tmp_path / "models"
    shutil.copytree(MODELS, other)
    cfg = json.loads((other / "deploy.json").read_text())
    cfg["contract_version"] = "1.0.0"
    (other / "deploy.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="contract"):
        main(["--replay", str(FIXTURE / "demo_capture.txt"), "--models", str(other),
              "--no-persist", "--max-windows", "1"])


@pytest.mark.skipif(platform.system() == "Linux", reason="checks the non-Linux refusal")
def test_cli_refuses_live_capture_off_linux():
    from iudbms.live.cli import main
    with pytest.raises(SystemExit, match="needs Linux"):
        main(["--no-persist"])
