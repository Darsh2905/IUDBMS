"""
Corpus construction and session-stratified splitting.

**Split discipline (plan §5): split by session, stratified by profile — never
by window.** Windows from one session appearing in both train and test is the
most likely way this project reports fake accuracy, because consecutive windows
of a session are near-duplicates of each other.

Sessions are generated, featurised and discarded one at a time, so peak memory
stays flat instead of growing with the number of URBs observed. A live
deployment behaves the same way (a bounded ring buffer drained by the feature
extractor).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from iudbms.capture.simulator import SimulatedSource, Simulator
from iudbms.contract import CONTRACT_VERSION, DeviceDescriptor
from iudbms.features.extractor import WindowFeatureExtractor, window_events

__all__ = ["SessionData", "Corpus", "build_corpus", "stratified_split",
           "save_corpus", "load_corpus", "simulator_fingerprint"]


@dataclass
class SessionData:
    sid: str
    profile: str
    label: int
    descriptor: DeviceDescriptor
    X: np.ndarray          # [W, 28] float32
    t: np.ndarray          # [W] window start times
    y: np.ndarray          # [W] per-window label
    onset: float | None
    duration: float


@dataclass
class Corpus:
    sessions: list[SessionData]
    n_lines: int
    n_events: int

    @property
    def n_windows(self) -> int:
        return sum(len(s.X) for s in self.sessions)

    def by_sid(self) -> dict[str, SessionData]:
        return {s.sid: s for s in self.sessions}


def build_corpus(
    n_benign: int = 210,
    n_attack: int = 140,
    seed: int = 7,
    *,
    verbose: bool = True,
) -> Corpus:
    sim = Simulator(seed=seed)
    sessions: list[SessionData] = []
    n_lines = n_events = 0

    for sid, profile in sim.plan(n_benign=n_benign, n_attack=n_attack):
        s = sim.session(sid, profile)
        # One extractor per session. It carries payload history for
        # payload_seq_novelty (contract 2.0.0); sharing one across sessions
        # would make each device inherit the previous device's history and
        # read as suspiciously familiar traffic.
        ex = WindowFeatureExtractor()
        n_lines += len(s.lines)
        events = list(SimulatedSource(s).events())
        n_events += len(events)

        X, t, y = [], [], []
        for t0, bucket in window_events(events, duration=s.duration):
            X.append(ex.extract(bucket, t0, s.descriptor))
            t.append(t0)
            # Plan §4.4: the label applies from attack onset onward, so
            # pre-onset windows of a malicious session are labelled BENIGN.
            # This is what forces Stage 2 to learn a transition rather than
            # device identity.
            malicious = s.label == 1 and s.onset is not None and (t0 + ex.WINDOW) > s.onset
            y.append(int(malicious))

        s.lines = []      # release the raw capture immediately
        sessions.append(SessionData(
            sid=sid, profile=profile, label=s.label, descriptor=s.descriptor,
            X=np.vstack(X), t=np.asarray(t), y=np.asarray(y),
            onset=s.onset, duration=s.duration,
        ))

    if verbose:
        n_b = sum(1 for s in sessions if s.label == 0)
        print(f"  sessions        : {len(sessions)}  ({n_b} benign / "
              f"{len(sessions) - n_b} malicious)")
        print(f"  usbmon lines    : {n_lines:,}")
        print(f"  URB events      : {n_events:,}")
    return Corpus(sessions=sessions, n_lines=n_lines, n_events=n_events)


def stratified_split(
    sessions: list[SessionData],
    seed: int = 7,
    test_frac: float = 0.30,
    val_frac: float = 0.20,
) -> tuple[list[SessionData], list[SessionData], list[SessionData]]:
    """Session-level split, stratified by profile.

    Returns ``(fit, val, test)``. ``val`` is carved from the training half and
    is what every threshold is calibrated on; ``test`` is touched exactly once,
    at reporting time.
    """
    rng = np.random.default_rng(seed)
    by_profile: dict[str, list[SessionData]] = defaultdict(list)
    for s in sessions:
        by_profile[s.profile].append(s)

    train: list[SessionData] = []
    test: list[SessionData] = []
    for profile in sorted(by_profile):
        group = by_profile[profile]
        idx = rng.permutation(len(group))
        cut = int(round((1 - test_frac) * len(group)))
        train += [group[i] for i in idx[:cut]]
        test += [group[i] for i in idx[cut:]]

    idx = rng.permutation(len(train))
    n_val = int(round(val_frac * len(train)))
    val = [train[i] for i in idx[:n_val]]
    fit = [train[i] for i in idx[n_val:]]

    overlap = ({s.sid for s in fit} | {s.sid for s in val}) & {s.sid for s in test}
    assert not overlap, f"session leaked across the split: {overlap}"
    return fit, val, test


def simulator_fingerprint() -> str:
    """Hash of the code that generates the corpus.

    A cache that only records the contract version will happily serve traffic
    generated by a simulator that has since changed -- which happened during
    development and silently produced results for the previous adversary. The
    fingerprint covers the simulator and the feature extractor, the two things
    that determine what a cached window vector contains.
    """
    import hashlib
    from pathlib import Path as _P

    import iudbms.capture.simulator as sim
    import iudbms.features.extractor as ext

    h = hashlib.sha256()
    for mod in (sim, ext):
        h.update(_P(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


def save_corpus(corpus: Corpus, path) -> None:
    """Cache a built corpus so a hyperparameter sweep does not regenerate it.

    Descriptors are stored structurally (not pickled) so a cache written by one
    version of the contract cannot silently deserialise against another.
    """
    import json
    from pathlib import Path

    path = Path(path)
    meta = {
        "contract_version": CONTRACT_VERSION,
        "simulator_fingerprint": simulator_fingerprint(),
        "n_lines": corpus.n_lines,
        "n_events": corpus.n_events,
        "sessions": [
            {
                "sid": s.sid, "profile": s.profile, "label": s.label,
                "onset": s.onset, "duration": s.duration,
                "descriptor": {
                    "bus": s.descriptor.bus, "devnum": s.descriptor.devnum,
                    "vid": s.descriptor.vid, "pid": s.descriptor.pid,
                    "bDeviceClass": s.descriptor.bDeviceClass,
                    "serial_present": s.descriptor.serial_present,
                    "manufacturer": s.descriptor.manufacturer,
                    "product": s.descriptor.product,
                    "interfaces": [
                        {"cls": i.bInterfaceClass, "sub": i.bInterfaceSubClass,
                         "proto": i.bInterfaceProtocol, "first_seen_ts": i.first_seen_ts,
                         "endpoints": [{"addr": e.addr, "type": e.type,
                                        "interval": e.interval} for e in i.endpoints]}
                        for i in s.descriptor.interfaces
                    ],
                },
            }
            for s in corpus.sessions
        ],
    }
    arrays = {}
    for i, s in enumerate(corpus.sessions):
        arrays[f"X{i}"], arrays[f"t{i}"], arrays[f"y{i}"] = s.X, s.t, s.y
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_corpus(path) -> Corpus:
    import json

    from iudbms.contract import Endpoint, Interface

    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    if meta["contract_version"].split(".")[0] != CONTRACT_VERSION.split(".")[0]:
        raise RuntimeError(
            f"corpus cache was built against contract {meta['contract_version']}, "
            f"running {CONTRACT_VERSION}. Delete the cache and rebuild."
        )
    want = simulator_fingerprint()
    got = meta.get("simulator_fingerprint")
    if got != want:
        raise RuntimeError(
            f"corpus cache was generated by a different simulator or feature "
            f"extractor (fingerprint {got}, current {want}).\n"
            f"Evaluating it would report results for code that no longer exists. "
            f"Delete the cache file and re-run `make eval`."
        )

    sessions = []
    for i, m in enumerate(meta["sessions"]):
        d = m["descriptor"]
        desc = DeviceDescriptor(
            bus=d["bus"], devnum=d["devnum"], vid=d["vid"], pid=d["pid"],
            bDeviceClass=d["bDeviceClass"], serial_present=d["serial_present"],
            manufacturer=d["manufacturer"], product=d["product"],
            interfaces=tuple(
                Interface(bInterfaceClass=f["cls"], bInterfaceSubClass=f["sub"],
                          bInterfaceProtocol=f["proto"],
                          endpoints=tuple(Endpoint(addr=e["addr"], type=e["type"],
                                                   interval=e["interval"])
                                          for e in f["endpoints"]),
                          first_seen_ts=f["first_seen_ts"])
                for f in d["interfaces"]),
        )
        sessions.append(SessionData(
            sid=m["sid"], profile=m["profile"], label=m["label"], descriptor=desc,
            X=blob[f"X{i}"], t=blob[f"t{i}"], y=blob[f"y{i}"],
            onset=m["onset"], duration=m["duration"]))
    return Corpus(sessions=sessions, n_lines=meta["n_lines"], n_events=meta["n_events"])
