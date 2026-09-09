"""Write an evaluation run into SQLite so the API and dashboard can serve it."""

from __future__ import annotations

from pathlib import Path

from iudbms.policy import PolicyEngine
from iudbms.stage1 import class_group
from iudbms.storage import Store

from .pipeline import GATE_K, GATE_N

__all__ = ["persist_run"]


def persist_run(db_path: Path, sessions, result) -> dict[str, int]:
    """Populate devices/windows/alerts/actions from a completed run.

    The ``baselines`` table stays empty and will report 0. That is not a bug
    here: a DeviceBaseline is per-device online state belonging to the live
    detector, and the evaluation harness constructs a fresh one per session and
    discards it. It will be populated by the streaming detector (not yet built),
    which is the only component that owns a long-lived baseline.
    """
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.unlink()

    eng = PolicyEngine(threshold=result.threshold, k=GATE_K, n=GATE_N)
    with Store(db_path) as store:
        for s in sessions:
            if s.sid not in result.stage2:
                continue
            store.add_device(s.sid, s.descriptor, class_group(s.descriptor))
            store.add_windows(s.sid, s.t, result.stage1[s.sid], result.stage2[s.sid],
                              s.y, s.X)

            ctx = eng.new_device(s.sid)
            for i, t in enumerate(s.t):
                a = eng.step(ctx, float(t), float(result.stage1[s.sid][i]),
                             float(result.stage2[s.sid][i]),
                             result.evidence[s.sid][i], s.descriptor)
                if a is not None:
                    aid = store.add_alert(s.sid, a.ts, a.rule["stage1"], a.rule["stage2"],
                                          a.severity, result.evidence[s.sid][i])
                    store.add_action(aid, s.sid, a.ts, a.action, a.rule)
            store.set_trust(s.sid, ctx.state)
        counts = store.counts()
    return counts
