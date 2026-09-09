"""
L4 — control API (FastAPI). Endpoints per plan §4.6.

Read-only over the SQLite store, plus a policy endpoint. The one state-changing
endpoint (``POST /devices/{id}/authorize``) deliberately does **not** touch the
kernel: it records an operator decision. Actual sysfs writes are gated behind
``policy.engine.ENFORCEMENT_REVIEWED``, which is False.

    uvicorn iudbms.api.app:app --reload
    IUDBMS_DB=out/iudbms.db uvicorn iudbms.api.app:app
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..contract import CONTRACT_VERSION
from ..policy import ENFORCEMENT_REVIEWED, KILL_SWITCH_ENV, PolicyEngine
from ..storage import Store

__all__ = ["app", "get_db_path"]

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = _ROOT / "out" / "iudbms.db"
DASHBOARD = _ROOT / "dashboard" / "index.html"
METRICS = _ROOT / "out" / "metrics.json"

app = FastAPI(
    title="IUDBMS Control API",
    version=CONTRACT_VERSION,
    description="Behavioural trust for USB peripherals. Read-only over the "
                "window/alert store; enforcement is gated separately.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def get_db_path() -> Path:
    return Path(os.environ.get("IUDBMS_DB", DEFAULT_DB))


def deployed_policy() -> dict:
    """The operating point the persisted run actually used.

    Read from out/metrics.json rather than recomputed, so the dashboard shows
    the threshold that produced the stored alerts instead of inferring one.
    """
    try:
        m = json.loads(METRICS.read_text())
    except (OSError, ValueError):
        return {}
    return {
        "threshold": m.get("full_system", {}).get("threshold"),
        "target_fpr": m.get("config", {}).get("target_fpr"),
        "gate": m.get("config", {}).get("gate"),
        "deployed_features": m.get("deployed_features"),
    }


def _store() -> Store:
    path = get_db_path()
    if not path.exists():
        raise HTTPException(
            503, f"no database at {path}. Run `python3 -m eval.run_eval --persist`.")
    return Store(path)


class PolicyUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    k: int = Field(4, ge=1)
    n: int = Field(6, ge=1)
    dry_run: bool = True


class AuthorizeRequest(BaseModel):
    authorized: bool
    reason: str = ""


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the operator dashboard (plan §4.6 / milestone 9)."""
    if not DASHBOARD.exists():
        raise HTTPException(404, "dashboard/index.html not found")
    return FileResponse(DASHBOARD)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "policy": deployed_policy(),
        "contract_version": CONTRACT_VERSION,
        "database": str(get_db_path()),
        "database_present": get_db_path().exists(),
        "enforcement_reviewed": ENFORCEMENT_REVIEWED,
        "kill_switch_engaged": PolicyEngine.kill_switch_engaged(),
        "kill_switch_env": KILL_SWITCH_ENV,
    }


@app.get("/devices")
def devices() -> list[dict]:
    with _store() as s:
        return s.devices()


@app.get("/devices/{device_id}/timeline")
def timeline(device_id: str, limit: int = Query(5000, ge=1, le=50000)) -> dict:
    with _store() as s:
        if not any(d["device_id"] == device_id for d in s.devices()):
            raise HTTPException(404, f"unknown device {device_id}")
        return {"device_id": device_id, "windows": s.timeline(device_id, limit)}


@app.post("/devices/{device_id}/authorize")
def authorize(device_id: str, req: AuthorizeRequest) -> dict:
    """Record an operator trust decision.

    This does not write to /sys/bus/usb/.../authorized. Enforcement is gated
    behind policy.engine.ENFORCEMENT_REVIEWED, which is False until the safety
    interlocks have been reviewed.
    """
    with _store() as s:
        if not any(d["device_id"] == device_id for d in s.devices()):
            raise HTTPException(404, f"unknown device {device_id}")
        state = "observe" if req.authorized else "revoked"
        s.set_trust(device_id, state)
        s.commit()
    return {"device_id": device_id, "trust_state": state,
            "applied_to_kernel": False,
            "note": "operator decision recorded; sysfs writes are gated by "
                    "ENFORCEMENT_REVIEWED"}


@app.get("/alerts")
def alerts(device_id: str | None = None,
           limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
    with _store() as s:
        return s.alerts(device_id, limit)


@app.get("/actions")
def actions(device_id: str | None = None,
            limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
    with _store() as s:
        return s.actions(device_id, limit)


@app.post("/policy")
def set_policy(update: PolicyUpdate) -> dict:
    """Validate a policy change. Returns the effective policy.

    Kept stateless for now: the running engine is constructed per evaluation,
    so this endpoint exists to validate and echo a policy rather than to mutate
    a long-lived detector that does not yet exist outside the harness.
    """
    if update.k > update.n:
        raise HTTPException(422, f"k ({update.k}) must be <= n ({update.n})")
    return {"threshold": update.threshold, "k": update.k, "n": update.n,
            "dry_run": update.dry_run, "enforcement_reviewed": ENFORCEMENT_REVIEWED}


@app.get("/stats")
def stats() -> dict:
    with _store() as s:
        return s.counts()
