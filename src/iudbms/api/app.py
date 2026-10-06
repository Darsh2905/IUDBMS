"""
L4 — control API (FastAPI). Endpoints per plan §4.6.

Read-only over the SQLite store, plus a policy endpoint. The one state-changing
endpoint (``POST /devices/{id}/authorize``) deliberately does **not** touch the
kernel: it records an operator decision. Actual sysfs writes are gated behind
``policy.engine.ENFORCEMENT_REVIEWED``, which is False.

    iudbms-api                                   # 127.0.0.1:8000
    IUDBMS_DB=out/iudbms.db uvicorn iudbms.api.app:app --reload

Configuration, all from the environment so one file configures the deployed
units in ``deploy/``:

  IUDBMS_DB            the SQLite store (default out/iudbms.db)
  IUDBMS_MODELS        the deployed model directory, whose deploy.json is the
                       operating point the monitor runs (default out/models)
  IUDBMS_CORS_ORIGINS  comma-separated origins allowed cross-origin. Unset by
                       default: the dashboard is served from this app, so it is
                       same-origin and needs none.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from importlib.resources import files

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..contract import CONTRACT_VERSION
from ..policy import ENFORCEMENT_REVIEWED, KILL_SWITCH_ENV, PolicyEngine
from ..storage import Store

__all__ = ["app", "get_db_path"]

# Package data, not a path relative to this file's position in a checkout.
# `parents[3]` located the repository root from a source tree and pointed at
# nothing in particular once installed into site-packages, so the dashboard
# 404'd in any real deployment.
DASHBOARD = Path(str(files("iudbms.api").joinpath("static", "index.html")))
DEFAULT_DB = Path("out") / "iudbms.db"
DEFAULT_MODELS = Path("out") / "models"

app = FastAPI(
    title="IUDBMS Control API",
    version=CONTRACT_VERSION,
    description="Behavioural trust for USB peripherals. Read-only over the "
                "window/alert store; enforcement is gated separately.",
)
# No cross-origin access unless configured. This was allow_origins=["*"] on an
# API whose POST /devices/{id}/authorize records operator trust decisions: any
# web page the operator happened to visit could have called it. The dashboard is
# served by this app and is same-origin, so it needs no CORS at all.
_CORS = [o.strip() for o in os.environ.get("IUDBMS_CORS_ORIGINS", "").split(",")
         if o.strip()]
if _CORS:
    app.add_middleware(CORSMiddleware, allow_origins=_CORS,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def get_db_path() -> Path:
    return Path(os.environ.get("IUDBMS_DB", DEFAULT_DB))


def get_models_dir() -> Path:
    return Path(os.environ.get("IUDBMS_MODELS", DEFAULT_MODELS))


def deployed_policy() -> dict:
    """The operating point the monitor actually runs.

    Read from the deployed models' deploy.json -- the same file the monitor
    loads -- rather than recomputed, so the dashboard shows the threshold that
    produced the stored alerts instead of inferring one.
    """
    try:
        cfg = json.loads((get_models_dir() / "deploy.json").read_text())
    except (OSError, ValueError):
        return {}
    return {
        "threshold": cfg.get("threshold"),
        "target_fpr": cfg.get("target_fpr"),
        "gate": f"{cfg.get('k')}-of-{cfg.get('n')}",
        "deployed_features": len(cfg.get("feature_idx", [])),
        "contract_version": cfg.get("contract_version"),
    }


def _store() -> Store:
    path = get_db_path()
    if not path.exists():
        raise HTTPException(
            503, f"no database at {path}. Start the monitor (iudbms-monitor, or the "
                 f"iudbms-monitor service), or populate it from an evaluation with "
                 f"`python3 -m eval.run_eval --persist`.")
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
        raise HTTPException(404, f"dashboard not found at {DASHBOARD}")
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


@app.get("/traces/runs")
def trace_run_list() -> list[dict]:
    """Every recording run: when, from what, and in which payload mode."""
    from ..storage import trace_runs
    with _store() as s:
        return trace_runs(s)


@app.get("/devices/{device_id}/trace")
def device_trace(device_id: str, limit: int = Query(500, ge=1, le=5000)) -> dict:
    """The device's most recent raw URBs, newest first -- the traffic behind its
    scores. Payloads are as the recording run stored them (``payload_mode`` on
    every row): pseudonyms by default, never clear text unless configured."""
    with _store() as s:
        rows = s.trace(device_id, limit)
        if not rows and not any(d["device_id"] == device_id for d in s.devices()):
            raise HTTPException(404, f"unknown device {device_id}")
        return {"device_id": device_id, "events": rows}
