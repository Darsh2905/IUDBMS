"""
L3/L4 — closed-loop enforcement (novelty pillar N4).

LBM and USBGuard both enforce *static, human-authored* rules. Here the decision
boundary is **learned**: ``threshold`` is not a magic number but the Stage-2
probability that produces a target false-positive rate on held-out benign
traffic, and every emitted action carries the machine-generated evidence that
triggered it.

Escalation ladder::

    OBSERVE  --k-of-n windows above threshold-->  THROTTLE
    THROTTLE --sustained (2k hits)------------->  DEAUTHORIZE

The **k-of-n persistence gate** is the main false-positive control and is what
makes automatic revocation safe enough to contemplate: a single anomalous
window never triggers anything. Per-window FPR compounds across a session — at
0.5 % per window over ~70 windows the prototype produced a 9.1 % benign session
FP rate; 0.1 % with a 4-of-6 gate brought it to 3.0 % with no loss of detection.

Safety
------
Four independent interlocks, all of which must be open before a single byte
reaches sysfs:

1. ``dry_run=True`` by default. Actions are generated and logged, never executed.
2. A device allowlist protecting the boot keyboard and root-filesystem device.
   Revoking the only keyboard on a machine locks the operator out.
3. A kill switch checked on every tick (``IUDBMS_KILL_SWITCH`` in the
   environment), so enforcement can be disabled without restarting anything.
4. ``ENFORCEMENT_REVIEWED``, which is **False** and must stay that way until the
   safety checks have been reviewed by a human. While it is False, executing an
   action raises rather than writing to sysfs, regardless of the other three.

Interlock 4 is not belt-and-braces: writing 0 to a device's ``authorized`` node
on a developer laptop can detach the keyboard being typed on. The plan puts
this work on a snapshotted VM (§3, §8) for exactly that reason.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from ..contract import DeviceDescriptor

__all__ = [
    "PolicyEngine", "DeviceContext", "Action", "TrustState",
    "ENFORCEMENT_REVIEWED", "KILL_SWITCH_ENV", "EnforcementBlocked",
]

#: "unmodelled": the device declares an interface type the model was never trained on
#: (policy.support). It is scored and stored but never escalated.
TrustState = Literal["observe", "throttled", "revoked", "unmodelled"]

#: MUST remain False until a human has reviewed the safety interlocks.
#: Flipping this is what allows a real write to /sys/bus/usb/.../authorized.
ENFORCEMENT_REVIEWED = False

KILL_SWITCH_ENV = "IUDBMS_KILL_SWITCH"


class EnforcementBlocked(RuntimeError):
    """Raised when an action was suppressed by a safety interlock."""


@dataclass
class Action:
    action: Literal["THROTTLE", "DEAUTHORIZE"]
    severity: Literal["high", "critical"]
    ts: float
    device_id: str
    rule: dict

    def as_row(self) -> tuple:
        return (self.device_id, self.ts, self.action, self.severity)


@dataclass
class DeviceContext:
    """Per-device enforcement state. One of these per monitored device."""

    device_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    state: TrustState = "observe"
    hits: int = 0

    @property
    def hits_in_window(self) -> int:
        return sum(self.history)


class PolicyEngine:
    def __init__(
        self,
        threshold: float,
        k: int = 4,
        n: int = 6,
        *,
        dry_run: bool = True,
        allowlist: set[str] | None = None,
        support=None,
    ):
        if not 0 < k <= n:
            raise ValueError(f"need 0 < k <= n, got k={k}, n={n}")
        self.threshold = threshold
        self.k, self.n = k, n
        self.dry_run = dry_run
        # Devices that must never be throttled or revoked, whatever the model
        # says. The boot keyboard belongs here on every real deployment.
        self.allowlist: set[str] = set(allowlist or ())
        #: policy.support.ModelSupport, or None to judge every device (the evaluation
        #: harness, whose simulated devices are all inside the training support).
        self.support = support

    def new_device(self, device_id: str) -> DeviceContext:
        return DeviceContext(device_id=device_id, history=deque(maxlen=self.n))

    @staticmethod
    def kill_switch_engaged() -> bool:
        return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

    def step(
        self,
        ctx: DeviceContext,
        ts: float,
        stage1: float,
        stage2: float,
        evidence: list[tuple[str, float]],
        descriptor: DeviceDescriptor,
    ) -> Action | None:
        """Advance one window. Returns an Action when the ladder escalates."""
        ctx.history.append(1 if stage2 >= self.threshold else 0)
        if self.support is not None and not self.support.covers(descriptor):
            ctx.state = "unmodelled"          # outside the model's competence: never escalate
            return None
        if ctx.hits_in_window < self.k:
            return None

        if ctx.device_id in self.allowlist:
            return None                       # interlock 2
        if self.kill_switch_engaged():
            return None                       # interlock 3

        ctx.hits += 1
        if ctx.hits >= 2 * self.k and ctx.state != "revoked":
            action, severity = "DEAUTHORIZE", "critical"
            ctx.state = "revoked"
        elif ctx.state == "observe":
            action, severity = "THROTTLE", "high"
            ctx.state = "throttled"
        else:
            return None

        return Action(
            action=action, severity=severity, ts=ts, device_id=ctx.device_id,
            rule=self._rule(ctx, action, ts, stage1, stage2, evidence, descriptor),
        )

    def _rule(self, ctx, action, ts, stage1, stage2, evidence, descriptor: DeviceDescriptor) -> dict:
        """The generated rule. This object is the pillar-N4 artefact: it is what
        makes enforcement *learned* rather than a static table entry, because
        the trigger, the threshold and the evidence all come from the model."""
        return {
            "device": ctx.device_id,
            "action": action,
            "trigger": f"stage2>={self.threshold:.3f} in {self.k}/{self.n} windows",
            "ts": round(float(ts), 3),
            "stage1": round(float(stage1), 3),
            "stage2": round(float(stage2), 3),
            "evidence_features": evidence,
            "declared_class": [hex(i.bInterfaceClass) for i in descriptor.interfaces],
            "kernel_cmd": self.kernel_command(action, descriptor, ctx.device_id),
            "dry_run": self.dry_run,
            "enforcement_reviewed": ENFORCEMENT_REVIEWED,
        }

    @staticmethod
    def kernel_command(action: str, descriptor: DeviceDescriptor, device_id: str) -> str:
        if action == "DEAUTHORIZE":
            return (f"echo 0 > /sys/bus/usb/devices/"
                    f"{descriptor.bus}-{descriptor.devnum}/authorized")
        return f"iudbms-ctl throttle {device_id} --max-reports-per-sec 25"

    # ------------------------------------------------------------------
    # execution — deliberately inert
    # ------------------------------------------------------------------

    def execute(self, action: Action) -> str:
        """Would apply ``action`` to the kernel. Currently always refuses.

        Every interlock is checked here rather than at the call site, so there
        is exactly one place a real write could ever originate.
        """
        if self.dry_run:
            return f"DRY-RUN: {action.rule['kernel_cmd']}"
        if action.device_id in self.allowlist:
            raise EnforcementBlocked(f"{action.device_id} is on the allowlist")
        if self.kill_switch_engaged():
            raise EnforcementBlocked(f"kill switch engaged ({KILL_SWITCH_ENV})")
        if not ENFORCEMENT_REVIEWED:
            raise EnforcementBlocked(
                "Real enforcement is disabled: policy.engine.ENFORCEMENT_REVIEWED "
                "is False. Writing 0 to a device's 'authorized' node can detach "
                "the keyboard in use. Do not flip this until the safety "
                "interlocks have been reviewed and the target is a snapshotted VM."
            )
        raise NotImplementedError(
            "sysfs write path is intentionally unimplemented; see ENFORCEMENT_REVIEWED"
        )
