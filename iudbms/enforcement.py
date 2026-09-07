"""
IUDBMS - Layer 5: closed-loop enforcement.

Novelty pillar 4. LBM and USBGuard both enforce *static, human-authored* rules.
Here the decision boundary is learned: `threshold` is not a magic number but the
Stage-2 probability that produces a target false-positive rate on held-out benign
traffic, and the emitted rule carries the machine-generated evidence.

Escalation ladder (k-of-n hysteresis suppresses single-window flaps):

  k of n windows above threshold          -> ALERT + RATE_LIMIT
  2k consecutive-ish windows sustained    -> DEAUTHORIZE

DEAUTHORIZE maps to the one-line kernel action:
    echo 0 > /sys/bus/usb/devices/<bus>-<port>/authorized
which drops the device without unbinding the whole controller. The prototype
emits the command instead of executing it (dry_run=True).
"""

from collections import deque


class PolicyEngine:
    def __init__(self, threshold, k=3, n=5, dry_run=True):
        self.threshold = threshold
        self.k, self.n = k, n
        self.dry_run = dry_run

    def new_device(self):
        return {"hist": deque(maxlen=self.n), "state": "learning", "hits": 0}

    def step(self, ctx, device_id, ts, s1, s2, evidence, descriptor):
        """Returns None or an action dict."""
        ctx["hist"].append(1 if s2 >= self.threshold else 0)
        if sum(ctx["hist"]) < self.k:
            if ctx["state"] == "learning" and ts > 5:
                ctx["state"] = "trusted"
            return None

        ctx["hits"] += 1
        if ctx["hits"] >= 2 * self.k and ctx["state"] != "revoked":
            action, severity = "DEAUTHORIZE", "critical"
            ctx["state"] = "revoked"
        elif ctx["state"] in ("learning", "trusted"):
            action, severity = "RATE_LIMIT", "high"
            ctx["state"] = "restricted"
        else:
            return None

        rule = {
            "device": device_id,
            "action": action,
            "trigger": f"stage2>={self.threshold:.3f} in {self.k}/{self.n} windows",
            "stage1": round(float(s1), 3),
            "stage2": round(float(s2), 3),
            "evidence_features": evidence,
            "declared_class": [hex(i["bInterfaceClass"]) for i in descriptor["interfaces"]],
            "kernel_cmd": (f"echo 0 > /sys/bus/usb/devices/{descriptor['bus']}-"
                           f"{descriptor['devnum']}/authorized" if action == "DEAUTHORIZE"
                           else f"iudbms-ctl throttle {device_id} --max-reports-per-sec 25"),
            "dry_run": self.dry_run,
        }
        return {"action": action, "severity": severity, "ts": ts, "rule": rule}
