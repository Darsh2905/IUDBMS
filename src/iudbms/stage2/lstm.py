"""
L3 Stage 2 — temporal sequence model (novelty pillar N3).

Stage 1 is memoryless: it can only ask "is this second odd?". A staged BadUSB
payload that idles as a human-driven keyboard for 40 s and then fires looks,
second by second, like two individually plausible devices. Only the
**transition** is anomalous. Stage 2 consumes a rolling sequence of the last
T window vectors (each augmented with its Stage-1 score) and emits
P(malicious) for the newest window.

Labelling (plan §4.4)
---------------------
The label applies **from attack onset onward**, so pre-onset windows of a
malicious session are labelled benign. This is what forces the model to learn a
transition rather than device identity: the same device, same descriptor, same
VID:PID appears on both sides of the boundary.

Implementation note
-------------------
This replaces the prototype's hand-written NumPy BPTT, which existed only
because PyTorch was unavailable in that sandbox — it was never a design choice.
Architecture is unchanged (single layer, hidden 32, sigmoid head) so the
prototype's numbers remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..contract import CONTRACT_VERSION

__all__ = ["LSTMDetector", "build_sequences", "SEQ_LEN", "TrainConfig"]

SEQ_LEN = 10


def build_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Left-padded rolling sequences: one sequence per window.

    Window *w* yields the sequence ``X[w-seq_len+1 : w+1]``, left-padded by
    repeating the earliest available row. Padding by repetition rather than
    zeros keeps early windows on the same scale as later ones, so a device's
    first seconds do not look like a discontinuity to the model.

    Strictly causal: the sequence for window *w* never contains a row from
    after *w*, so training and live inference see identical inputs.
    """
    X = np.asarray(X, dtype=np.float32)
    W, D = X.shape
    out = np.empty((W, seq_len, D), dtype=np.float32)
    for w in range(W):
        lo = max(0, w - seq_len + 1)
        chunk = X[lo:w + 1]
        if len(chunk) < seq_len:
            pad = np.repeat(chunk[:1], seq_len - len(chunk), axis=0)
            chunk = np.vstack([pad, chunk])
        out[w] = chunk
    return out


@dataclass
class TrainConfig:
    epochs: int = 14
    batch_size: int = 128
    lr: float = 2e-3
    hidden: int = 32
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    seed: int = 7


class LSTMDetector(nn.Module):
    """Single-layer LSTM over K-window sequences, sigmoid head on the last step."""

    def __init__(self, d_in: int, hidden: int = 32, seed: int = 7):
        super().__init__()
        torch.manual_seed(seed)
        self.d_in = d_in
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        # Forget-gate bias to 1: the standard trick to keep gradients flowing
        # across the sequence early in training.
        with torch.no_grad():
            for name, p in self.lstm.named_parameters():
                if "bias" in name:
                    p[hidden:2 * hidden].fill_(1.0)
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.contract_version = CONTRACT_VERSION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> logits (B,)"""
        x = (x - self.mu) / self.sd
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

    # ------------------------------------------------------------------
    # training / inference
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cfg: TrainConfig | None = None,
        val: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        cfg = cfg or TrainConfig()
        torch.manual_seed(cfg.seed)
        g = torch.Generator().manual_seed(cfg.seed)

        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
        flat = Xt.reshape(-1, Xt.shape[2])
        with torch.no_grad():
            self.mu.copy_(flat.mean(0))
            self.sd.copy_(flat.std(0).clamp_min(1e-6))

        # Class weighting: post-onset windows are the minority, and an
        # unweighted model can reach high accuracy by predicting "benign".
        n_pos = float(yt.sum())
        pos_weight = torch.tensor([(len(yt) - n_pos) / max(n_pos, 1.0)])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        history: list[dict] = []
        n = len(Xt)
        for ep in range(cfg.epochs):
            self.train()
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for s in range(0, n, cfg.batch_size):
                idx = perm[s:s + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                logits = self(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
                opt.step()
                total += loss.detach().item() * len(idx)

            rec = {"epoch": ep + 1, "loss": total / n}
            if val is not None:
                pv = self.predict_proba(val[0])
                yv = np.asarray(val[1])
                rec["val_acc"] = float(((pv > 0.5).astype(int) == yv).mean())
                # Accuracy alone is misleading under class imbalance; recall on
                # the post-onset class is what Stage 2 exists to deliver.
                if yv.sum() > 0:
                    rec["val_recall"] = float(((pv > 0.5) & (yv == 1)).sum() / yv.sum())
            history.append(rec)
            if verbose:
                extra = "  ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                  if k not in ("epoch", "loss"))
                print(f"  epoch {rec['epoch']:02d}/{cfg.epochs}  "
                      f"loss={rec['loss']:.4f}  {extra}".rstrip(), flush=True)
        return history

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        out = np.empty(len(Xt), dtype=np.float64)
        for s in range(0, len(Xt), batch):
            out[s:s + batch] = torch.sigmoid(self(Xt[s:s + batch])).numpy()
        return out

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "d_in": self.d_in,
            "hidden": self.hidden,
            "contract_version": self.contract_version,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> LSTMDetector:
        blob = torch.load(path, weights_only=False)
        got, want = blob.get("contract_version"), CONTRACT_VERSION
        if got is not None and got.split(".")[0] != want.split(".")[0]:
            raise RuntimeError(
                f"checkpoint was trained against contract {got}, running {want}. "
                f"The feature vector changed shape; retrain rather than load."
            )
        m = cls(d_in=blob["d_in"], hidden=blob["hidden"])
        m.load_state_dict(blob["state_dict"])
        m.contract_version = got or want
        return m
