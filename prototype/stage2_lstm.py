"""
IUDBMS - Stage 2: temporal sequence model.

Novelty pillar 3. Stage 1 is memoryless: it can only ask "is this second odd?".
A staged BadUSB payload that idles as a human-driven keyboard for 40 s and then
fires looks, second by second, like two individually plausible devices. Only the
*transition* is anomalous. Stage 2 consumes a rolling sequence of the last T
window vectors (plus their Stage-1 scores) and emits P(malicious) for the newest
window.

Implemented as a single-layer LSTM with explicit BPTT in NumPy so the prototype
runs with zero GPU/PyTorch dependency; the production build swaps this class for
the PyTorch module of identical shape (see implementation plan, Phase 4).
"""

import numpy as np

SEQ_LEN = 10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def build_sequences(X, T=SEQ_LEN):
    """Left-padded rolling sequences: one sequence per window."""
    W, D = X.shape
    out = np.empty((W, T, D), dtype=np.float64)
    for w in range(W):
        lo = max(0, w - T + 1)
        chunk = X[lo:w + 1]
        if len(chunk) < T:
            chunk = np.vstack([np.repeat(chunk[:1], T - len(chunk), axis=0), chunk])
        out[w] = chunk
    return out


class LSTMClassifier:
    def __init__(self, d_in, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden + d_in)
        shape = (hidden, d_in + hidden)
        self.p = {
            "Wf": rng.normal(0, s, shape), "bf": np.ones(hidden),     # forget-bias 1
            "Wi": rng.normal(0, s, shape), "bi": np.zeros(hidden),
            "Wg": rng.normal(0, s, shape), "bg": np.zeros(hidden),
            "Wo": rng.normal(0, s, shape), "bo": np.zeros(hidden),
            "Wy": rng.normal(0, s, (hidden,)), "by": np.zeros(1),
        }
        self.H, self.D = hidden, d_in
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.mu = None
        self.sd = None

    # ---------- forward / backward ----------

    def _forward(self, X):
        B, T, D = X.shape
        H, p = self.H, self.p
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        for t in range(T):
            z = np.hstack([X[:, t, :], h])
            f = sigmoid(z @ p["Wf"].T + p["bf"])
            i = sigmoid(z @ p["Wi"].T + p["bi"])
            g = np.tanh(z @ p["Wg"].T + p["bg"])
            o = sigmoid(z @ p["Wo"].T + p["bo"])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            cache.append((z, f, i, g, o, c_prev, c, tc))
        logits = h @ p["Wy"] + p["by"]
        return logits, h, cache

    def _backward(self, X, y, logits, h_last, cache):
        B, T, D = X.shape
        H, p = self.H, self.p
        prob = sigmoid(logits)
        dlogit = (prob - y) / B
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wy"] = h_last.T @ dlogit
        grads["by"] = np.array([dlogit.sum()])

        dh = np.outer(dlogit, p["Wy"])
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            z, f, i, g, o, c_prev, c, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dcp = dc * f

            zf = df * f * (1 - f)
            zi = di * i * (1 - i)
            zg = dg * (1 - g ** 2)
            zo = do * o * (1 - o)

            grads["Wf"] += zf.T @ z; grads["bf"] += zf.sum(0)
            grads["Wi"] += zi.T @ z; grads["bi"] += zi.sum(0)
            grads["Wg"] += zg.T @ z; grads["bg"] += zg.sum(0)
            grads["Wo"] += zo.T @ z; grads["bo"] += zo.sum(0)

            dz = zf @ p["Wf"] + zi @ p["Wi"] + zg @ p["Wg"] + zo @ p["Wo"]
            dh = dz[:, D:]
            dc = dcp
        return grads

    def _adam(self, grads, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for k in self.p:
            g = np.clip(grads[k], -clip, clip)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    # ---------- api ----------

    def fit(self, X, y, epochs=20, batch=128, lr=0.01, seed=0, verbose=True, val=None):
        self.mu = X.reshape(-1, X.shape[2]).mean(0)
        self.sd = X.reshape(-1, X.shape[2]).std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        rng = np.random.default_rng(seed)
        n = len(Xs)
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                logits, h_last, cache = self._forward(Xs[b])
                pr = sigmoid(logits)
                tot += float(-(y[b] * np.log(pr + 1e-9) +
                               (1 - y[b]) * np.log(1 - pr + 1e-9)).sum())
                self._adam(self._backward(Xs[b], y[b], logits, h_last, cache), lr=lr)
            line = f"  epoch {ep + 1:02d}/{epochs}  loss={tot / n:.4f}"
            if val is not None:
                pv = self.predict_proba(val[0])
                acc = ((pv > 0.5).astype(int) == val[1]).mean()
                line += f"  val_acc={acc:.4f}"
            hist.append(tot / n)
            if verbose:
                print(line, flush=True)
        return hist

    def predict_proba(self, X, batch=512):
        Xs = (X - self.mu) / self.sd
        out = np.empty(len(Xs))
        for s in range(0, len(Xs), batch):
            logits, _, _ = self._forward(Xs[s:s + batch])
            out[s:s + batch] = sigmoid(logits)
        return out
