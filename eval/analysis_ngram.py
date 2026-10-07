"""
A long-repeat feature for replayed scripts, as an *experimental* extractor column.

``payload_seq_novelty`` (contract 2.0.0) measures the fraction of payload
*transitions* (2-grams) not seen earlier in the session. It catches a replayed script
only while benign typing never repeats a 2-gram; real typing repeats 2-grams
constantly, so under realistic benign text the loop attack falls to ~7% detection.

What distinguishes a replayed script from language is not 2-gram repetition but
**long** exact repetition. A script of 24-48 keys that cycles repeats *every*
subsequence after its first cycle; language repeats common words and short
phrases, but a window full of long exact repeats is rare. ``repeat_ngram`` is the
fraction of this window's N-grams (N = 12 consecutive reports) that already
occurred earlier in the session, judged causally.

**Privacy-compatible by construction.** It compares whole 8-byte reports for
equality and nothing else. The trace store pseudonymises each report with a
length-preserving injective permutation (storage/traces.py), under which equality
survives and byte structure does not, so a feature that parsed key codes out of a
report (a keystroke n-gram, say) would break bit-exact re-featurisation of stored
traces and require storing keystrokes in the clear.

Experimental: appended as column 30, past the frozen contract. Adopting it would be a
contract bump (4.0.0) and a retrain of every checkpoint.
"""

from __future__ import annotations

import numpy as np

from iudbms.features.extractor import COUNTED_KINDS, WindowFeatureExtractor

EXT_COL = 30            # index of the appended column (the frozen schema ends at 29)
MAX_REMEMBERED = 16384  # bound memory on devices that emit endless distinct payloads


class NGramExtractor(WindowFeatureExtractor):
    N = 12

    def reset(self) -> None:
        super().reset()
        self._ng_seen: set[tuple] = set()
        self._ng_tail: list[bytes] = []

    def extract(self, events, t0, descriptor, host=None) -> np.ndarray:
        vec = super().extract(events, t0, descriptor, host)
        payloads = [e.data for e in events if e.evt in COUNTED_KINDS and e.data]
        seq = self._ng_tail + payloads
        off, matched, total = len(self._ng_tail), 0, 0
        for i in range(len(payloads)):
            end = off + i
            start = end - self.N + 1
            if start < 0:
                continue
            gram = tuple(seq[start:end + 1])
            total += 1
            if gram in self._ng_seen:
                matched += 1
            elif len(self._ng_seen) < MAX_REMEMBERED:
                self._ng_seen.add(gram)
        self._ng_tail = seq[-(self.N - 1):] if len(seq) >= self.N - 1 else seq
        frac = matched / total if total else 0.0       # no evidence of repetition = 0
        return np.append(vec, np.float32(frac))
