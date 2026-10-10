"""
A fresh, large, balanced evaluation set for the realistic simulator.

    python3 -m eval.analysis_powered            # builds and caches (~6 min)

Why: the seed-7 test split holds 15-24 sessions per attack profile and 178 benign
sessions. At n=20 a detection rate cannot be resolved finer than 5 points (Wilson
interval about +/-20), and 0 false positives in 178 sessions only bounds the rate
below 2.1%. Every comparison in the improvement analysis is therefore re-run on this
set -- 100 sessions per attack profile and 3,000 benign -- which is generated with a
**different seed (2025), never used to train, calibrate or select anything**. It
resolves false positives to ~0.1% and detection to ~+/-10 points per profile.

Same generators, same idle mix, same realism as eval.analysis_realistic.
"""

from __future__ import annotations

import json
import time

import numpy as np

from iudbms.capture.simulator import ATTACK_PROFILES, BENIGN_PROFILES

from .analysis_common import OUT, featurize, load_splits
from .analysis_idle_retrain import IDLE_MIX, QUIET_CLASSES
from .analysis_ngram import NGramExtractor
from .analysis_realistic import RealisticSim
from .corpus import Corpus, SessionData, load_corpus, save_corpus

CACHE = OUT / "analysis_corpus_powered_v2.npz"
META = CACHE.with_suffix(".cond.json")
SEED = 2025


def build(per_profile: int = 100, benign_multiple: int = 6, seed: int = SEED,
          extractor_cls=NGramExtractor):
    sim = RealisticSim(seed=seed)
    rng = np.random.default_rng(seed + 99)
    probs = np.array([m[1] for m in IDLE_MIX])
    plan = sim.plan_balanced(per_profile, benign_profiles=BENIGN_PROFILES,
                             attack_profiles=ATTACK_PROFILES, benign_multiple=benign_multiple)
    sessions, cond = [], {}
    for sid, profile in plan:
        if profile in QUIET_CLASSES:
            name, _, kw = IDLE_MIX[int(rng.choice(len(IDLE_MIX), p=probs))]
        else:
            name, kw = "attack", IDLE_MIX[0][2]
        sim._idle, sim._active, sim._fully_idle = kw["idle"], kw["active"], kw["fully_idle"]
        sessions.append(featurize(sim.session(sid, profile), sid, profile, extractor_cls))
        cond[sid] = name
    return sessions, cond


def load_or_build() -> tuple[list[SessionData], dict]:
    if CACHE.exists() and META.exists():
        return load_corpus(CACHE).sessions, json.loads(META.read_text())
    t0 = time.time()
    sessions, cond = build()
    save_corpus(Corpus(sessions=sessions, n_lines=0, n_events=0), CACHE)
    META.write_text(json.dumps(cond))
    n_b = sum(s.label == 0 for s in sessions)
    print(f"built powered set: {len(sessions)} sessions ({n_b} benign) in {time.time() - t0:.0f} s")
    return sessions, cond


if __name__ == "__main__":
    load_or_build()
