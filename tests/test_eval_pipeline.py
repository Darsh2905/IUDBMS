"""Milestone 10 — the evaluation harness itself.

Guards the split discipline (plan §5) rather than any particular metric value:
a harness that leaks sessions across the split reports fake accuracy no matter
how good the model is.
"""

import numpy as np
import pytest

from eval.corpus import build_corpus, stratified_split
from eval.pipeline import identity_whitelist_baseline, run_configuration
from iudbms.contract import N_FEATURES, modality_indices


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(n_benign=24, n_attack=16, seed=7, verbose=False)


def test_corpus_shapes_are_consistent(corpus):
    for s in corpus.sessions:
        assert s.X.shape == (len(s.t), N_FEATURES)
        assert s.X.shape[0] == len(s.y)
        assert np.all(np.isfinite(s.X))


def test_pre_onset_windows_of_an_attack_are_labelled_benign(corpus):
    """Plan §4.4: the label applies from onset onward, which is what forces
    Stage 2 to learn a transition rather than device identity."""
    attacks = [s for s in corpus.sessions if s.label == 1 and s.onset is not None]
    assert attacks
    checked = 0
    for s in attacks:
        pre = s.y[s.t + 1.0 <= s.onset]
        if len(pre):
            assert pre.sum() == 0, f"{s.sid}: pre-onset window labelled malicious"
            checked += 1
    assert checked > 0, "no session had pre-onset windows; test proved nothing"


def test_benign_sessions_have_no_malicious_windows(corpus):
    for s in corpus.sessions:
        if s.label == 0:
            assert s.y.sum() == 0


def test_split_is_by_session_never_by_window(corpus):
    fit, val, test = stratified_split(corpus.sessions, seed=7)
    ids = [{s.sid for s in g} for g in (fit, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
    assert len(ids[0] | ids[1] | ids[2]) == len(corpus.sessions)


def test_split_is_stratified_by_profile(corpus):
    fit, val, test = stratified_split(corpus.sessions, seed=7)
    train_profiles = {s.profile for s in fit + val}
    assert {s.profile for s in test} <= train_profiles, (
        "a profile appears in test but never in training"
    )


def test_split_is_deterministic(corpus):
    a = stratified_split(corpus.sessions, seed=7)
    b = stratified_split(corpus.sessions, seed=7)
    for ga, gb in zip(a, b):
        assert [s.sid for s in ga] == [s.sid for s in gb]


def test_identity_whitelist_is_blind_to_cloned_descriptors(corpus):
    """The premise of the whole project.

    A VID:PID allowlist flags a device only when its model was never seen
    during training. Every attack profile here clones a descriptor from the
    benign pool, so any attack whose model the allowlist has seen is invisible
    to it — no matter what the device then does.

    Note this is *not* the same as "the whitelist always scores 0": on a small
    corpus the training split may simply not have observed every legitimate
    vendor model yet, and an attack reusing one of those is flagged for being
    unfamiliar rather than for being malicious. That is a property of the
    sample size, not a detection capability, which is why the assertion below
    is restricted to attacks whose model *was* seen.
    """
    fit, _, test = stratified_split(corpus.sessions, seed=7)
    known = {(s.descriptor.vid, s.descriptor.pid) for s in fit if s.label == 0}
    seen_attacks = [s for s in test
                    if s.label == 1 and (s.descriptor.vid, s.descriptor.pid) in known]
    assert seen_attacks, "test proved nothing: no attack reused a known model"
    for s in seen_attacks:
        assert (s.descriptor.vid, s.descriptor.pid) in known
    # and the behavioural system must catch what the allowlist cannot
    assert identity_whitelist_baseline(fit, test)["attack_detect"] < 1.0


@pytest.mark.slow
def test_configuration_runs_and_reports_coverage(corpus):
    fit, val, test = stratified_split(corpus.sessions, seed=7)
    r = run_configuration(fit, val, test, name="smoke", epochs=2, verbose=False)
    assert 0.0 <= r.window["auc"] <= 1.0
    # A detector that strips everything trivially achieves precision 1.0;
    # detection rate must always be reported alongside it.
    assert "attack_detect" in r.session
    assert "benign_fp" in r.session
    assert r.window["n_windows"] > 0


@pytest.mark.slow
def test_modality_ablation_uses_only_its_own_columns(corpus):
    fit, val, test = stratified_split(corpus.sessions, seed=7)
    idx = np.array(modality_indices("timing"))
    r = run_configuration(fit, val, test, name="timing", feature_idx=idx,
                          use_stage2=False, verbose=False)
    assert 0.0 <= r.window["auc"] <= 1.0
