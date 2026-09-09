"""Tuning discipline and the operating-point curve.

The load-bearing property here is that a hyperparameter sweep cannot see the
test split. Sweeping on test and then reporting test is the easiest way to
manufacture a result, so it is guarded by tests rather than by care.
"""

import numpy as np
import pytest

from eval.corpus import build_corpus, load_corpus, save_corpus, stratified_split
from eval.operating_curve import format_curves, sweep_gates, sweep_thresholds
from eval.pipeline import run_configuration
from eval.tune import inner_splits


@pytest.fixture(scope="module")
def splits():
    corpus = build_corpus(n_benign=30, n_attack=20, seed=7, verbose=False)
    return corpus, *stratified_split(corpus.sessions, seed=7)


def test_inner_splits_are_mutually_disjoint(splits):
    _, fit, val, test = splits
    i_fit, i_cal, i_sel = inner_splits(fit, val, seed=7)
    ids = [{s.sid for s in g} for g in (i_fit, i_cal, i_sel)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_inner_splits_never_touch_the_test_set(splits):
    """The whole point of tuning discipline."""
    _, fit, val, test = splits
    test_ids = {s.sid for s in test}
    for group in inner_splits(fit, val, seed=7):
        assert {s.sid for s in group} & test_ids == set(), (
            "a tuning split contains held-out test sessions"
        )


def test_inner_splits_cover_the_non_test_pool(splits):
    _, fit, val, test = splits
    pool = {s.sid for s in fit + val}
    covered = set().union(*({s.sid for s in g} for g in inner_splits(fit, val, seed=7)))
    assert covered == pool


def test_inner_splits_are_deterministic(splits):
    _, fit, val, _ = splits
    a = [[s.sid for s in g] for g in inner_splits(fit, val, seed=7)]
    b = [[s.sid for s in g] for g in inner_splits(fit, val, seed=7)]
    assert a == b


def test_calibrating_and_scoring_on_the_same_split_is_refused(splits):
    """Otherwise the false-positive rate hits the target by construction and
    every configuration in a sweep looks equally good."""
    _, fit, val, test = splits
    with pytest.raises(ValueError, match="same sessions"):
        run_configuration(fit, val, test, name="bad", use_stage2=False,
                          score_on="calib", calibrate_on="calib", verbose=False)


# --------------------------------------------------------------------------
# corpus cache
# --------------------------------------------------------------------------

def test_corpus_cache_round_trips_exactly(tmp_path, splits):
    corpus = splits[0]
    p = tmp_path / "c.npz"
    save_corpus(corpus, p)
    back = load_corpus(p)
    assert len(back.sessions) == len(corpus.sessions)
    assert back.n_lines == corpus.n_lines
    for a, b in zip(corpus.sessions, back.sessions):
        assert a.sid == b.sid and a.profile == b.profile and a.label == b.label
        assert a.onset == b.onset and a.duration == b.duration
        assert a.descriptor == b.descriptor
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)


def test_cache_refuses_a_mismatched_contract_major(tmp_path, splits, monkeypatch):
    import json

    p = tmp_path / "c.npz"
    save_corpus(splits[0], p)
    blob = dict(np.load(p, allow_pickle=False))
    meta = json.loads(str(blob["meta"]))
    meta["contract_version"] = "99.0.0"
    blob["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(p, **blob)
    with pytest.raises(RuntimeError, match="rebuild"):
        load_corpus(p)


# --------------------------------------------------------------------------
# operating curve
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(splits):
    _, fit, val, test = splits
    r = run_configuration(fit, val, test, name="curve", epochs=2,
                          keep_scores=True, verbose=False)
    return fit, val, test, r


def test_threshold_sweep_is_monotone_in_the_right_direction(scored):
    """Loosening the target FPR must not reduce detection or reduce false
    positives -- if it does, the sweep is wired backwards."""
    _, val, test, r = scored
    rows = sweep_thresholds(test, val, r, [0.0005, 0.005, 0.05])
    assert rows[0]["threshold"] >= rows[-1]["threshold"]
    assert rows[0]["detect"] <= rows[-1]["detect"]
    assert rows[0]["benign_fp"] <= rows[-1]["benign_fp"]


def test_gate_sweep_is_monotone_in_the_right_direction(scored):
    """A stricter persistence gate suppresses more of everything."""
    _, _, test, r = scored
    rows = sweep_gates(test, r, r.threshold, [(2, 6), (4, 6), (6, 6)])
    assert rows[0]["detect"] >= rows[-1]["detect"]
    assert rows[0]["benign_fp"] >= rows[-1]["benign_fp"]


def test_every_curve_row_reports_both_axes(scored):
    """A detector that alerts on nothing has a perfect false-positive rate.
    Detection must always be reported alongside it."""
    _, val, test, r = scored
    for row in sweep_thresholds(test, val, r, [0.001, 0.01]):
        assert "detect" in row and "benign_fp" in row


def test_format_marks_the_deployed_point(scored):
    _, val, test, r = scored
    rows = sweep_thresholds(test, val, r, [0.001])
    out = format_curves(rows, sweep_gates(test, r, r.threshold, [(4, 6)]), rows[0]["threshold"])
    assert "<- deployed" in out
    assert "4-of-6" in out
