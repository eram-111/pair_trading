"""Tests for src/models/common.py: the EntryModel base contract, E0, and
the save/load helpers. Uses a tiny Dummy subclass — the base class is
what's under test, not any real model.

Run: pytest -q tests/test_models.py
or:  python -m tests.test_models   (from the repo root)
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.common import EntryModel, e0_decisions, sha256_of_file, verify_load
from tests.test_engine import fake_triggers


class Dummy(EntryModel):
    """Smallest possible subclass: predicts the training base rate for every row."""

    def __init__(self):
        super().__init__("dummy")

    def _fit_scaled(self, X_scaled, y):
        self.rate = float(np.mean(y))

    def _predict_scaled(self, X_scaled):
        return np.full(len(X_scaled), self.rate)


def make_X(n_rows):
    """A feature frame with the 7 config.FEATURES columns, seeded noise."""
    rng = np.random.default_rng(311)
    return pd.DataFrame(rng.normal(size=(n_rows, 7)), columns=list(config.FEATURES))


def test_fit_then_predict():
    X = make_X(20)
    y = np.array([0, 1] * 10)  # base rate 0.5
    model = Dummy().fit(X, y)
    p = model.predict_proba(X)
    assert len(p) == 20
    assert (p == 0.5).all()


def test_predict_before_fit_fails():
    model = Dummy()
    try:
        model.predict_proba(make_X(5))
        failed = False
    except AssertionError:
        failed = True
    assert failed, "predict_proba before fit() must raise"


def test_scaler_is_fit_on_train_only():
    X_train = make_X(20)
    model = Dummy().fit(X_train, np.array([0, 1] * 10))
    means_after_fit = model.scaler.mean_.copy()

    # predicting on very different data must NOT change the stored scaler
    X_other = make_X(20) + 100.0
    model.predict_proba(X_other)
    assert (model.scaler.mean_ == means_after_fit).all(), "predict_proba must never refit the scaler"


def test_save_load_round_trip(tmp_path):
    X = make_X(20)
    model = Dummy().fit(X, np.array([0, 1] * 10))
    p_before = model.predict_proba(X)

    path = tmp_path / "dummy.joblib"
    model.save(path)

    loaded = Dummy.load(path)
    p_after = loaded.predict_proba(X)
    assert (p_before == p_after).all(), "loaded model must predict identically"
    assert loaded.name == "dummy"

    # the helpers built on top of save/load
    assert len(sha256_of_file(path)) == 64  # a sha256 hex string
    verify_load(path, X, p_before)          # raises if the round-trip broke


def test_e2_matches_sklearn_lda():
    # shared-covariance GDA is mathematically LDA; our numpy must agree
    # with sklearn's implementation to 1e-6
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from src.models.e2 import E2

    X = make_X(200)
    y = np.array(([1] * 3 + [0] * 7) * 20)  # base rate 0.3, both classes present
    model = E2().fit(X, y)
    ours = model.predict_proba(X)

    X_scaled = model.scaler.transform(X[list(config.FEATURES)])
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    theirs = lda.predict_proba(X_scaled)[:, 1]

    assert np.abs(ours - theirs).max() < 1e-6, "numpy GDA must match sklearn LDA"


def test_e0_decisions():
    triggers = fake_triggers()
    decisions = e0_decisions(triggers)
    assert len(decisions) == len(triggers)
    assert decisions["enter"].all()
    assert decisions["p_hat"].isna().all()
    assert list(decisions["trigger_id"]) == list(triggers["trigger_id"])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
