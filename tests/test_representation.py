"""Golden and invariance tests for src/representation.py.

Run: pytest -q tests/test_representation.py
"""
import numpy as np
import pandas as pd
import pytest

from src.representation import pair_from_labels, pca_one_window, run_rolling_pca


def test_pca_one_window_finds_a_planted_factor():
    # 40 fake stocks driven by ONE common factor + small noise: the first
    # component must dominate, so the smallest candidate m (3) is enough
    rng = np.random.default_rng(311)
    factor = rng.normal(0.0, 0.02, size=300)
    betas = rng.uniform(0.5, 1.5, size=40)
    noise = rng.normal(0.0, 0.002, size=(300, 40))
    returns = pd.DataFrame(np.outer(factor, betas) + noise)

    weights, n_components, cum_var, corr = pca_one_window(returns)

    assert weights.shape == (40, 5)
    assert n_components == 3
    assert cum_var >= 0.60
    # the sign rule: every kept column's weights sum positive
    for column in range(5):
        assert weights[:, column].sum() > 0


def test_pair_from_labels_ignores_label_numbering():
    # relabeling clusters (0,1 -> 7,2) must produce the same pair set
    tickers = ["A", "B", "C", "D"]
    pairs_one = pair_from_labels(np.array([0, 0, 1, 1]), tickers)
    pairs_two = pair_from_labels(np.array([7, 7, 2, 2]), tickers)
    assert pairs_one == pairs_two == {("A", "B"), ("C", "D")}


def test_rolling_pca_mini_golden():
    # golden on the first 400 real trading days, computed once and frozen:
    # any change to windowing, sign rule, or m selection moves these
    returns = pd.read_parquet("data/processed/returns.parquet")
    factors, meta, residuals, loadings = run_rolling_pca(returns.iloc[:400])

    assert meta["n_components"].value_counts().to_dict() == {4: 123, 5: 25}
    assert len(residuals) == 148
    assert len(loadings) == 1600
    assert abs(float(residuals.iloc[0, 0]) - (-0.02048836)) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
