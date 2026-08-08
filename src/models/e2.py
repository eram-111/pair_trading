"""E2: GDA with shared covariance. Run: python -m src.train --model e2 --track a"""
from __future__ import annotations

import numpy as np

from src.models.common import EntryModel

class E2(EntryModel):
    """GDA with one Gaussian per class and shared covariance."""

    def __init__(self):
        super().__init__("e2")
        self.prior_1 = None
        self.mean_0 = None
        self.mean_1 = None
        self.inf_cov = None

    def _fit_scaled(self, X_scaled, y) -> None:
        """Estimate prior, class means, and pooled covariance."""
        X = np.asarray(X_scaled, dtype="float64")
        y = np.asarray(y)

        rows_1 = X[y == 1]
        rows_0 = X[y == 0]
        assert len(rows_1) > 0 and len(rows_0) > 0, "training labels must contain both classes"

        self.prior_1 = len(rows_1) / len(X)
        self.mean_1 = rows_1.mean(axis=0)
        self.mean_0 = rows_0.mean(axis=0)

        
        diff_0 = rows_0 - self.mean_0
        diff_1 = rows_1 - self.mean_1
        diffs = np.vstack([diff_0, diff_1])

        covariance = np.dot(diffs.T, diffs)
        covariance  = covariance / (len(X) - 2)
        
        self.inf_cov = np.linalg.inv(covariance )

    def _predict_scaled(self, X_scaled):
        """Return P(label=1 | x) by Bayes' rule."""
        X = np.asarray(X_scaled, dtype="float64")

        diff = self.mean_1 - self.mean_0
        w = np.dot(self.inf_cov, diff)

        mean_sum = self.mean_1 + self.mean_0
        prior = self.prior_1 / (1 - self.prior_1)

        b = -0.5 * np.dot(mean_sum, w) + np.log(prior)

        log_odds = np.dot(X, w) + b

        return 1 / (1 + np.exp(-log_odds))

    def tune(self, X_val, y_val) -> dict:
        return {}

    def get_params_report(self) -> dict:
        return {"prior_1": float(self.prior_1)}
