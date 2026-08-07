"""E1: L2 logistic regression behind the EntryModel interface.

python -m src.models.e1 --track a   fits, tunes, freezes
results/frozen/e1_{track}.joblib and records its tau in
results/frozen/taus.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.contracts import read_parquet, seed_everything
from src.experiments import select_tau
from src.models.common import EntryModel, sha256_of_file, verify_load


class E1(EntryModel):
    """Logistic regression with L2 penalty; C picked by validation AUC."""

    def __init__(self, C: float = 1.0):
        super().__init__("e1")
        self.C = C
        self.model = None          # the fitted sklearn LogisticRegression
        self.X_train_scaled = None  # kept so tune() can refit
        self.y_train = None

    def _fit_scaled(self, X_scaled, y) -> None:
        """Fit LogisticRegression(penalty="l2", C=self.C,
        class_weight="balanced", solver="lbfgs", max_iter=2000,
        random_state=config.SEED). Store X_scaled and y for tune()."""
        raise NotImplementedError

    def _predict_scaled(self, X_scaled):
        """P(label=1) per row from the fitted model."""
        raise NotImplementedError

    def tune(self, X_val, y_val) -> dict:
        """Refit once per C in (0.01, 0.1, 1, 10) on the stored training
        data; keep the C with the best validation AUC and end fitted
        with it. fit() must have run first.
        Returns {"C": ..., "val_auc": ...}."""
        raise NotImplementedError

    def get_params_report(self) -> dict:
        """{"C": the chosen C}."""
        raise NotImplementedError

    def coefficient_table(self) -> pd.DataFrame:
        """One row per feature: its standardized coefficient (features
        are scaled, so magnitudes are comparable across features)."""
        raise NotImplementedError


def main() -> None:
    """Fit + tune + freeze e1 for one track.

    Steps:
      1. read data/datasets/triggers_{track}.parquet
      2. fit on train rows, tune on val rows
      3. tau, table = select_tau(model, track)
      4. model.save(results/frozen/e1_{track}.joblib)
      5. freeze smoke test: verify_load(path, some val rows, the live
         model's predictions on them) — the reloaded file must predict
         identically
      6. put tau under "e1_{track}" in results/frozen/taus.json
         (read the dict if the file exists, set the key, write back)
      7. write the coefficient table to
         results/tables/e1_coefficients_{track}.csv
      8. print the joblib's sha256 for DECISIONS.md
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
