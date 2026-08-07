"""E3: small MLP (7 -> hidden -> 1) behind the EntryModel interface.

python -m src.models.e3 --track a   fits, tunes, freezes
results/frozen/e3_{track}.joblib and records its tau in
results/frozen/taus.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.contracts import read_parquet, seed_everything
from src.experiments import select_tau
from src.models.common import EntryModel, sha256_of_file, verify_load


class E3(EntryModel):
    """Small MLP: Linear(7, hidden) -> ReLU -> Linear(hidden, 1).

    Trained full-batch on CPU with Adam(lr=1e-3) and
    BCEWithLogitsLoss(pos_weight = n_neg / n_pos) — the class-balance
    analog of E1's class_weight="balanced".
    """

    def __init__(self, hidden: int = 16, weight_decay: float = 1e-3):
        super().__init__("e3")
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.net = None             # the trained torch network
        self.X_train_scaled = None  # kept so tune() can refit
        self.y_train = None

    def _build_net(self) -> torch.nn.Module:
        """Fresh Linear(7, self.hidden) -> ReLU -> Linear(self.hidden, 1).
        Outputs a raw logit; sigmoid happens in _predict_scaled."""
        raise NotImplementedError

    def _train_net(self, X_scaled, y, X_val_scaled=None, y_val=None) -> None:
        """The one training loop, used by both fit and tune.

        torch.manual_seed(config.SEED) first, then full-batch Adam for
        up to 500 epochs. With val data: early-stop on validation AUC,
        patience 25, and keep the best-val-AUC weights. Without val
        data: run the full 500 epochs.
        """
        raise NotImplementedError

    def _fit_scaled(self, X_scaled, y) -> None:
        """_train_net without val data. Store X_scaled and y for tune()."""
        raise NotImplementedError

    def _predict_scaled(self, X_scaled):
        """sigmoid(net(X)) as a numpy array, no gradients."""
        raise NotImplementedError

    def tune(self, X_val, y_val) -> dict:
        """Try every hidden size in config.E3_HIDDEN x weight_decay in
        (1e-4, 1e-3, 1e-2), each trained by _train_net with early
        stopping on the val rows; end fitted with the best config.
        fit() must have run first.
        Returns {"hidden": ..., "weight_decay": ..., "val_auc": ...}."""
        raise NotImplementedError

    def get_params_report(self) -> dict:
        """{"hidden": ..., "weight_decay": ...}."""
        raise NotImplementedError


def main() -> None:
    """Fit + tune + freeze e3 for one track.

    Steps:
      1. read data/datasets/triggers_{track}.parquet
      2. fit on train rows, tune on val rows
      3. tau, table = select_tau(model, track)
      4. model.save(results/frozen/e3_{track}.joblib)
      5. freeze smoke test: verify_load(path, some val rows, the live
         model's predictions on them) — the reloaded file must predict
         identically
      6. put tau under "e3_{track}" in results/frozen/taus.json
         (read the dict if the file exists, set the key, write back)
      7. print the joblib's sha256 for DECISIONS.md
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
