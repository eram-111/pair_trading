"""E3: small MLP (7 -> hidden -> 1) behind the EntryModel interface.

Trained and frozen by the runner: python -m src.train --model e3 --track a
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src import config
from src.models.common import EntryModel


class E3(EntryModel):
    """Small MLP: Linear(7, hidden) -> ReLU -> Linear(hidden, 1).
    """

    def __init__(self, hidden: int = 16, weight_decay: float = 0.001):
        super().__init__("e3")
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.net = None             # the trained torch network
        self.X_train_scaled = None  # kept so tune() can refit
        self.y_train = None

    def _build_net(self) -> torch.nn.Module:
        """Fresh Linear(7, self.hidden) -> ReLU -> Linear(self.hidden, 1).
        Outputs a raw logit; sigmoid happens in _predict_scaled."""
        n_features = len(config.FEATURES)
        return torch.nn.Sequential(torch.nn.Linear(n_features, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1))

    def _train_net(self, X_scaled, y, X_val_scaled=None, y_val=None):
        """The one training loop, used by both fit and tune.

        torch.manual_seed(config.SEED) first, so every config starts
        from the same weights. Full-batch Adam for up to 500 iterations.
        With val data: early-stop on validation AUC, patience 25, keep
        the best-val-AUC weights, return that AUC. Without val data:
        run the full 500 iterations, return None.
        """
        torch.manual_seed(config.SEED)

        X = torch.as_tensor(np.asarray(X_scaled), dtype=torch.float32)
        y_true = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

        total_true = float(y_true.sum())
        total_false = float(len(y_true)) - total_true

        assert total_true > 0 and total_false > 0, "training labels must contain both classes"

        weight_true = torch.tensor(total_false / total_true)  # the class_weight="balanced" analog

        self.net = self._build_net()

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weight_true)

        parameters = self.net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=self.weight_decay)

        best_auc = None
        best_state = None
        iteration_since_best = 0

        for i in range(500):
            optimizer.zero_grad()
            loss = loss_func(self.net(X), y_true)
            loss.backward()
            optimizer.step()

            if X_val_scaled is None:
                continue

            p_val = self._predict_scaled(X_val_scaled)
            val_auc = roc_auc_score(y_val, p_val)

            if best_auc is None or val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(self.net.state_dict())
                iteration_since_best = 0
            else:
                iteration_since_best = iteration_since_best + 1
                if iteration_since_best >= 25:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            
        return best_auc

    def _fit_scaled(self, X_scaled, y) -> None:
        """_train_net without val data. Store X_scaled and y for tune()."""
        self.X_train_scaled = X_scaled
        self.y_train = y
        self._train_net(X_scaled, y)

    def _predict_scaled(self, X_scaled):
        """sigmoid(net(X)) as a numpy array, no gradients."""
        X = torch.as_tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.net(X)
            probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(-1)
        probabilities = probabilities.numpy()
        probabilities = probabilities.astype("float64")

        return probabilities

    def tune(self, X_val, y_val) -> dict:
        """Try every hidden size in config.E3_HIDDEN x weight_decay in
        (0.0001, 0.001, 0.01), each trained by _train_net with early
        stopping on the val rows; end fitted with the best config.
        fit() must have run first.
        Returns {"hidden": ..., "weight_decay": ..., "val_auc": ...}."""
        assert self.X_train_scaled is not None, "fit() must run before tune()"

        features = X_val[list(config.FEATURES)]
        X_val_scaled = self.scaler.transform(features)

        weight_decays = (0.0001, 0.001, 0.01)
        best = None
        for hidden in config.E3_HIDDEN:
            for weight_decay in weight_decays:
                self.hidden = hidden
                self.weight_decay = weight_decay
                val_auc = self._train_net(self.X_train_scaled, self.y_train, X_val_scaled, y_val)
                print(f"e3 tune: hidden={hidden} weight_decay={weight_decay} val_auc={val_auc:.4f}")
                if best is None or val_auc > best["val_auc"]:
                    best = {"hidden": hidden, "weight_decay": weight_decay, "val_auc": val_auc, "net": self.net}

        # end fitted with the winning config
        self.hidden = best["hidden"]
        self.weight_decay = best["weight_decay"]
        self.net = best["net"]
        return {"hidden": best["hidden"], "weight_decay": best["weight_decay"], "val_auc": best["val_auc"]}

    def get_params_report(self) -> dict:
        """{"hidden": ..., "weight_decay": ...}."""
        return {"hidden": self.hidden, "weight_decay": self.weight_decay}
