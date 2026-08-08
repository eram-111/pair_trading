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
from sklearn.metrics import roc_auc_score

from src import config
from src.contracts import read_parquet, seed_everything
from src.experiments import select_tau
from src.models.common import EntryModel, sha256_of_file, verify_load


class E1(EntryModel):
    """Logistic regression with L2 penalty; C picked by validation AUC"""

    def __init__(self, C: float = 1.0):
        super().__init__("e1")
        self.C = C
        self.model = None          # the fitted sklearn LogisticRegression
        self.X_train_scaled = None  # kept so tune() can refit
        self.y_train = None

    def _fit_scaled(self, X_scaled, y) -> None:
        """Fit logistic regression using the standardized training data"""
        self.X_train_scaled = np.asarray(X_scaled).copy()
        self.y_train = np.asarray(y).copy()
        self.model = LogisticRegression(penalty="l2",C=self.C,class_weight="balanced",
                                        solver="lbfgs",max_iter=2000,random_state=config.SEED)
        self.model.fit(self.X_train_scaled, self.y_train)

    def _predict_scaled(self, X_scaled):
        """Return the probability of label 1 for each row"""
        prob = self.model.predict_proba(X_scaled)
        return prob[:, 1]

    def tune(self, X_val, y_val) -> dict:
        """Pick the C with the best validation AUC and refit with it"""
        if self.X_train_scaled is None:
            raise ValueError("need to call fit() before tune()")
        
        best_C = None
        best_auc = -1.0

        for C in (0.01, 0.1, 1, 10):
            self.C = C
            self._fit_scaled(self.X_train_scaled, self.y_train)
            auc = roc_auc_score(y_val, self.predict_proba(X_val))
            print(f"e1 tune: C={C} val AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_C = C

        self.C = best_C
        self._fit_scaled(self.X_train_scaled, self.y_train)
        return {"C": best_C, "val_auc": best_auc}

    def get_params_report(self) -> dict:
        """{"C": the chosen C}."""
        return {"C": self.C}

    def coefficient_table(self) -> pd.DataFrame:
        """Return one coefficient for each feature"""
        if self.model is None:
            raise ValueError("need to call fit() before coefficient_table()")
        
        features = list(config.FEATURES)
        coefs = self.model.coef_[0]
        result = pd.DataFrame({"feature": features,"coefficient": coefs})
        return result
        

def main() -> None:
    """Fit and tune and freeze e1 for one track"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], required=True)
    args = parser.parse_args()
    track = args.track

    seed_everything()

    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")
    train = triggers[triggers["split"] == "train"]
    val = triggers[triggers["split"] == "val"]

    model = E1()
    model.fit(train, train["label"])
    tuning_result = model.tune(val, val["label"])
    print(f"e1 track {track}: {tuning_result}")

    tau, tau_table = select_tau(model, track)
    print(tau_table.to_string(index=False))

    model_path = Path(f"results/frozen/e1_{track}.joblib")
    model.save(model_path)
    verify_load(model_path, val, model.predict_proba(val))

    taus_path = Path("results/frozen/taus.json")
    taus = {}
    if taus_path.exists():
        taus = json.loads(taus_path.read_text())
    taus[f"e1_{track}"] = tau
    taus_path.write_text(json.dumps(taus, indent=2) + "\n")

    coefs_path = Path(f"results/tables/e1_coefficients_{track}.csv")
    coefs_path.parent.mkdir(parents=True, exist_ok=True)
    model.coefficient_table().to_csv(coefs_path, index=False)

    print(f"model saved to {model_path}")
    print(f"model sha256: {sha256_of_file(model_path)}")


if __name__ == "__main__":
    main()
