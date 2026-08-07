"""EntryModel: abstract base class all entry models (E1, E2, E3) subclass,
so the runner uses every model through one interface. Also E0 (a plain
function — nothing to learn) and the model freeze helpers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.contracts import validate_artifact


class EntryModel:
    """Abstract base class for entry models.

    The base guarantees, for every subclass:
    - features = the 7 config.FEATURES columns, in that order
    - scaler is fit on TRAIN rows only (inside fit) and reused everywhere
      after — refitting it later would leak
    - save/load keep model + scaler in one file

    Concrete (do not override): fit, predict_proba, save, load.
    Abstract (must override): _fit_scaled, _predict_scaled, tune,
    get_params_report.
    """

    name: str      # subclasses set "e1" / "e2" / "e3"
    scaler: StandardScaler | None

    def __init__(self, name:str):
        self.name = name
        self.scaler = None

    # ---- concrete: do not override ----

    def fit(self, X_train: pd.DataFrame, y_train) -> "EntryModel":
        """Fit scaler on train rows, then _fit_scaled. Returns self."""
        features = X_train[list(config.FEATURES)]
        self.scaler = StandardScaler()
        X_standardized = self.scaler.fit_transform(features)
        self._fit_scaled(X_standardized, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame):
        """Return the predicted probability of (label=1) for each row using the fitted training scaler."""
        assert self.scaler is not None, "Model must be fitted with fit() before predict_proba()"
        features = X[list(config.FEATURES)]
        X_standardized = self.scaler.transform(features)
        return self._predict_scaled(X_standardized)

    def save(self, path: Path | str) -> None:
        """Whole object (model + scaler) to one joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "EntryModel":
        """Read a save()d model back; predicts identically."""
        model = joblib.load(path)
        assert isinstance(model, EntryModel), f"{path} is not a saved EntryModel"
        return model

    # ---- abstract: must override ----

    def _fit_scaled(self, X_scaled, y) -> None:
        """Learn from already-scaled features."""
        raise NotImplementedError("subclass must override _fit_scaled")

    def _predict_scaled(self, X_scaled):
        """Probabilities from already-scaled features."""
        raise NotImplementedError("subclass must override _predict_scaled")

    def tune(self, X_val, y_val) -> dict:
        """Pick hyperparameters by validation AUC; refit; return report."""
        raise NotImplementedError("subclass must override tune")

    def get_params_report(self) -> dict:
        """Chosen hyperparameters as a dict."""
        raise NotImplementedError("subclass must override get_params_report")


def e0_decisions(triggers: pd.DataFrame) -> pd.DataFrame:
    """Enter-always baseline: one row per trigger, enter=True, p_hat=NaN."""
    decisions = pd.DataFrame({"trigger_id": triggers["trigger_id"],"enter": True, "p_hat": float("nan")})
    validate_artifact(decisions, "decisions")
    return decisions


def sha256_of_file(path: Path | str) -> str:
    """Fingerprint of a saved model file (recorded at the freeze)."""
    file_bytes = Path(path).read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()


def verify_load(path: Path | str, X_sample: pd.DataFrame, expected_p) -> None:
    """Cold-load the file; assert predictions match expected_p exactly."""
    model = EntryModel.load(path)
    p = model.predict_proba(X_sample)
    matches = (p == expected_p)
    assert matches.all(), f"frozen model at {path} predicts differently after loading"
