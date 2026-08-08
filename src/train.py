"""Training runner: fits, tunes, and freezes one model for one track.

The mirror of experiments.py — that file evaluates frozen models, this
one creates them. Model classes are imported here under their real
module names, so frozen files always load from any process (a model
module freezing itself as __main__ pickles an unloadable class path).

Run: python -m src.train --model e3 --track a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.contracts import read_parquet, seed_everything
from src.experiments import select_tau
from src.models.common import sha256_of_file, verify_load
from src.models.e1 import E1
from src.models.e3 import E3

# the normal imports of E1/E3 above are what make frozen files loadable


def train_and_freeze(model_name: str, track: str) -> None:
    """The one freeze protocol, identical for every model.

    1. read data/datasets/triggers_{track}.parquet
    2. build the model (E1 or E3); fit on train rows, tune on val
       rows (print the tune report)
    3. tau, table = select_tau(model, track); print the table
    4. model.save(results/frozen/{model_name}_{track}.joblib)
    5. freeze smoke test: verify_load(path, val rows, the live model's
       predictions) — the reloaded file must predict identically
    6. put tau under "{model_name}_{track}" in results/frozen/taus.json
       (read the dict if the file exists, set the key, write back)
    7. only e1: write model.coefficient_table() to
       results/tables/e1_coefficients_{track}.csv
    8. print the joblib's sha256 for DECISIONS.md
    """
    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")
    train_mask = triggers["split"] == "train"
    train = triggers[train_mask]
    val_mask = triggers["split"] == "val"
    val = triggers[val_mask]

    if model_name == "e1":
        model = E1()
    else:
        model = E3()
    model.fit(train, train["label"].values)
    report = model.tune(val, val["label"].values)
    print(f"{model_name} track {track}: best config {report}")

    tau, tau_table = select_tau(model, track)
    print(tau_table.to_string(index=False))

    path = Path(f"results/frozen/{model_name}_{track}.joblib")
    model.save(path)

    # freeze smoke test: the reloaded file must predict identically
    expected_p = model.predict_proba(val)
    verify_load(path, val, expected_p)

    taus_file = Path("results/frozen/taus.json")
    taus = {}
    if taus_file.exists():
        taus = json.loads(taus_file.read_text())
    taus[f"{model_name}_{track}"] = tau
    taus_file.write_text(json.dumps(taus, indent=2))

    if model_name == "e1":
        coefs_file = Path(f"results/tables/e1_coefficients_{track}.csv")
        coefs_file.parent.mkdir(parents=True, exist_ok=True)
        model.coefficient_table().to_csv(coefs_file, index=False)

    print(f"{model_name} track {track}: tau = {tau}, sha256 = {sha256_of_file(path)}")


def main() -> None:
    """Parse --model and --track, run train_and_freeze."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("e1", "e3"), required=True)
    parser.add_argument("--track", default="a")
    args = parser.parse_args()

    seed_everything()
    train_and_freeze(args.model, args.track)


if __name__ == "__main__":
    main()
