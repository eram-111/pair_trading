"""Runner: turns triggers + a model into decisions, trades, and tables.

Subcommands:
  grid   python -m src.experiments grid --tracks a,b --models e0,e1,e3 --split trainval
  noise  python -m src.experiments noise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.contracts import read_parquet, seed_everything, validate_artifact, write_parquet
from src.engine import run_backtest
from src.models.common import EntryModel, e0_decisions


def load_tau(model_name: str, track: str) -> float:
    """Read this cell's tau from results/frozen/taus.json, key "{model}_{track}"."""
    taus_file = Path("results/frozen/taus.json")
    assert taus_file.exists(), "taus.json missing, fit and freeze the model first (make models)"

    taus = json.loads(taus_file.read_text())
    key = f"{model_name}_{track}"
    assert key in taus, f"no tau recorded for '{key}'"
    return float(taus[key])


def split_rows(triggers: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one evaluation split.

    split "trainval" -> rows with split in (train, val); "test" -> test
    rows. purged/embargo rows are never traded.
    """
    assert split in ("trainval", "test"), f"unknown split '{split}'"

    if split == "trainval":
        wanted = ("train", "val")
    else:
        wanted = ("test",)

    split_mask = triggers["split"].isin(wanted)
    return triggers[split_mask]


def model_decisions(model: EntryModel, triggers: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Decisions table from a fitted model: p_hat = predict_proba(triggers),
    enter = (p_hat >= tau)."""
    p_hat = model.predict_proba(triggers)
    enter = p_hat >= tau
    decisions = pd.DataFrame({ "trigger_id": triggers["trigger_id"],"enter": enter,"p_hat": p_hat,})
    validate_artifact(decisions, "decisions")
    return decisions


def run_cell(track: str, model_name: str, split: str) -> pd.DataFrame:
    """One grid cell: triggers -> decisions -> engine -> trades ledger.

    Reads data/datasets/triggers_{track}.parquet, data/spreads/
    zscores_{track}.parquet, data/raw/prices.parquet. "e0" needs no
    model file; other models load results/frozen/{model}_{track}.joblib
    and their tau. Writes results/decisions_{track}_{model}_{split}.parquet
    and results/trades_{track}_{model}_{split}.parquet (split in the name
    so a test run never overwrites trainval files). Returns the ledger.
    """
    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")
    zscores = read_parquet(f"data/spreads/zscores_{track}.parquet")
    prices = read_parquet("data/raw/prices.parquet")

    rows = split_rows(triggers, split)

    if model_name == "e0":
        decisions = e0_decisions(rows)
    else:
        model = EntryModel.load(f"results/frozen/{model_name}_{track}.joblib")
        tau = load_tau(model_name, track)
        decisions = model_decisions(model, rows, tau)

    trades = run_backtest(zscores, prices, rows, decisions)

    write_parquet(decisions, f"results/decisions_{track}_{model_name}_{split}.parquet", "decisions")
    write_parquet(trades, f"results/trades_{track}_{model_name}_{split}.parquet", "trades")

    entered = int(decisions["enter"].sum())
    print(f"{track} x {model_name} ({split}): {len(rows)} triggers, {entered} entered, {len(trades)} trades")
    return trades


def make_grid_table(tracks: list, models: list, split: str) -> pd.DataFrame:
    """One row per (track, model) read from its written ledger: n_trades,
    mean gross, mean net at the headline cost. Cells with no ledger file
    (skipped in the run) get no row. Deeper stats are metrics.py's job,
    not the runner's.
    """
    net_col = f"net_ret_{config.HEADLINE_COST_BPS}bps"
    rows = []
    for track in tracks:
        for model_name in models:
            ledger_file = Path(f"results/trades_{track}_{model_name}_{split}.parquet")
            if not ledger_file.exists():
                continue
            trades = read_parquet(ledger_file)
            mean_gross = trades["gross_ret"].mean()
            mean_net_headline = trades[net_col].mean()
            rows.append({"track": track, "model": model_name, "split": split, "n_trades": len(trades), "mean_gross": mean_gross, "mean_net_headline": mean_net_headline})
    return pd.DataFrame(rows)


def run_noise_test() -> None:
    """Full pipeline on synthetic random-walk prices; everything under
    results/noise/, ending in PASS_FAIL.md.

    Criteria, one printed line each:
      1. every model's mean net return at 10 bps is <= 0, or its CI covers 0
      2. no model's AUC beats an abs-z-only logistic baseline (CI of the
         AUC difference covers 0); raw AUC vs 0.5 printed as advisory only
      3. > 50 triggers fired and the label base rate is strictly inside (0, 1)
    Any FAIL means a leakage bug: stop and find it before trusting results.
    """
    raise NotImplementedError


def main() -> None:
    """Parse the subcommand and run it.

    grid: run_cell for every (track, model); a model whose frozen file
    does not exist yet is skipped with a printed line, so the grid can
    run before every model has landed. A track with no triggers file
    yet is skipped the same way. Then make_grid_table ->
    results/tables/grid_{split}.csv.
    noise: run_noise_test.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="arg1", required=True)

    grid_parser = subparsers.add_parser("grid")
    grid_parser.add_argument("--tracks", default="a,b")
    grid_parser.add_argument("--models", default="e0,e1,e3")
    grid_parser.add_argument("--split", default="trainval")

    subparsers.add_parser("noise")

    args = parser.parse_args()
    seed_everything()

    if args.arg1 == "noise":
        run_noise_test()
        return

    tracks = args.tracks.split(",")
    models = args.models.split(",")

    for track in tracks:
        triggers_file = Path(f"data/datasets/triggers_{track}.parquet")
        if not triggers_file.exists():
            print(f"track {track}: no triggers file, skipped")
            continue
        for model_name in models:
            if model_name != "e0":
                model_file = Path(f"results/frozen/{model_name}_{track}.joblib")
                if not model_file.exists():
                    print(f"{track} x {model_name}: no frozen model file, skipped")
                    continue
            run_cell(track, model_name, args.split)

    table = make_grid_table(tracks, models, args.split)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(f"results/tables/grid_{args.split}.csv", index=False)
    print(f"grid table -> results/tables/grid_{args.split}.csv")


if __name__ == "__main__":
    main()
