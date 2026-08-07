"""Runner: turns triggers + a model into decisions, trades, and tables.

Run: python -m src.experiments --tracks a,b --models e0,e1,e3 --split trainval
(The noise test lives in src/noise_test.py.)
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

    split "trainval" -> rows with split in (train, val); "val" -> val
    rows only (the clean pre-test model comparison); "test" -> test
    rows. purged/embargo rows are never traded.
    """
    assert split in ("trainval", "val", "test"), f"unknown split '{split}'"

    if split == "trainval":
        wanted = ("train", "val")
    elif split == "val":
        wanted = ("val",)
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

    valid_trigger_rows = split_rows(triggers, split)

    if model_name == "e0":
        decisions = e0_decisions(valid_trigger_rows)
    else:
        model = EntryModel.load(f"results/frozen/{model_name}_{track}.joblib")
        tau = load_tau(model_name, track)
        decisions = model_decisions(model, valid_trigger_rows, tau)

    trades = run_backtest(zscores, prices, valid_trigger_rows, decisions)

    write_parquet(decisions, f"results/decisions_{track}_{model_name}_{split}.parquet", "decisions")
    write_parquet(trades, f"results/trades_{track}_{model_name}_{split}.parquet", "trades")

    entered = int(decisions["enter"].sum())
    print(f"{track} x {model_name} ({split}): {len(valid_trigger_rows)} triggers, {entered} entered, {len(trades)} trades")
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


def select_tau(model: EntryModel, track: str) -> tuple[float, pd.DataFrame]:
    """Apply the pre-registered tau rule for one (model, track) cell.

    On VALIDATION rows only: for each tau in 0.40, 0.45, ... 0.80,
    build decisions (enter = p_hat >= tau), run the engine, record
    n_trades and total net P&L at the headline cost. Then pick:
      1. among taus with >= 25 trades: max net P&L; P&L ties (within
         1e-6) go to the HIGHER tau (trade less when indifferent)
      2. if no tau reaches 25 trades: the tau with the most trades
         (ties to the higher tau)
      3. if that maximum is 0 trades: degenerate cell, tau = 0.5
         (record it in DECISIONS.md)
    Returns (tau, table of tau / n_trades / net_pnl). The caller writes
    tau into results/frozen/taus.json.
    """
    potential_taus = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    net_col = f"net_ret_{config.HEADLINE_COST_BPS}bps"

    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")
    zscores = read_parquet(f"data/spreads/zscores_{track}.parquet")
    prices = read_parquet("data/raw/prices.parquet")

    val_trigger_mask = triggers["split"] == "val"
    val = triggers[val_trigger_mask]

    tau_rows = []
    for tau in potential_taus:
        decisions = model_decisions(model, val, tau)
        trades = run_backtest(zscores, prices, val, decisions)
        net_pnl = trades[net_col].sum()
        tau_rows.append({"tau": tau, "n_trades": len(trades), "net_pnl": net_pnl})
    tau_table = pd.DataFrame(tau_rows)

    # best net P&L among taus with more than 25 trades
    chosen = None
    best_pnl = float("-inf")
    for row in tau_rows:
        if row["n_trades"] < 25:
            continue
        if row["net_pnl"] > best_pnl + 1e-6:
            best_pnl = row["net_pnl"]
            chosen = row["tau"]
        elif row["net_pnl"] > best_pnl - 1e-6:
            # taking highest tau if within 1e-6 of best_pnl
            chosen = row["tau"]

    # if no tau reached 25 trades, tau with most trades is chosen
    if chosen is None:
        best_trades = 0
        for row in tau_rows:
            if row["n_trades"] > 0 and row["n_trades"] >= best_trades:
                best_trades = row["n_trades"]
                chosen = row["tau"]

    # 0 trades
    if chosen is None:
        chosen = 0.5
        print(f"select_tau: ({model.name}, track {track}): 0 trades at every tau; tau = 0.5")

    print(f"select_tau: {model.name} track {track}: tau = {chosen}")
    return chosen, tau_table


def main() -> None:
    """Run run_cell for every (track, model), then write the grid table.

    A model whose frozen file does not exist yet is skipped with a
    printed line, so the grid can run before every model has landed.
    A track with no triggers file yet is skipped the same way. Then
    make_grid_table -> results/tables/grid_{split}.csv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="a,b")
    parser.add_argument("--models", default="e0,e1,e3")
    parser.add_argument("--split", default="trainval")

    args = parser.parse_args()
    seed_everything()

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
