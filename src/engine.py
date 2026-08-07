"""Backtest engine: turns entry decisions into a ledger of completed trades.

For each trigger the decisions table accepts, the engine replays the
trade day by day under the same fixed rules:
- Enter at the close of the trading day after the trigger day (t+1).
- Direction from the sign of z at trigger: z > 0 -> short stock_a,
  long stock_b; z < 0 -> the reverse. $1 per leg.
- Exit at the first close with |z| < 0.5 ("reverted"), else at the
  close 5 trading days after entry ("timeout").
- gross_ret = sum of daily (long return - short return) over the
  holding days, from simple returns.
- Costs: c basis points per leg per transaction = 4c per round trip,
  precomputed as one net_ret_{c}bps column per value in the cost grid.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.contracts import validate_artifact

def _simple_ret(prices: pd.DataFrame)-> pd.DataFrame:
    """
    Table of simple returns
    """
    prices_shifted_forward = prices.shift(1)
    return (prices - prices_shifted_forward ) / prices_shifted_forward

def _split_stocks(pair_id: str, z_trigger: float) -> tuple[str, str]:
    """
    Return long_stock, short_stock from the pair id and the trigger's z sign.
    z > 0: -> short stock_a, long stock_b.
    z < 0: -> long stock_a, short stock_b.
    """
    stock_a, stock_b = pair_id.split("__")
    if z_trigger > 0:
        return stock_b, stock_a
    return stock_a, stock_b

def run_backtest(zscores: pd.DataFrame, prices: pd.DataFrame, triggers: pd.DataFrame, decisions: pd.DataFrame,  cost_grid_bps: tuple = config.COST_GRID_BPS) -> pd.DataFrame:
    """Replay every accepted trigger; return the trades ledger.

    Steps:
      1. Precompute simple returns once: prices / prices.shift(1) - 1.
      2. For each trigger whose decision says enter=True:
         - entry position = position of trigger_date in the index + 1
           (if that is past the last day, drop the trigger and count it);
         - direction from the sign of z_trigger;
         - walk forward one day at a time starting the day AFTER entry:
           add (long return - short return) to gross, exit when |z| < 0.5
           or when 5 trading days have passed since entry, whichever
           comes first (check reversion before timeout);
         - append one ledger row: trigger_id, pair_id, entry_date,
           exit_date, exit_reason, days_held, gross_ret.
      3. Add net_ret_{c}bps = gross_ret - 4 * c * 0.0001 per grid value.
      4. Validate against the "trades" schema and return.
    """
    returns = _simple_ret(prices)

    decision_dict = {}
    for index, row in decisions.iterrows():
        decision_dict[row["trigger_id"]] = row["enter"]

    last_day = len(zscores.index) - 1
    ledger_rows = []
    dropped = 0

    for index, trig in triggers.iterrows():
        trigger_id = trig["trigger_id"]
        # dont enter trade
        if trigger_id not in decision_dict:
            continue
        if not decision_dict[trigger_id]:
            continue

        pair_id = trig["pair_id"]
        z_trigger = trig["z_trigger"]
        trigger_date = trig["trigger_date"]

        long_stock, short_stock = _split_stocks(pair_id, z_trigger)

        z_pair = zscores[pair_id]

        # Entry: one day after the trigger day (t+1).
        trigger_day = zscores.index.get_loc(trigger_date)
        entry_day = trigger_day + 1
        if entry_day > last_day:
            dropped = dropped + 1
            continue

        exit_day = entry_day + config.MAX_HOLD_DAYS
        if exit_day > last_day:
            exit_day = last_day
        exit_reason = "timeout"

        gross_ret = 0.0
        curr_day = entry_day + 1
        while curr_day <= exit_day:
            date = zscores.index[curr_day]
            ret = returns.loc[date, long_stock] - returns.loc[date, short_stock]
            gross_ret += ret

            if abs(z_pair[date]) < config.EXIT_Z:
                exit_day = curr_day
                exit_reason = "reverted"
                break
            curr_day = curr_day + 1

        ledger_rows.append({
            "trigger_id": trigger_id,
            "pair_id": pair_id,
            "entry_date": zscores.index[entry_day],
            "exit_date": zscores.index[exit_day],
            "exit_reason": exit_reason,
            "days_held": exit_day - entry_day,
            "gross_ret": gross_ret,
        })

    if dropped > 0:
        print(f"run_backtest: dropped {dropped} trigger(s) with no next day to enter on")

    if len(ledger_rows):
        trades = pd.DataFrame(ledger_rows)
    else:
        # Empty trade
        trades = pd.DataFrame({
            "trigger_id": pd.Series([], dtype="str"),
            "pair_id": pd.Series([], dtype="str"),
            "entry_date": pd.Series([], dtype="datetime64[us]"),
            "exit_date": pd.Series([], dtype="datetime64[us]"),
            "exit_reason": pd.Series([], dtype="str"),
            "days_held": pd.Series([], dtype="int64"),
            "gross_ret": pd.Series([], dtype="float64"),
        })

    # Cost adjusted returns columns
    for c in cost_grid_bps:
        trades[f"net_ret_{c}bps"] = trades["gross_ret"] - 4 * c * 0.0001

    validate_artifact(trades, "trades")
    return trades


def daily_strategy_returns(trades: pd.DataFrame, prices: pd.DataFrame, triggers: pd.DataFrame, cost_bps: int = config.HEADLINE_COST_BPS) -> pd.Series:
    """One return per calendar day: equal-weight mean of open trades' daily P&L.

    For each day, take every trade open that day, average their
    (long return - short return) for the day, equal weight. Days with no
    open trade contribute 0.0. Costs enter the daily stream as -2c bps on
    each trade's entry day and -2c bps on its exit day (at the moment of
    each transaction, never as a lump at the end). No capital constraint:
    every accepted trade is always taken.

    Needs `prices` to rebuild each trade's daily leg returns (the ledger
    stores only the summed gross_ret) and `triggers` to recover each
    trade's direction (the ledger does not store which leg was long;
    z_trigger does, via _split_stocks).
    """
    returns = _simple_ret(prices)
    cost_per_transaction = 2 * cost_bps * 0.0001

    z_trigger_dict = {}
    for index, row in triggers.iterrows():
        z_trigger_dict[row["trigger_id"]] = row["z_trigger"]

    # each day's list of open-trade P&L values
    pnl_lists_dict = {}
    for date in prices.index:
        pnl_lists_dict[date] = []

    for index, trade in trades.iterrows():
        pair_id = trade["pair_id"]
        trigger_id = trade["trigger_id"]
        long_stock, short_stock = _split_stocks(pair_id, z_trigger_dict[trigger_id])
        entry_day = prices.index.get_loc(trade["entry_date"])
        exit_day = prices.index.get_loc(trade["exit_date"])

        # entry transaction cost
        entry_date = prices.index[entry_day]
        pnl_lists_dict[entry_date].append(-cost_per_transaction)
        if exit_day == entry_day:
            pnl_lists_dict[entry_date].append(-cost_per_transaction)
            continue

        # Every held day P&L
        curr_day = entry_day + 1
        while curr_day <= exit_day:
            date = prices.index[curr_day]
            pnl = returns.loc[date, long_stock] - returns.loc[date, short_stock]
            if curr_day == exit_day:
                # exit transaction cost
                pnl -= cost_per_transaction
            pnl_lists_dict[date].append(pnl)
            curr_day = curr_day + 1

    # Average P&L per day
    daily = []
    for date in prices.index:
        values = pnl_lists_dict[date]
        daily_total = 0.0
        if len(values):
            daily_total = sum(values) / len(values)

        daily.append(daily_total)

    return pd.Series(daily, index=prices.index)
