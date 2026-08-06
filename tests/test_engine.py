"""Engine tests: hand-made fixture with three planted trades.

The fixture below is a tiny invented world — 10 business days, 3 pairs,
6 tickers. Every number is written by hand so the correct ledger is
known before the engine runs. The expected ledger lives in
tests/golden/golden_engine_trades.csv.

The three planted stories (all trigger on day 3, 2020-01-06):
  Trade 1  AAA__BBB  z crosses +2.0, reverts below 0.5 on day 6
           -> entry day 4, exit day 6, "reverted", gross +0.04
  Trade 2  CCC__DDD  z crosses +2.0, never reverts
           -> entry day 4, timeout at day 9 (5 trading days), gross +0.01
  Trade 3  EEE__FFF  z crosses -2.0 (direction flips: long a, short b),
           reverts on day 5 -> entry day 4, exit day 5, gross +0.03

Run: pytest -q tests/test_engine.py
"""
import pandas as pd
import pytest

from src.engine import run_backtest, daily_strategy_returns


# The 10-day calendar (business days, hand-picked).
DAYS = pd.DatetimeIndex(
    [
        "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
        "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15",
    ],
    name="date",
)


def fake_zscores():
    """date x pair_id. Each column is a hand-written z story (see docstring)."""
    return pd.DataFrame(
        {
            "AAA__BBB": [0.5, 1.2, 2.4, 1.8, 1.0, 0.3, 0.1, 0.1, 0.1, 0.1],
            "CCC__DDD": [0.8, 1.5, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9],
            "EEE__FFF": [-0.4, -1.5, -2.2, -1.2, -0.4, -0.2, -0.1, -0.1, -0.1, -0.1],
        },
        index=DAYS,
    )


def fake_prices():
    """date x ticker. Flat except hand-placed round-percent moves:

    AAA: -1% on day 5                 BBB: +2% day 5, +1% day 6
    CCC: +1% on day 7                 DDD: +1% day 6, +1% day 9
    EEE: +2% on day 5                 FFF: -1% day 5
    """
    return pd.DataFrame(
        {
            "AAA": [100.0, 100.0, 100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "BBB": [50.0, 50.0, 50.0, 50.0, 51.0, 51.51, 51.51, 51.51, 51.51, 51.51],
            "CCC": [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 202.0, 202.0, 202.0, 202.0],
            "DDD": [80.0, 80.0, 80.0, 80.0, 80.0, 80.8, 80.8, 80.8, 81.608, 81.608],
            "EEE": [40.0, 40.0, 40.0, 40.0, 40.8, 40.8, 40.8, 40.8, 40.8, 40.8],
            "FFF": [120.0, 120.0, 120.0, 120.0, 118.8, 118.8, 118.8, 118.8, 118.8, 118.8],
        },
        index=DAYS,
    )


def fake_triggers():
    """Only the columns the engine reads (the full triggers schema has more).

    The engine gets each trade's two legs by splitting pair_id on "__".
    """
    return pd.DataFrame(
        {
            "trigger_id": ["AAA__BBB__20200106", "CCC__DDD__20200106", "EEE__FFF__20200106"],
            "pair_id": ["AAA__BBB", "CCC__DDD", "EEE__FFF"],
            "trigger_date": pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-06"]),
            "z_trigger": [2.4, 2.6, -2.2],
        }
    )


def fake_decisions():
    """Enter everything (the E0 shape): enter=True, p_hat=NaN."""
    return pd.DataFrame(
        {
            "trigger_id": ["AAA__BBB__20200106", "CCC__DDD__20200106", "EEE__FFF__20200106"],
            "enter": [True, True, True],
            "p_hat": [float("nan"), float("nan"), float("nan")],
        }
    )


def expected_ledger():
    """The hand-derived answer key, loaded from the committed golden CSV."""
    df = pd.read_csv(
        "tests/golden/golden_engine_trades.csv",
        parse_dates=["entry_date", "exit_date"],
    )
    return df


def assert_ledgers_match(got, want):
    """Compare two ledgers column by column, row by row.

    Text, date, and integer columns must match exactly; float columns
    must agree within 1e-9.
    """
    assert len(got) == len(want), f"row count: got {len(got)}, want {len(want)}"
    for col in want.columns:
        for i in range(len(want)):
            got_value = got[col].iloc[i]
            want_value = want[col].iloc[i]
            if isinstance(want_value, float):
                assert abs(got_value - want_value) < 1e-9, f"{col} row {i}: got {got_value}, want {want_value}"
            else:
                assert got_value == want_value, f"{col} row {i}: got {got_value}, want {want_value}"


# ------------------------------------------------- the four engine tests

def test_golden_ledger():
    trades = run_backtest(fake_zscores(), fake_prices(), fake_triggers(), fake_decisions())
    assert_ledgers_match(trades, expected_ledger())


def test_shift_trigger_day_price_changes_nothing():
    # Leakage check, part 1: move a traded stock's price ON the trigger
    # day. That changes the day-3 and day-4 returns — but the first P&L
    # day of any trade is day 5, so the ledger must not move at all.
    prices = fake_prices()
    prices.loc[prices.index[2], "AAA"] = 150.0  # day 3, the trigger day
    trades = run_backtest(fake_zscores(), prices, fake_triggers(), fake_decisions())
    assert_ledgers_match(trades, expected_ledger())


def test_shift_pnl_day_price_moves_gross_exactly():
    # Leakage check, part 2: move AAA from 99 to 98 for day 5 onward.
    # Day 5's AAA return becomes -2% (was -1%); later AAA returns stay 0.
    # Only trade 1 holds AAA, short side, so its gross becomes
    # (+2% - (-2%)) + (+1% - 0%) = 0.05. Trades 2 and 3 must not move.
    prices = fake_prices()
    for pos in range(4, 10):
        prices.loc[prices.index[pos], "AAA"] = 98.0
    trades = run_backtest(fake_zscores(), prices, fake_triggers(), fake_decisions())
    assert abs(trades["gross_ret"].iloc[0] - 0.05) < 1e-9
    assert abs(trades["gross_ret"].iloc[1] - 0.01) < 1e-9
    assert abs(trades["gross_ret"].iloc[2] - 0.03) < 1e-9
    assert trades["exit_date"].iloc[0] == expected_ledger()["exit_date"].iloc[0]


def test_cost_arithmetic():
    # net_ret_{c}bps = gross - 4 * c * 0.0001, for every trade.
    trades = run_backtest(fake_zscores(), fake_prices(), fake_triggers(), fake_decisions())
    for i in range(len(trades)):
        gross = trades["gross_ret"].iloc[i]
        assert abs(trades["net_ret_0bps"].iloc[i] - gross) < 1e-12
        assert abs(trades["net_ret_10bps"].iloc[i] - (gross - 0.004)) < 1e-12
        assert abs(trades["net_ret_50bps"].iloc[i] - (gross - 0.02)) < 1e-12


def test_daily_returns_concurrency():
    # Hand-derived daily series at 10 bps. Day numbers from the fixture
    # docstring; positions are day number - 1.
    #   day 4: three entry costs               -> -0.002
    #   day 5: T1 +0.03, T2 0.0, T3 +0.028    -> 0.058 / 3
    #   day 6: T1 +0.008 (exits), T2 +0.01    -> +0.009
    #   day 7: T2 -0.01                       -> -0.01
    #   day 8: T2 0.0                         ->  0.0
    #   day 9: T2 +0.008 (exits)              -> +0.008
    #   all other days: nothing open          ->  0.0
    trades = run_backtest(fake_zscores(), fake_prices(), fake_triggers(), fake_decisions())
    daily = daily_strategy_returns(trades, fake_prices(), fake_triggers(), cost_bps=10)
    assert len(daily) == 10
    expected_by_position = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: -0.002,
        4: 0.058 / 3,
        5: 0.009,
        6: -0.01,
        7: 0.0,
        8: 0.008,
        9: 0.0,
    }
    for pos in range(10):
        got = daily.iloc[pos]
        want = expected_by_position[pos]
        assert abs(got - want) < 1e-9, f"day position {pos}: got {got}, want {want}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
