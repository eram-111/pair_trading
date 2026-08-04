"""Tests for src/contracts.py and src/make_synthetic.py.

Two kinds of test: a good frame passes validation, and a frame with
exactly one thing broken fails it. Plus: the synthetic fixture files
validate, and two same-seed runs are byte-identical.

Run: pytest -q tests/test_contracts.py
or:  python -m tests.test_contracts   (from the repo root)

(tmp_path is a pytest built-in: each test that names it as a parameter
receives a fresh empty temp directory as a Path.)
"""
import pandas as pd
import pytest

from src import config
from src.contracts import read_parquet, validate_artifact, write_parquet
from src.make_synthetic import make_synthetic


# ------------------------------------------------------- tiny valid frames

def good_decisions():
    return pd.DataFrame({
        "trigger_id": ["AAA__BBB__20200106"],
        "enter": [True],
        "p_hat": [0.5],
    })


def good_prices():
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="date")
    return pd.DataFrame({"SYN00": [100.0, 101.0], "SYN01": [50.0, 49.5]}, index=dates)


def good_pairs():
    return pd.DataFrame({
        "pair_id": ["AAA__BBB"],
        "stock_a": ["AAA"],
        "stock_b": ["BBB"],
        "group_id": ["20200101_0_0"],
        "source": ["track_a"],
        "active_from": pd.to_datetime(["2020-01-02"]),
        "active_to": pd.to_datetime(["2020-02-03"]),
    })


def good_triggers():
    df = pd.DataFrame({
        "trigger_id": ["AAA__BBB__20200106"],
        "pair_id": ["AAA__BBB"],
        "source": ["track_a"],
        "trigger_date": pd.to_datetime(["2020-01-06"]),
        "z_trigger": [2.1],
        "f_abs_z": [2.1],
        "f_spread_vol_60d": [0.01],
        "f_resid_mom_5d": [0.5],
        "f_mkt_vol_20d": [0.02],
        "f_rel_volume_20d": [1.1],
        "f_days_since_trigger": [126.0],
        "f_cluster_stability": [1.0],
        "label": [1],
        "horizon_end_date": pd.to_datetime(["2020-01-13"]),
        "split": ["train"],
    })
    df["label"] = df["label"].astype("int8")
    return df


def good_trades():
    df = pd.DataFrame({
        "trigger_id": ["AAA__BBB__20200106"],
        "pair_id": ["AAA__BBB"],
        "entry_date": pd.to_datetime(["2020-01-07"]),
        "exit_date": pd.to_datetime(["2020-01-09"]),
        "exit_reason": ["reverted"],
        "days_held": [2],
        "gross_ret": [0.01],
    })
    for c in config.COST_GRID_BPS:
        df[f"net_ret_{c}bps"] = [0.01 - 4 * c * 0.0001]
    return df


def validation_fails(df, name):
    """True if validate_artifact rejects (df, name), False if it passes.

    The try/except lives here, once, so the tests below stay one line.
    """
    try:
        validate_artifact(df, name)
        return False
    except AssertionError:
        return True


# ------------------------------------------------- structure checks

def test_good_decisions_passes():
    validate_artifact(good_decisions(), "decisions")


def test_unknown_artifact_name_fails():
    assert validation_fails(good_decisions(), "no_such_artifact")


def test_missing_column_fails():
    df = good_decisions().drop(columns=["enter"])
    assert validation_fails(df, "decisions")


def test_wrong_dtype_fails():
    df = good_decisions()
    df["p_hat"] = [1]  # int64, schema wants float64
    assert validation_fails(df, "decisions")


def test_unexpected_extra_column_fails():
    df = good_decisions()
    df["debug"] = [1.0]
    assert validation_fails(df, "decisions")


def test_good_prices_passes():
    validate_artifact(good_prices(), "prices")


def test_unsorted_dates_fail():
    df = good_prices().iloc[::-1]  # reverses the row order
    assert validation_fails(df, "prices")


def test_non_float_ticker_column_fails():
    df = good_prices()
    df["SYN02"] = ["a", "b"]
    assert validation_fails(df, "prices")


# ------------------------------------------------- per-artifact rule checks

def test_good_pairs_passes():
    validate_artifact(good_pairs(), "pairs")


def test_pair_id_not_alphabetical_fails():
    df = good_pairs()
    df["stock_a"] = ["ZZZ"]  # now stock_a > stock_b
    assert validation_fails(df, "pairs")


def test_bad_source_fails():
    df = good_pairs()
    df["source"] = ["track_z"]
    assert validation_fails(df, "pairs")


def test_good_triggers_passes():
    validate_artifact(good_triggers(), "triggers")


def test_bad_split_fails():
    df = good_triggers()
    df["split"] = ["trian"]
    assert validation_fails(df, "triggers")


def test_label_not_binary_fails():
    df = good_triggers()
    df["label"] = pd.array([2], dtype="int8")
    assert validation_fails(df, "triggers")


def test_missing_cost_column_fails():
    df = good_trades().drop(columns=["net_ret_30bps"])
    assert validation_fails(df, "trades")


# ------------------------------------------------- io + synthetic fixture

def test_parquet_round_trip(tmp_path):
    write_parquet(good_prices(), tmp_path / "prices.parquet", "prices")
    df = read_parquet(tmp_path / "prices.parquet")
    assert df.index.name == "date"
    assert list(df.columns) == ["SYN00", "SYN01"]


def test_make_synthetic_outputs_validate(tmp_path):
    make_synthetic(n_tickers=3, start="2020-01-01", end="2020-03-01", out_dir=tmp_path)
    validate_artifact(read_parquet(tmp_path / "prices.parquet"), "prices")
    validate_artifact(read_parquet(tmp_path / "volume.parquet"), "volume")
    validate_artifact(read_parquet(tmp_path / "spy.parquet"), "spy")


def test_make_synthetic_is_deterministic(tmp_path):
    make_synthetic(n_tickers=3, start="2020-01-01", end="2020-03-01", seed=311, out_dir=tmp_path / "run1")
    make_synthetic(n_tickers=3, start="2020-01-01", end="2020-03-01", seed=311, out_dir=tmp_path / "run2")
    bytes1 = (tmp_path / "run1" / "prices.parquet").read_bytes()
    bytes2 = (tmp_path / "run2" / "prices.parquet").read_bytes()
    assert bytes1 == bytes2, "same seed must produce byte-identical files"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
