"""Defines what every data file in this project must look like, and checks it.

SCHEMAS lists each file's expected columns, dtypes, and index.
validate_artifact(df, name) raises AssertionError when a frame doesn't
match. write_parquet/read_parquet route all file IO through that check,
so a malformed file never lands on disk. seed_everything makes every run
repeatable. Changing a schema needs team agreement + a DECISIONS.md entry.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


class ArtifactSchema:
    """Expected shape of one data artifact.

    columns — dict(fixed column name, pandas dtype string) ({} if no fixed columns).

    index — required index name ("date"), or None for tables with a plain
    RangeIndex. A date index must be datetime64, tz-naive, and
    strictly increasing.

    extra_float_cols — True for tables that also carry data-dependent
    columns (one per ticker / pair_id / characteristics field); those
    must all be float64. False = no columns beyond `columns` allowed.

    checks — extra per-artifact rules, each a function (df) -> None that
    raises AssertionError with a clear message.
    """
    def __init__(self, columns: dict[str, str], index: str | None = None, extra_float_cols: bool = False, checks: tuple = ()):
        self.columns = columns
        self.index = index
        self.extra_float_cols = extra_float_cols
        self.checks = checks


# --------------------------------------------------------------- invariants
# Check functions

def _check_pair_id_alphabetical(df: pd.DataFrame) -> None:
    """pair_id legs are alphabetical.

    Where stock_a/stock_b columns exist (pairs): stock_a < stock_b and
    pair_id == f"{stock_a}__{stock_b}". Where they don't (stability has
    only pair_id): split pair_id on "__" and assert the legs are sorted.
    """
    if "stock_a" in df.columns:
        for index, row in df.iterrows():
            assert row["stock_a"] < row["stock_b"], f"stock_a must be < stock_b; bad pair_id: '{row['pair_id']}'"
            expected = row["stock_a"] + "__" + row["stock_b"]
            assert row["pair_id"] == expected, f"pair_id must equal stock_a__stock_b; got '{row['pair_id']}'"
    else:
        for pair_id in df["pair_id"]:
            legs = pair_id.split("__")
            assert len(legs) == 2, f"pair_id must look like 'AAA__BBB'; got '{pair_id}'"
            assert legs[0] < legs[1], f"pair_id legs must be alphabetical; got '{pair_id}'"


def _check_trigger_id_format(df: pd.DataFrame) -> None:
    """trigger_id == f"{pair_id}__{trigger_date:%Y%m%d}" (double underscore), unique."""
    seen = set()
    for index, row in df.iterrows():
        trigger_id = row["trigger_id"]
        assert trigger_id not in seen, f"duplicate trigger_id '{trigger_id}'"
        seen.add(trigger_id)
        expected = row["pair_id"] + "__" + row["trigger_date"].strftime("%Y%m%d")
        assert trigger_id == expected, f"trigger_id must be pair_id__YYYYMMDD; got '{trigger_id}', expected '{expected}'"


def _check_split_values(df: pd.DataFrame) -> None:
    """split ∈ {train, val, test, purged, embargo}."""
    allowed = ("train", "val", "test", "purged", "embargo")
    for value in df["split"]:
        assert value in allowed, f"split value '{value}' not allowed; allowed: {allowed}"


def _check_label_binary(df: pd.DataFrame) -> None:
    """label ∈ {0, 1}."""
    for value in df["label"]:
        assert value == 0 or value == 1, f"label must be 0 or 1; got {value}"


def _check_source_values(df: pd.DataFrame) -> None:
    """source ∈ {track_a, track_b, track_c}."""
    allowed = ("track_a", "track_b", "track_c")
    for value in df["source"]:
        assert value in allowed, f"source value '{value}' not allowed; allowed: {allowed}"


def _check_cost_columns(df: pd.DataFrame) -> None:
    """One net_ret_{c}bps column per c in config.COST_GRID_BPS, all present."""
    for c in config.COST_GRID_BPS:
        col = f"net_ret_{c}bps"
        assert col in df.columns, f"missing cost column '{col}'; cost grid is {config.COST_GRID_BPS}"


# Add more ArtifactSchema if needed or change the format if needed
SCHEMAS: dict[str, ArtifactSchema] = {
    # ---------------- raw data (written by data.py) ----------------
    "prices": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    "volume": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    "spy": ArtifactSchema(columns={"SPY": "float64"}, index="date"),
    "universe": ArtifactSchema(columns={"ticker": "str", "sector": "str", "included": "bool"}),
    # ---------------- processed substrate (written by data.py / representation.py) ----------------
    "returns": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    "factors_a": ArtifactSchema(columns={"pc_1": "float64", "pc_2": "float64", "pc_3": "float64", "pc_4": "float64", "pc_5": "float64"}, index="date"),
    "pca_meta": ArtifactSchema(columns={"n_components": "int64", "cum_var_explained": "float64"}, index="date"),
    "loadings_a": ArtifactSchema(columns={"date": "datetime64[us]", "ticker": "str", "component": "int64", "loading": "float64", "beta": "float64"}),
    "residuals_a": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    # ---------------- Track B characteristics (written by characteristics.py): fixed cols + one float64 col per field ----------------
    "characteristics_raw": ArtifactSchema(columns={"quarter_end": "datetime64[us]", "ticker": "str"}, extra_float_cols=True),
    "characteristics_clean": ArtifactSchema(columns={"quarter_end": "datetime64[us]", "ticker": "str"}, extra_float_cols=True),
    # ---------------- clustering outputs (one file per track) ----------------
    "labels": ArtifactSchema(columns={"window_end": "datetime64[us]", "ticker": "str", "cluster_id": "int64"}),
    "stability": ArtifactSchema(columns={"window_end": "datetime64[us]", "pair_id": "str", "co_clustered": "bool"}, checks=(_check_pair_id_alphabetical,)),
    "pairs": ArtifactSchema(columns={"pair_id": "str", "stock_a": "str", "stock_b": "str", "group_id": "str", "source": "str", "active_from": "datetime64[us]", "active_to": "datetime64[us]"}, checks=(_check_pair_id_alphabetical, _check_source_values)),
    # ---------------- spreads / z-scores (written by representation.py, one file per track) ----------------
    "spreads": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    "zscores": ArtifactSchema(columns={}, index="date", extra_float_cols=True),
    # ---------------- trigger dataset (written by dataset.py) — the central contract ----------------
    "triggers": ArtifactSchema(columns={"trigger_id": "str", "pair_id": "str", "source": "str", "trigger_date": "datetime64[us]", "z_trigger": "float64", "f_abs_z": "float64", "f_spread_vol_60d": "float64", "f_resid_mom_5d": "float64", "f_mkt_vol_20d": "float64", "f_rel_volume_20d": "float64", "f_days_since_trigger": "float64", "f_cluster_stability": "float64", "label": "int8", "horizon_end_date": "datetime64[us]", "split": "str"}, checks=(_check_trigger_id_format, _check_split_values, _check_label_binary, _check_source_values)),
    # ---------------- results (written by engine.py / experiments.py) ----------------
    "decisions": ArtifactSchema(columns={"trigger_id": "str", "enter": "bool", "p_hat": "float64"}),
    "trades": ArtifactSchema(columns={"trigger_id": "str", "pair_id": "str", "entry_date": "datetime64[us]", "exit_date": "datetime64[us]", "exit_reason": "str", "days_held": "int64", "gross_ret": "float64", "net_ret_0bps": "float64", "net_ret_5bps": "float64", "net_ret_10bps": "float64", "net_ret_15bps": "float64", "net_ret_20bps": "float64", "net_ret_30bps": "float64", "net_ret_40bps": "float64", "net_ret_50bps": "float64"}, checks=(_check_cost_columns,)),  
}


# ---------------------------------------------------------------- validation
def validate_artifact(df: pd.DataFrame, name: str) -> None:
    """Assert df matches SCHEMAS[name]; raise AssertionError otherwise.

    Checks, in order:
      1. name is registered.
      2. Every column named in schema.columns is present with its exact
         dtype. Any other column is an error unless extra_float_cols is
         True, in which case it must be float64.
      3. Index, when schema.index is set: name matches, datetime64,
         tz-naive, monotonic increasing, unique.
      4. Every callable in schema.checks passes.

    Producers call this before every write (normally via write_parquet).
    """
    # 1. name is registered
    assert name in SCHEMAS, f"unknown artifact name '{name}'; known: {sorted(SCHEMAS)}"
    artifact = SCHEMAS[name]

    # 2. fixed columns: present, exact dtype
    for col, wanted_dtype in artifact.columns.items():
        assert col in df.columns, f"{name}: missing column '{col}'"
        actual_dtype = str(df[col].dtype)
        assert actual_dtype == wanted_dtype, f"{name}: column '{col}' must be {wanted_dtype}, got {actual_dtype}"

    extras = []
    for c in df.columns:
        if c not in artifact.columns:
            extras.append(c)

    if artifact.extra_float_cols:
        for col in extras:
            actual_dtype = str(df[col].dtype)
            assert actual_dtype == "float64", f"{name}: extra column '{col}' must be float64, got {actual_dtype}"
    else:
        assert extras == [], f"{name}: unexpected columns {extras}"

    # 3. index discipline
    if artifact.index is not None:
        assert df.index.name == artifact.index, f"{name}: index must be named '{artifact.index}', got '{df.index.name}'"
        assert isinstance(df.index, pd.DatetimeIndex),  f"{name}: index must be a DatetimeIndex, got {type(df.index).__name__}"
        assert df.index.tz is None, f"{name}: index must be tz-naive, got tz={df.index.tz}"
        assert df.index.is_monotonic_increasing, f"{name}: index dates must be sorted ascending"
        assert df.index.is_unique, f"{name}: index has duplicate dates"

    # 4. per-artifact rules
    for check in artifact.checks:
        check(df)



# ---------------------------------------------------------------- parquet IO
def write_parquet(df: pd.DataFrame, path: Path | str, name: str) -> None:
    """Validate then write — the only sanctioned parquet write path.

    Order of operations:
      1. validate_artifact(df, name)
      2. Coerce column labels to str (pyarrow refuses non-string labels).
      3. Assert any date index is named "date" and tz-naive.
      4. df.to_parquet(path)  # pyarrow engine; parent dir created if missing

    `name` is the SCHEMAS key (base name, e.g. "triggers" for
    data/datasets/triggers_a.parquet).
    """
    validate_artifact(df, name)

    df = df.copy()  # never mutate the caller's frame
    new_columns = []
    for c in df.columns:
        new_columns.append(str(c))
    df.columns = new_columns

    if isinstance(df.index, pd.DatetimeIndex):
        assert df.index.name == "date", f"{name}: date index must be named 'date', got '{df.index.name}'"
        assert df.index.tz is None, f"{name}: date index must be tz-naive, got tz={df.index.tz}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def read_parquet(path: Path | str) -> pd.DataFrame:
    """pd.read_parquet + assert the index name survived the round-trip."""
    df = pd.read_parquet(path, engine="pyarrow")
    if isinstance(df.index, pd.DatetimeIndex):
        assert df.index.name == "date", f"{path}: date index lost its name in the round-trip, got '{df.index.name}'"
    return df


def write_validated_csv(df: pd.DataFrame, path: Path | str, name: str) -> None:
    """Validate then write as CSV — for human-readable artifacts."""
    validate_artifact(df, name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ------------------------------------------------------------------- seeding
def seed_everything(seed: int = config.SEED) -> None:
    """Seed random, numpy, and torch; enable deterministic algorithms.

    Called by every pipeline entry point and every test. torch is imported
    lazily inside this function so non-model code paths never require it at
    import time.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
