"""Report analyses: label base rates and cross-track pair overlap.

Reads triggers/pairs files; writes tables under results/tables/ and
the by-year figure under results/figures/.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.contracts import read_parquet


def base_rate_table(triggers: pd.DataFrame) -> pd.DataFrame:
    """Label base rate overall, per split, per calendar year, and per
    regime (calm vs stressed, split at the median f_mkt_vol_20d).

    One row per grouping; columns: group, n_triggers, base_rate. The
    "split=train" row is the decision number: outside 15-85% must be
    escalated before any model is fit (label parameters may then be
    revisited on train+val only, with a DECISIONS.md entry).
    """
    raise NotImplementedError


def base_rate_by_year_figure(triggers: pd.DataFrame,
                             out_path: str = "results/figures/base_rate_by_year.png") -> None:
    """The decay figure: reversion rate per calendar year as a line,
    with each year's trigger count drawn as bars behind it."""
    raise NotImplementedError


def consensus_table(pairs_by_track: dict, triggers_by_track: dict,
                    trades_by_track: dict) -> pd.DataFrame:
    """Selection-count buckets: does agreement between tracks predict
    better trades?

    Expand each track's pairs table to (pair_id, quarter) rows from
    active_from/active_to, then bucket every pair-quarter by how many
    tracks selected it (1, 2, ...). Per bucket: n pair-quarters,
    n triggers, reversion rate (label mean from the triggers), and mean
    net return at the headline cost (from the trades ledgers). Plain
    pair-level intersection ignores WHEN a pair was live — the quarter
    expansion is what makes the overlap honest. Needs 2+ tracks; with
    one track the caller skips this and notes it in the report.
    """
    raise NotImplementedError


def main() -> None:
    """Base-rate table + figure for every triggers_{track}.parquet that
    exists (table also printed for the sync); consensus_table only when
    2+ tracks exist."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
