"""Fake random-walk stock data with no real relationships (noise-test input).
Run: python -m src.make_synthetic --seed 311 --out data/synth/raw
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.contracts import write_parquet


def make_synthetic(n_tickers: int = 40, start: str = "2014-01-01", end: str = "2025-01-01", seed: int = config.SEED, out_dir: Path = Path("data/synth/raw")):
    """Write prices.parquet, volume.parquet, spy.parquet under out_dir.
    One rng(seed) drives all draws; same seed gives byte-identical files, so draw order matters.
    """
    num_generator = np.random.default_rng(seed)

    calendar = pd.bdate_range(start=start, end=end, name="date").astype("datetime64[us]")
    n_days = len(calendar)

    tickers = []
    for i in range(n_tickers):
        tickers.append(f"SYN{i:02d}")

    prices = {}
    for ticker in tickers:
        volatility = num_generator.uniform(0.008, 0.025)
        starting_price = np.log(num_generator.uniform(20, 500))
        returns  = num_generator.normal(0, volatility, size=n_days)
        returns [0] = 0 
        cum_returns = np.cumsum(returns)
        log_price = np.array([starting_price] * n_days) + cum_returns
        prices[ticker] = np.exp(log_price)
    prices_df = pd.DataFrame(prices, index=calendar)

    volumes = {}
    for ticker in tickers:
        base = 10 ** num_generator.uniform(5, 7)  
        noise = num_generator.normal(0, 0.3, size=n_days)
        noise = np.exp(noise)
        volumes[ticker] = np.round(base * noise)
    volume_df = pd.DataFrame(volumes, index=calendar)

    sigma = num_generator.uniform(0.008, 0.025)
    starting_price = np.log(num_generator.uniform(20, 500))
    returns = num_generator.normal(0.0, sigma, size=n_days)
    returns[0] = 0.0
    cum_returns = np.cumsum(returns)
    log_price = np.array([starting_price] * n_days) + cum_returns
    spy_price=np.exp(log_price)
    spy_df = pd.DataFrame({"SPY": spy_price}, index=calendar)

    write_parquet(prices_df, out_dir / "prices.parquet", "prices")
    write_parquet(volume_df, out_dir / "volume.parquet", "volume")
    write_parquet(spy_df, out_dir / "spy.parquet", "spy")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-tickers", type=int, default=40)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", default="data/synth/raw")
    args = parser.parse_args()
    make_synthetic(n_tickers=args.n_tickers, start=args.start, end=args.end, seed=args.seed, out_dir=Path(args.out))
