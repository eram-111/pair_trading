"""Makes fake stock data: random-walk prices and volumes for 40 made-up
tickers (SYN00..SYN39) plus SPY, with no real relationships in it.

Used two ways: stand-in data while we build the pipeline, and the input
for the noise test — if the pipeline finds profit in this data, we have
a leakage bug. Files have the same shape as the real ones in data/raw/.

Create fake data: python -m src.make_synthetic --seed 311 --out data/synth/raw

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

    Log prices p_t = p_{t-1} + eps, eps ~ N(0, sigma_i), with
    sigma_i ~ U(0.008, 0.025) drawn once per ticker; p_0 = ln(U(20, 500)).
    Volume: round(base_i * exp(N(0, 0.3))), base_i ~ logU(1e5, 1e7).
    One np.random.default_rng(seed) drives everything — same seed, same
    files, byte for byte. (Draw order is part of that guarantee: reordering
    the loops below changes the output.)
    """
    num_generator = np.random.default_rng(seed)

    calendar = pd.bdate_range(start=start, end=end, name="date").astype("datetime64[us]")
    n_days = len(calendar)

    tickers = []
    for i in range(n_tickers):
        tickers.append(f"SYN{i:02d}")

    prices = {}
    for ticker in tickers:
        valotility = num_generator.uniform(0.008, 0.025)
        starting_price = np.log(num_generator.uniform(20, 500))
        returns  = num_generator.normal(0, valotility, size=n_days)
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
