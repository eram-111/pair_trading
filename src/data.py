import argparse
from datetime import datetime
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

from src import config
from src.contracts import write_parquet, write_validated_csv

ticks = {}
ticks['it'] = ['AAPL', 'MSFT', 'NVDA', 'ORCL', 'CSCO', 'INTC', 'IBM', 'TXN', 'ADBE', 'QCOM']
ticks['fn'] = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB','PNC','AXP','BLK']
ticks['en'] = ['XOM', 'CVX','COP','SLB','EOG','OXY','VLO','MPC','PSX','HAL']
ticks['cs'] = ['PG','KO','PEP','WMT','COST','MDLZ','CL','KMB','GIS','SYY']

def get_universe() -> pd.DataFrame:
    """One row per ticker, columns: ticker, sector, included.

    `included` starts True for everyone; clean_prices flips it to False 
    for any ticker dropped due to missing-data
    """
    rows = []
    for sector in ticks:
        for ticker in ticks[sector]:
            rows.append({"ticker": ticker, "sector": sector, "included": True})
    return pd.DataFrame(rows)

def download_prices(tickers: list[str], start_date: str = config.DOWNLOAD_START, end_date: str = config.DATA_END,
                    cache_dir: str = "data/raw/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance pull, auto_adjust=True asserted explicitly (do not trust the default
    silently). Raw per-ticker CSVs cached and committed, so the pull is
    reproducible even if yfinance data shifts later. No retry logic: the one
    attended pull is simply rerun if it fails.
    Returns (prices, volume), date x ticker.""" 
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    closing_prices = data["Close"][tickers]
    volume = data["Volume"][tickers]

    # Remove timezone
    if closing_prices.index.tz is not None:
        closing_prices.index = closing_prices.index.tz_localize(None)
    if volume.index.tz is not None:
        volume.index = volume.index.tz_localize(None)

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        csv_data = pd.DataFrame({"close": closing_prices[ticker], "volume": volume[ticker]})
        csv_data.index.name = "date"
        csv_data.to_csv(cache / f"{ticker}.csv")

    return closing_prices, volume


def load_cached_prices(tickers: list[str], cache_dir: str = "data/raw/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (prices, volume) from the saved CSVs."""
    closing_prices = {}
    volumes = {}
    for ticker in tickers:
        per_ticker = pd.read_csv(Path(cache_dir) / f"{ticker}.csv", index_col="date", parse_dates=["date"])
        closing_prices[ticker] = per_ticker["close"]
        volumes[ticker] = per_ticker["volume"]
    return pd.DataFrame(closing_prices), pd.DataFrame(volumes)

def clean_prices(prices: pd.DataFrame, volume: pd.DataFrame, spy: pd.DataFrame,
                 universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turn the holey downloaded tables into a perfect rectangle of numbers.

    Downloaded data has holes (NaN) and junk days; everything downstream
    does math on this table and cannot handle holes. This function
    removes the holes in a fixed, documented way and records what it
    threw out. Returns (prices, volume, spy, universe).
    """
    # 1. trading calendar = SPY's calendar
    trading_days = prices.index[prices.index.isin(spy.index)]
    prices = prices.loc[trading_days]
    volume = volume.loc[trading_days]

    # 2. drop tickers with too much missing data; record it in universe
    universe = universe.copy()
    dropped_tickers = []
    for ticker in list(prices.columns):
        missing_frac = prices[ticker].isna().mean()
        if missing_frac > config.MAX_MISSING_FRAC:
            dropped_tickers.append(ticker)
            row_select_mask = universe["ticker"] == ticker
            column = "included"
            universe.loc[row_select_mask, column] = False
    prices = prices.drop(columns=dropped_tickers)
    volume = volume.drop(columns=dropped_tickers)

    # 3. keep only dates where every surviving ticker has a price
    full_rows = prices.notna().all(axis=1)
    prices = prices.loc[full_rows]
    volume = volume.loc[full_rows]
    spy = spy[spy.index.isin(prices.index)]

    # 4. a missing volume on a kept day counts as zero shares
    volume = volume.fillna(0.0)
    volume = volume.astype("float64")

    # 5.  prices must be strictly positive at any scale
    assert (prices > 0).all().all(), "non-positive price: bad download or adjustment"

    print(f"clean_prices: {prices.shape[0]} days x {prices.shape[1]} tickers, from {prices.index[0].date()} to {prices.index[-1].date()}")
    if len(dropped_tickers) > 0:
        print(f"clean_prices: dropped {dropped_tickers}")
    else:
        print("clean_prices: all tickers survived")

    return prices, volume, spy, universe


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily LOG returns: ln(p_t / p_{t-1}), one column per ticker.

    Log, not simple, per config.LOG_RETURNS: log returns add up over
    time, which the PCA/statistics substrate relies on. The engine
    separately computes SIMPLE returns from prices for P&L (money and
    statistics deliberately use different return definitions — do not
    unify them). The first row (no previous day) is dropped.
    """

    prices_shifted_forward = prices.shift(1)
    returns = np.log(prices / prices_shifted_forward)
    returns = returns.drop(index=returns.index[0])
    return returns


def main() -> None:
    """Build every raw/processed data artifact.

    Two modes:
        python -m src.data          read the existing cache instead of downloading.
                                    Fails loudly if any cache file is
                                    missing.
        python -m src.data --pull   downloads data and refreshes the whole cache 
                                    and writes LOG.md.
                                   
    Then clean data, compute returns and write everything in files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()

    universe = get_universe()
    tickers = list(universe["ticker"])

    if args.pull:
        prices, volume = download_prices(tickers)
        spy, spy_volume = download_prices(["SPY"])
        log = (f"Pulled at {datetime.now()} with yfinance, from {config.DOWNLOAD_START} to {config.DATA_END}\n")
        Path("data/raw/LOG.md").write_text(log)
    else:
        cache = Path("data/raw/cache")
        missing = []
        for ticker in tickers + ["SPY"]:
            if not (cache / f"{ticker}.csv").exists():
                missing.append(ticker)
        assert missing == [], (f"cache is missing {missing}, restore data/raw/cache/ or pull new data.")
        prices, volume = load_cached_prices(tickers)
        spy, spy_volume = load_cached_prices(["SPY"])

    # a fresh pull names the index "Date"; the contract wants "date"
    prices.index.name = "date"
    volume.index.name = "date"
    spy.index.name = "date"

    prices, volume, spy, universe = clean_prices(prices, volume, spy, universe)

    returns = compute_returns(prices)

    write_parquet(prices, "data/raw/prices.parquet", "prices")
    write_parquet(volume, "data/raw/volume.parquet", "volume")
    write_parquet(spy, "data/raw/spy.parquet", "spy")
    write_parquet(returns, "data/processed/returns.parquet", "returns")
    write_validated_csv(universe, "data/raw/universe.csv", "universe")
    print("data: wrote prices, volume, spy, returns, universe")


if __name__ == "__main__":
    main()



