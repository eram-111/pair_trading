import yfinance as yf
import pandas as pd

#SECTORS: dict[str, list[str]]   # the table below, as a literal
#TICKERS: list[str]              # sorted flat list of 40

ticks = {}
ticks['it'] = ['AAPL', 'MSFT', 'NVDA', 'ORCL', 'CSCO', 'INTC', 'IBM', 'TXN', 'ADBE', 'QCOM']
ticks['fn'] = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB','PNC','AXP','BLK']
ticks['en'] = ['XOM', 'CVX','COP','SLB','EOG','OXY','VLO','MPC','PSX','HAL']
ticks['cs'] = ['PG','KO','PEP','WMT','COST','MDLZ','CL','KMB','GIS','SYY']

def get_universe() -> pd.DataFrame:
    """Returns DataFrame[ticker, sector]. Single source of truth for every module."""
    #Unsure if the shape is correct here
    df = pd.DataFrame(ticks)
    return df

def download_prices(tickers: list[str], start: str = "2014-01-01",
                    end: str = "2025-01-01", max_retries: int = 3,
                    cache_dir: str = "data/raw/cache/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance pull, auto_adjust=True asserted explicitly (do not trust the default
    silently). Per-ticker retry with exponential backoff; raw per-ticker CSVs cached
    and committed, so the pull is reproducible even if yfinance data shifts later.
    Returns (prices, volume), date x ticker.""" 
                      
    ticked = yf.Tickers(tickers)
    price = {}
    vol = {}
    
    for t in ticked:
      get = t.history(start, end, auto_adjust = True)
      ptic = get['Close'], get['Date']
      vtic = get['Volume'], get['Date']
      price[t] = ptic
      vol[t] = vtic

    prices = pd.DataFrame(price)
    volume = pd.Dataframe(vol)
                      
    return prices, volume

  #This works for now as pseudocode
  #NOT IMPLEMENTED: RETRY, CACHING


def clean_prices(prices: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs every spec Step 0 check as an explicit assert + printed report."""

#Report here is tangential. Just clean

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """np.log(p / p.shift(1)), first row dropped. LOG_RETURNS per config."""
