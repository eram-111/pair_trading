import yfinance as yf
import pandas as pd
import numpy as np

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
    #MAX_MISSING FRAC = 0.02, drop ticker if >2% of trading days are missing, 
    #implied manual replacement, have not checked 

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
    
    for t in ticked: #For each ticker 
      get = t.history(start, end, auto_adjust = True)
      
      #Date here is already accounted for  
      price[t] = get['Close']
      vol[t] = get['Volume']

    #MSFT: table with date and close as columns

    prices = pd.DataFrame(price)
    volume = pd.Dataframe(vol)
                      
    return prices, volume

  #This works for now as pseudocode
  #NOT IMPLEMENTED: RETRY, CACHING

def clean_prices(prices: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs every spec Step 0 check as an explicit assert + printed report."""

    #Report here is tangential. Just clean. Idk what step 0 is. Hallucination?

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """np.log(p / p.shift(1)), first row dropped. LOG_RETURNS per config."""
    #Clean prices up here
    drop = {}

    for i in range(1, len(prices)):
        p = prices[i, 'Close']
        drop[i] = np.log(p/prices[i-1, 'Close'])

        #Unknown if Dataframe is indexed by number, unknown if Datatable datatype is mutable
        

    #drop the first row. calculate the returns (percentage im assuming) for every ticker for every date?
    #we get a dataframe of prices



