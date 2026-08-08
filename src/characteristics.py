"""Build Track B pairs from quarterly Bloomberg characteristics"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.contracts import read_parquet, write_parquet, write_validated_csv
from src.representation import build_pairs, cluster_all_windows


EXPECTED_FIELDS = (
    "pe_ratio",
    "price_to_book",
    "price_to_sales",
    "price_to_ebitda",
    "market_cap",
    "shares_outstanding",
    "sales_growth",
    "cash_flow_growth",
    "free_cash_flow_growth",
    "normalized_roe",
    "dividend_per_share",
    "volatility_60d",
    "rsi_14d",
    "close_price",
    "ask_price",
    "bid_price",
    "analyst_rating",
    "buy_recommendations",
    "sell_recommendations",
)


def read_field_csv(file_path: Path) -> pd.DataFrame:
    """Read one characteristic CSV and return it in long format"""
    file_path = Path(file_path)
    field = file_path.stem

    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError(f"{file_path.name} needs a date column and at least one ticker")

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"{file_path.name} has dates that could not be parsed")

    new_names = {}
    for col in df.columns[1:]:
        new_names[col] = col.split()[0]
    if len(set(new_names.values())) != len(new_names):
        raise ValueError(f"{file_path.name} has tickers that clash after cleaning")
    df = df.rename(columns=new_names)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.melt(id_vars="date", var_name="ticker", value_name="value")
    result["field"] = field
    return result[["date", "ticker", "field", "value"]]


def load_raw(raw_directory: str | Path = "data/raw/characteristics") -> pd.DataFrame:
    """Read all expected characteristic CSVs and combine them"""
    raw_directory = Path(raw_directory)

    files = []
    for f in sorted(raw_directory.glob("*.csv")):
        if f.stem in EXPECTED_FIELDS:
            files.append(f)
    if not files:
        raise ValueError(f"no known characteristic CSVs in {raw_directory}")

    found = []
    for f in files:
        found.append(f.stem)

    missing = []
    for field in EXPECTED_FIELDS:
        if field not in found:
            missing.append(field)
    if missing:
        raise ValueError(f"missing expected fields: {missing}")

    tables = []
    for f in files:
        tables.append(read_field_csv(f))
    result = pd.concat(tables, ignore_index=True)

    n_tickers = result["ticker"].nunique()
    if n_tickers != config.N_TICKERS:
        raise ValueError(f"expected {config.N_TICKERS} tickers, found {n_tickers}")

    print(f"load_raw found {len(files)} fields and {n_tickers} tickers")
    tickers = set(result["ticker"])
    for field, group in result.groupby("field"):
        if set(group["ticker"]) != tickers:
            raise ValueError(f"{field} does not have the same tickers as the other fields")
        print(f"{field}: {group['value'].notna().mean():.0%} coverage")

    result = result.sort_values(["date", "ticker", "field"]).reset_index(drop=True)
    return result


def convert_to_quarterly(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Move every ticker and field onto the same quarter end dates"""
    for col in ["date", "ticker", "field", "value"]:
        if col not in raw_data.columns:
            raise ValueError(f"raw_data is missing column: {col}")

    df = raw_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any():
        raise ValueError("date has missing values")
    if df.duplicated(["date", "ticker", "field"]).any():
        raise ValueError("raw_data has duplicate date, ticker and field rows")

    quarter_ends = pd.date_range(df["date"].min(), df["date"].max(), freq="QE")
    if quarter_ends.empty:
        raise ValueError("the data does not reach a single quarter end")

    rows = []
    for (field, ticker), group in df.groupby(["field", "ticker"]):
        series = group.set_index("date")["value"].sort_index()
        quarterly = series.reindex(quarter_ends, method="ffill")
        for quarter_end, value in quarterly.items():
            rows.append({"quarter_end": quarter_end, "ticker": ticker, "field": field, 
                         "value": value})

    result = pd.DataFrame(rows, columns=["quarter_end", "ticker", "field", "value"])
    result = result.sort_values(["quarter_end", "ticker", "field"]).reset_index(drop=True)
    return result


def make_raw_characteristics_table(quarterly_data: pd.DataFrame) -> pd.DataFrame:
    """Create one raw characteristics row per quarter and ticker"""
    for col in ["quarter_end", "ticker", "field", "value"]:
        if col not in quarterly_data.columns:
            raise ValueError(f"quarterly_data is missing column: {col}")
    if quarterly_data.duplicated(["quarter_end", "ticker", "field"]).any():
        raise ValueError("quarterly_data has duplicate quarter_end, ticker and field rows")

    result = quarterly_data.pivot(index=["quarter_end", "ticker"], columns="field", values="value")
    result = result.reset_index()
    result.columns.name = None
    result["quarter_end"] = result["quarter_end"].astype("datetime64[us]")
    result["ticker"] = result["ticker"].astype("str")
    for col in result.columns[2:]:
        result[col] = result[col].astype("float64")
    result = result.sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
    return result


def clean_snapshot(quarterly_data: pd.DataFrame, quarter_end: pd.Timestamp) -> pd.DataFrame:
    """Clean and standardize the characteristics for one quarter"""
    for col in ["quarter_end", "ticker", "field", "value"]:
        if col not in quarterly_data.columns:
            raise ValueError(f"quarterly_data is missing column: {col}")

    quarter_end = pd.Timestamp(quarter_end)
    current_quarter = quarterly_data[quarterly_data["quarter_end"] == quarter_end]
    if current_quarter.empty:
        raise ValueError(f"no rows for quarter {quarter_end:%Y-%m-%d}")
    if current_quarter.duplicated(["ticker", "field"]).any():
        raise ValueError(f"quarter {quarter_end:%Y-%m-%d} has duplicate ticker and field rows")

    table = current_quarter.pivot(index="ticker", columns="field", values="value")
    table.columns.name = None

    low_coverage = []
    for col in list(table.columns):
        if table[col].notna().mean() < 0.90:
            low_coverage.append(col)
            table = table.drop(columns=col)
    if low_coverage:
        print(f"{quarter_end:%Y-%m-%d}: dropped for low coverage: {low_coverage}")

    have_buys = "buy_recommendations" in table.columns
    have_sells = "sell_recommendations" in table.columns
    if have_buys and have_sells:
        buys = table["buy_recommendations"]
        sells = table["sell_recommendations"]
        total = buys + sells
        sentiment = (buys - sells) / total
        sentiment[total == 0] = np.nan
        table["sentiment"] = sentiment
        table = table.drop(columns=["buy_recommendations", "sell_recommendations"])
    elif have_buys:
        table = table.drop(columns="buy_recommendations")
        print(f"{quarter_end:%Y-%m-%d}: dropped buy_recommendations, no sell side")
    elif have_sells:
        table = table.drop(columns="sell_recommendations")
        print(f"{quarter_end:%Y-%m-%d}: dropped sell_recommendations, no buy side")

    bad_sizes = 0
    for col in ["market_cap", "shares_outstanding"]:
        if col in table.columns:
            bad = table[col] <= 0
            bad_sizes += int(bad.sum())
            table.loc[bad, col] = np.nan
            table[col] = np.log(table[col])
    if bad_sizes:
        print(f"{quarter_end:%Y-%m-%d}: {bad_sizes} non-positive size values set to missing")

    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())

    no_variation = []
    for col in list(table.columns):
        std = table[col].std()
        if pd.isna(std) or std < 1e-8:
            no_variation.append(col)
            table = table.drop(columns=col)
    if no_variation:
        print(f"{quarter_end:%Y-%m-%d}: dropped for no variation: {no_variation}")

    for col in table.columns:
        table[col] = (table[col] - table[col].mean()) / table[col].std()

    return table


def clean_all_snapshots(quarterly_data: pd.DataFrame) -> pd.DataFrame:
    """Clean each quarter separately and combine the results"""
    for col in ["quarter_end", "ticker", "field", "value"]:
        if col not in quarterly_data.columns:
            raise ValueError(f"quarterly_data is missing column: {col}")

    quarter_ends = sorted(quarterly_data["quarter_end"].unique())
    if len(quarter_ends) == 0:
        raise ValueError("quarterly_data is empty")

    cleaned = []
    for quarter_end in quarter_ends:
        table = clean_snapshot(quarterly_data, quarter_end)
        print(f"{pd.Timestamp(quarter_end):%Y-%m-%d}: {table.shape[0]} stocks, {table.shape[1]} fields")
        table = table.reset_index()
        table.insert(0, "quarter_end", quarter_end)
        cleaned.append(table)

    result = pd.concat(cleaned, ignore_index=True)
    result["quarter_end"] = result["quarter_end"].astype("datetime64[us]")
    result["ticker"] = result["ticker"].astype("str")
    for col in result.columns[2:]:
        result[col] = result[col].astype("float64")

    result = result.sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
    return result


def pca_characteristics(characteristic_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                                        pd.DataFrame]:
    """Run PCA on one quarter and return its components and scores"""
    if characteristic_matrix.empty:
        raise ValueError("characteristic_matrix is empty")
    if characteristic_matrix.shape[1] < 2:
        raise ValueError("characteristic_matrix needs at least two characteristics")

    values = characteristic_matrix.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("characteristic_matrix has missing or infinite values")

    corr = np.corrcoef(values, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("the correlation matrix has non-finite values")

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for i in range(eigenvectors.shape[1]):
        if eigenvectors[:, i].sum() < 0:
            eigenvectors[:, i] = -eigenvectors[:, i]

    cumulative = (eigenvalues / eigenvalues.sum()).cumsum()
    m = 1
    while cumulative[m - 1] < config.VAR_EXPLAINED_TARGET and m < len(eigenvalues):
        m += 1
    m = min(m, max(config.N_COMPONENTS_CANDIDATES))

    scores = values @ eigenvectors[:, :m]

    columns = []
    for i in range(m):
        columns.append(f"pc_{i + 1}")
    score_table = pd.DataFrame(scores, index=characteristic_matrix.index, columns=columns)

    return eigenvalues, eigenvectors, score_table


def name_components(eigenvectors: np.ndarray, characteristic_names: list[str]) -> pd.DataFrame:
    """List the five strongest loadings for the first three components"""
    eigenvectors = np.asarray(eigenvectors)
    if eigenvectors.ndim != 2:
        raise ValueError("eigenvectors must be a 2-D array")
    if len(characteristic_names) != eigenvectors.shape[0]:
        raise ValueError("characteristic_names must match the eigenvector rows")

    rows = []
    for i in range(min(3, eigenvectors.shape[1])):
        loadings = pd.Series(eigenvectors[:, i], index=characteristic_names)
        strongest = loadings.abs().sort_values(ascending=False).head(5)

        rank = 1
        for name in strongest.index:
            rows.append({"component": f"pc_{i + 1}", "rank": rank,"characteristic": name, 
                         "loading": loadings[name],"proposed_name": ""})
            rank += 1

    result = pd.DataFrame(rows, columns=["component", "rank", "characteristic","loading", 
                                         "proposed_name"])
    return result


def run_pca_for_all_quarters(clean_data: pd.DataFrame
                             ) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Run PCA for every quarter and collect the scores and summaries"""
    for col in ["quarter_end", "ticker"]:
        if col not in clean_data.columns:
            raise ValueError(f"clean_data is missing column: {col}")

    quarter_ends = sorted(clean_data["quarter_end"].unique())
    if len(quarter_ends) == 0:
        raise ValueError("clean_data is empty")

    scores_by_quarter = {}
    summaries = []
    for quarter_end in quarter_ends:
        quarter_end = pd.Timestamp(quarter_end)

        snapshot = clean_data[clean_data["quarter_end"] == quarter_end]
        if snapshot["ticker"].duplicated().any():
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} has duplicate tickers")

        matrix = snapshot.set_index("ticker").drop(columns="quarter_end")
        matrix = matrix.dropna(axis=1, how="all")

        eigenvalues, eigenvectors, scores = pca_characteristics(matrix)
        scores_by_quarter[quarter_end] = scores

        summary = name_components(eigenvectors, list(matrix.columns))
        summary.insert(0, "quarter_end", quarter_end)
        summaries.append(summary)

    result = pd.concat(summaries, ignore_index=True)
    return scores_by_quarter, result


def create_clusters_and_pairs(scores_by_quarter: dict[pd.Timestamp, pd.DataFrame],
                              trading_calendar: pd.DatetimeIndex
                              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create Track B clusters, stability records and pairs from the scores"""
    if not scores_by_quarter:
        raise ValueError("scores_by_quarter is empty")
    if not isinstance(trading_calendar, pd.DatetimeIndex):
        raise TypeError("trading_calendar must be a DatetimeIndex")

    windows = {}
    for quarter_end in sorted(scores_by_quarter):
        on_or_before = trading_calendar[trading_calendar <= quarter_end]
        if len(on_or_before) == 0:
            raise ValueError(f"no trading day on or before {quarter_end}")
        windows[on_or_before[-1]] = scores_by_quarter[quarter_end]

    if len(windows) != len(scores_by_quarter):
        raise ValueError("two quarter ends snapped to the same trading day")

    labels, stability = cluster_all_windows(windows, config.K_RANGE["b"])
    pairs = build_pairs(labels, windows, "track_b", trading_calendar)

    labels["window_end"] = labels["window_end"].astype("datetime64[us]")
    labels["ticker"] = labels["ticker"].astype("str")
    labels["cluster_id"] = labels["cluster_id"].astype("int64")

    stability["window_end"] = stability["window_end"].astype("datetime64[us]")
    stability["pair_id"] = stability["pair_id"].astype("str")
    stability["co_clustered"] = stability["co_clustered"].astype("bool")

    for col in ["pair_id", "stock_a", "stock_b", "group_id", "source"]:
        pairs[col] = pairs[col].astype("str")
    pairs["active_from"] = pairs["active_from"].astype("datetime64[us]")
    pairs["active_to"] = pairs["active_to"].astype("datetime64[us]")

    return labels, stability, pairs


def check_characteristics_data(raw_data: pd.DataFrame, quarterly_data: pd.DataFrame,
                               raw_table: pd.DataFrame,clean_table: pd.DataFrame) -> None:
    """Check that the raw, quarterly and cleaned data is valid"""
    for col in ["date", "ticker", "field", "value"]:
        if col not in raw_data.columns:
            raise ValueError(f"raw_data is missing column: {col}")
    for col in ["quarter_end", "ticker", "field", "value"]:
        if col not in quarterly_data.columns:
            raise ValueError(f"quarterly_data is missing column: {col}")
    for table, name in [(raw_table, "raw_table"), (clean_table, "clean_table")]:
        for col in ["quarter_end", "ticker"]:
            if col not in table.columns:
                raise ValueError(f"{name} is missing column: {col}")

    if raw_data["date"].isna().any():
        raise ValueError("raw_data has missing dates")
    if not raw_data["date"].is_monotonic_increasing:
        raise ValueError("raw_data dates are not sorted")
    if quarterly_data["quarter_end"].isna().any():
        raise ValueError("quarterly_data has missing quarter ends")
    if not quarterly_data["quarter_end"].is_monotonic_increasing:
        raise ValueError("quarterly_data quarter ends are not sorted")

    tickers = set(raw_data["ticker"])
    if len(tickers) != config.N_TICKERS:
        raise ValueError(f"expected {config.N_TICKERS} tickers, found {len(tickers)}")
    for table, name in [(quarterly_data, "quarterly_data"), (raw_table, "raw_table"), 
                        (clean_table, "clean_table")]:
        if set(table["ticker"]) != tickers:
            raise ValueError(f"{name} does not have the same tickers as raw_data")

    if quarterly_data.duplicated(["quarter_end", "ticker", "field"]).any():
        raise ValueError("quarterly_data has duplicate quarter, ticker and field rows")

    for table, name in [(raw_data, "raw_data"), (quarterly_data, "quarterly_data")]:
        for field in table["field"].unique():
            if field not in EXPECTED_FIELDS:
                raise ValueError(f"{name} has an unexpected field: {field}")
    allowed = set(EXPECTED_FIELDS) - {"buy_recommendations", "sell_recommendations"}
    allowed.add("sentiment")
    for col in clean_table.columns[2:]:
        if col not in allowed:
            raise ValueError(f"clean_table has an unexpected column: {col}")

    if raw_data["value"].dtype != "float64":
        raise ValueError("raw_data value column must be float64")
    if quarterly_data["value"].dtype != "float64":
        raise ValueError("quarterly_data value column must be float64")
    for table, name in [(raw_table, "raw_table"), (clean_table, "clean_table")]:
        for col in table.columns[2:]:
            if table[col].dtype != "float64":
                raise ValueError(f"{name} column {col} must be float64")

    print("check_characteristics_data coverage:")
    for field, group in raw_data.groupby("field"):
        print(f"{field}: {group['value'].notna().mean():.0%}")

    first = raw_data["date"].min()
    last = raw_data["date"].max()
    if first.year > 2015 or last.year < 2024:
        raise ValueError(f"raw_data spans {first.date()} to {last.date()}, expected about 2015-2024")

    for quarter_end, group in clean_table.groupby("quarter_end"):
        fields = group.drop(columns=["quarter_end", "ticker"]).dropna(axis=1, how="all")
        if not np.isfinite(fields.to_numpy()).all():
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} has missing or infinite values")
        if not np.allclose(fields.mean(), 0, atol=1e-8):
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} columns are not centered at zero")
        if not np.allclose(fields.std(), 1, atol=1e-6):
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} columns do not have std one")
        if len(group) != config.N_TICKERS:
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} has {len(group)} stocks, expected {config.N_TICKERS}")
        if fields.shape[1] < 5:
            raise ValueError(f"quarter {quarter_end:%Y-%m-%d} has only {fields.shape[1]} usable fields")


def run_track_b(raw_directory: str | Path = "data/raw/characteristics") -> None:
    """Build and save all Track B characteristic, cluster and pair files"""
    raw_data = load_raw(raw_directory)
    quarterly_data = convert_to_quarterly(raw_data)
    raw_table = make_raw_characteristics_table(quarterly_data)
    clean_table = clean_all_snapshots(quarterly_data)

    check_characteristics_data(raw_data, quarterly_data, raw_table, clean_table)

    write_parquet(raw_table, config.DATA_DIR / "raw" / "characteristics_raw.parquet",
                  "characteristics_raw")
    write_parquet(clean_table, config.DATA_DIR / "processed" / "characteristics_clean.parquet",
                  "characteristics_clean")

    scores_by_quarter, components = run_pca_for_all_quarters(clean_table)

    components_file = config.RESULTS_DIR / "tables" / "track_b_components.csv"
    components_file.parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(components_file, index=False)

    calendar = read_parquet(config.DATA_DIR / "processed" / "returns.parquet").index

    labels, stability, pairs = create_clusters_and_pairs(scores_by_quarter, calendar)

    write_parquet(labels, config.DATA_DIR / "clusters" / "labels_b.parquet", "labels")
    write_parquet(stability, config.DATA_DIR / "clusters" / "stability_b.parquet", "stability")
    write_validated_csv(pairs, config.DATA_DIR / "pairs" / "pairs_b.csv", "pairs")

    print(f"track b: {labels['window_end'].nunique()} windows, {len(pairs)} pair rows, {pairs['pair_id'].nunique()} unique pairs")


if __name__ == "__main__":
    run_track_b()
