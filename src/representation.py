"""Market structure from returns: rolling PCA, out-of-sample residuals,
clustering, pair building, and spread/z-scores — the whole
returns -> zscores chain in one file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

from src import config
from src.contracts import read_parquet, seed_everything, validate_artifact, write_parquet, write_validated_csv


def pca_one_window(window_returns: pd.DataFrame) -> tuple[np.ndarray, int, float, np.ndarray]:
    """PCA of one window of returns (date x ticker).

    window_returns must contain only PAST days relative to any day the
    outputs will be applied to — never that day itself (that would leak
    it into its own factors).

    Returns (weights, n_components, cum_var, corr):
      weights: 40 x 5 eigenportfolio columns, biggest first, each column
               sign-fixed so its weights sum positive (consistent
               orientation across windows).
      n_components: smallest m in (3, 4, 5) explaining >= 60% of
               variance, else 5.
      cum_var: variance fraction those m components explain.
      corr: the window's 40 x 40 correlation matrix (Track C option).
    """
    n_component_candidates = config.N_COMPONENTS_CANDIDATES

    corr_matrix = np.corrcoef(window_returns.values, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort in biggest to smallest eigenvalue order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    columns_to_keep = max(n_component_candidates)  # always carry 5 columns 
    top_component_weights = eigenvectors[:, :columns_to_keep].copy()

    # fliping columns whose weights sum negative
    for column in range(columns_to_keep):
        current_column_weights  = top_component_weights[:, column]
        if current_column_weights.sum() < 0:
            top_component_weights[:, column] = -top_component_weights[:, column]

    # cumulative fraction of total variance explained
    cum_vars = np.cumsum(eigenvalues) / eigenvalues.sum()

    # smallest candidate m that is bigger than the target
    n_components = max(n_component_candidates)
    for m in n_component_candidates:
        if cum_vars[m - 1] >= config.VAR_EXPLAINED_TARGET:
            n_components = m
            break

    cum_var = float(cum_vars[n_components - 1])
    return top_component_weights, n_components, cum_var, corr_matrix


def run_rolling_pca(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Roll a 252-day window over all days; PCA + one regression per day.

    Returns (factors, meta, residuals, loadings):
      factors:   date x pc_1..pc_5, daily factor returns
      meta:      date x (n_components, cum_var_explained)
      residuals: date x ticker — return minus what the factors explain
      loadings:  long table, written every 21st day: loading + beta per
                 stock per component (always all 5, so the clustering
                 vectors have equal length)
    """
    n_days = len(returns.index)
    tickers = list(returns.columns)
    window_size = config.PCA_WINDOW

    dates = []
    factor_rows = []
    meta_rows = []
    residual_rows = []
    loading_rows = []

    for today in range(window_size, n_days):
        date = returns.index[today]
        # the past 252 days; today itself excluded (no peeking)
        start_day = today - window_size
        window_returns = returns.iloc[start_day : today]
        weights, n_components, cum_var, corr = pca_one_window(window_returns)

        selected_weights = weights[:, :n_components]

        window_factors = np.dot(window_returns.values, selected_weights)

        X_train = window_factors
        y_train = window_returns.values
        regression = LinearRegression()
        regression.fit(X_train, y_train)

        todays_returns = returns.iloc[today].values
        todays_factors_selected = np.dot(todays_returns, selected_weights)
        todays_factors_all = np.dot(todays_returns, weights)  # stored: always 5 columns

        today_factors_2d = todays_factors_selected.reshape(1, -1)
        predictions = regression.predict(today_factors_2d)
        explained_today = predictions[0]

        residual_rows.append(todays_returns - explained_today)
        dates.append(date)
        factor_rows.append(todays_factors_all)
        meta_rows.append({"n_components": n_components, "cum_var_explained": cum_var})

        # every 21st day, append loadings rows for clustering
        days_into_loop = today - window_size
        if days_into_loop % config.RECLUSTER_EVERY == 0:
            # betas on ALL components
            window_factors_all = np.dot(window_returns.values, weights)
            regression_all = LinearRegression()
            regression_all.fit(window_factors_all, window_returns.values)
            betas_all = regression_all.coef_
            for component in range(max(config.N_COMPONENTS_CANDIDATES)):
                for stock in range(len(tickers)):
                    ticker  = tickers[stock]
                    loading = weights[stock, component]
                    beta = betas_all[stock, component]
                    loading_rows.append({"date": date,"ticker":ticker ,"component": component + 1,"loading": loading,"beta": beta})

    index = pd.DatetimeIndex(dates, name="date")
    factor_names = ["pc_1", "pc_2", "pc_3", "pc_4", "pc_5"]
    factors = pd.DataFrame(factor_rows, index=index, columns=factor_names)
    meta = pd.DataFrame(meta_rows, index=index)
    residuals = pd.DataFrame(residual_rows, index=index, columns=tickers)
    loadings = pd.DataFrame(loading_rows)
    return factors, meta, residuals, loadings


# ------------------------- shared clustering machinery (all tracks) -------

def fit_kmeans_select_k(X: np.ndarray, k_range: range, seed: int = config.SEED) -> tuple[np.ndarray, int, float]:
    """Try k-means for every k in k_range; keep the best silhouette score.

    X: one row per stock. Returns (labels, best_k, best_score).
    """
    best_labels = None
    best_k = None
    best_score = float("-inf")
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=config.KMEANS_N_INIT, random_state=seed)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue  # silhouette needs at least 2 clusters
        score = silhouette_score(X, labels)
        if score > best_score:
            best_labels = labels
            best_k = k
            best_score = score
    return best_labels, best_k, best_score


def pair_from_labels(labels: np.ndarray, tickers: list) -> set:
    """All (first, second) ticker pairs sharing a cluster, alphabetical.

    Cluster numbers mean nothing across windows (k-means relabels
    freely); "these two are together" is the only comparable fact.
    """
    pairs = set()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            if labels[i] == labels[j]:
                pairs.add(tuple(sorted([tickers[i], tickers[j]])))
    return pairs


def pair_stability_table(prev_pairs: set | None, curr_pairs: set, window_end) -> pd.DataFrame:
    """One row per current pair. co_clustered = also together last window."""
    rows = []
    for first, second in sorted(curr_pairs):
        together_before = False
        if prev_pairs and (first, second) in prev_pairs:
            together_before = True
        rows.append({"window_end": window_end, "pair_id": first + "__" + second, "co_clustered": together_before})
    return pd.DataFrame(rows)


def cluster_all_windows(feature_table_by_window: dict, k_range: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster every window, in date order. Returns (labels, stability).

    feature_table_by_window: {window_end -> DataFrame, one row per
    ticker (the index), one column per feature}. Track a passes beta
    tables; track b passes characteristics tables. The caller
    standardizes the features first if they need it.
    """
    label_rows = []
    stability_tables = []    # one small table per window; glued at the end
    prev_pairs = None        # None = there is no previous window yet


    for window_end_date in sorted(feature_table_by_window):
        feature_table = feature_table_by_window[window_end_date]
        tickers = list(feature_table.index)

        labels, best_k, best_score = fit_kmeans_select_k(feature_table.values, k_range)

        # record one labels row per stock
        for stock in range(len(tickers)):
            ticker = tickers[stock]
            cluster_id = int(labels[stock])
            row = {"window_end": window_end_date, "ticker": ticker, "cluster_id": cluster_id}
            label_rows.append(row)

        # this window's pairs, compared against the previous window's pair
        curr_pairs = pair_from_labels(labels, tickers)
        stab_table = pair_stability_table(prev_pairs, curr_pairs, window_end_date)
        stability_tables.append(stab_table)
        prev_pairs = curr_pairs

    labels_table = pd.DataFrame(label_rows)
    stability_table = pd.concat(stability_tables, ignore_index=True)
    return labels_table, stability_table


# ------------------------- pair builder (all tracks) ----------------------

def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Euclidean distance between two feature rows."""
    return float(np.sqrt(((point_a - point_b) ** 2).sum()))


def split_large_cluster(members: list, features: pd.DataFrame) -> list:
    """Split a cluster of 5+ members into subgroups of 2-3.

    Greedy nearest-neighbour on the members' feature rows (Euclidean):
    repeatedly pull out the closest remaining couple; a final odd
    leftover joins the couple holding its nearest member, making one
    group of 3. Returns a list of subgroups (each 2-3 tickers).
    """
    remaining = list(members)
    subgroups = []

    # pull out the closest remaining couple until 0 or 1 member is left
    while len(remaining) >= 2:
        closest_pair = None
        closest_distance = float("inf")
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                point_i = features.loc[remaining[i]].values
                point_j = features.loc[remaining[j]].values
                distance = _distance(point_i, point_j)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pair = [remaining[i], remaining[j]]
        subgroups.append(closest_pair)
        remaining.remove(closest_pair[0])
        remaining.remove(closest_pair[1])

    # last_member joins the subgroup holding its nearest member
    if len(remaining) == 1 and len(subgroups) > 0:
        last_member = remaining[0]
        last_member_row = features.loc[last_member].values
        best_group = None
        best_distance = float("inf")
        for subgroup in subgroups:
            for member in subgroup:
                distance = _distance(last_member_row, features.loc[member].values)
                if distance < best_distance:
                    best_distance = distance
                    best_group = subgroup
        best_group.append(last_member)
    return subgroups


def build_pairs(labels: pd.DataFrame, feature_table_by_window: dict, source: str,
                calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Turn every window's clusters into the tradeable pairs table.

    Per window, per cluster: 1 member -> dropped; 2-4 members -> every
    pair; 5+ members -> split_large_cluster first, then every pair
    inside each subgroup. Every pair gets:
      active_from = first trading day AFTER window_end
      active_to   = the next window's window_end (or the last calendar
                    day for the final window)
    so consecutive windows tile with no gap — the spread builder and
    the trigger detector both rely on that.
    Returns the "pairs" schema table (caller writes it with
    write_validated_csv).
    """
    window_end_dates = sorted(labels["window_end"].unique())
    pair_rows = []

    for curr_window in range(len(window_end_dates)):
        window_end_date = window_end_dates[curr_window]

        dates_after_curr_window = calendar[calendar > window_end_date]
        if len(dates_after_curr_window) == 0:
            continue

        start_date = dates_after_curr_window[0]
        end_date = calendar[-1]
        if len(window_end_dates) > curr_window + 1:
            end_date = window_end_dates[curr_window + 1]

        curr_window_mask = labels["window_end"] == window_end_date
        window_labels = labels[curr_window_mask]
        features = feature_table_by_window[window_end_date]



        for cluster_id in sorted(window_labels["cluster_id"].unique()):
            belongs_to_cluster = window_labels["cluster_id"] == cluster_id
            in_cluster = window_labels[belongs_to_cluster]
            members = sorted(in_cluster["ticker"])

            # pairs can't be created
            if len(members) < 2:
                continue  

            # small cluster stays intact
            if len(members) <= config.MAX_WHOLE_CLUSTER:
                subgroups = [members]  
            else:
                subgroups = split_large_cluster(members, features)

            for subgroup_i in range(len(subgroups)):
                subgroup = sorted(subgroups[subgroup_i])
                group_id = f"{window_end_date:%Y%m%d}_{cluster_id}_{subgroup_i}"
                for i in range(len(subgroup)):
                    for j in range(i + 1, len(subgroup)):
                        stock_a = subgroup[i]
                        stock_b = subgroup[j]
                        pair_id = stock_a + "__" + stock_b
                        pair_rows.append({"pair_id": pair_id,"stock_a": stock_a, "stock_b": stock_b,"group_id": group_id,"source": source,"active_from": start_date,"active_to": end_date,})

                        
    return pd.DataFrame(pair_rows)


# ------------------------- spreads and z-scores (all tracks) --------------

def build_spreads(residuals: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spread and z-score day-series for every pair. Returns (spreads, zscores).

    Per pair_id: group its rows into RUNS — stretches of consecutive
    active windows that tile without a gap. Within a run:
      spread = running sum of (residual_first - residual_second),
               starting 60 trading days BEFORE the run's active_from
               (backward-looking burn-in: already-written history, no
               leak) so the z-score is valid from the first active day
      z = (spread - rolling 60d mean) / rolling 60d std
    Output frames are date x pair_id; values only on days inside active
    windows — burn-in days stay internal and are never written.
    """
    dates = residuals.index
    spread_series = {}   # pair_id -> its finished day-series
    z_series = {}

    for pair_id in sorted(pairs["pair_id"].unique()):
        curr_pair_mask = pairs["pair_id"] == pair_id
        pair = pairs[curr_pair_mask].sort_values("active_from")
        stock_a = pair.iloc[0]["stock_a"]
        stock_b = pair.iloc[0]["stock_b"]
        active_from_list = list(pair["active_from"])
        active_to_list = list(pair["active_to"])

        runs = []      
        run_start = active_from_list[0]
        run_end = active_to_list[0]
        for w in range(1, len(active_from_list)):
            next_window_start = active_from_list[w]
            after_run_end = dates > run_end
            before_next_window = dates < next_window_start
            days_between_window = dates[after_run_end & before_next_window]

            if len(days_between_window) > 0: 
                runs.append((run_start, run_end))
                run_start = active_from_list[w]
            run_end = active_to_list[w]

        runs.append((run_start, run_end))

        # one spread and one z series per run
        spread_pieces = []
        z_pieces = []
        for run_start, run_end in runs:
            start_position = dates.get_loc(run_start)
            end_position = dates.get_loc(run_end)
            calculation_start_position = max(0, start_position - config.SPREAD_WARMUP_DAYS)
            trade_run_days = dates[calculation_start_position : end_position + 1]

            residual_diff = residuals.loc[trade_run_days, stock_a] - residuals.loc[trade_run_days, stock_b]
            spread = residual_diff.cumsum()
            rolling_mean = spread.rolling(config.Z_WINDOW).mean()
            rolling_std = spread.rolling(config.Z_WINDOW).std()
            z = (spread - rolling_mean) / rolling_std

            active_days = dates[start_position : end_position + 1]
            spread_pieces.append(spread.loc[active_days])
            z_pieces.append(z.loc[active_days])

        spread_series[pair_id] = pd.concat(spread_pieces)
        z_series[pair_id] = pd.concat(z_pieces)

    spreads = pd.DataFrame(spread_series, index=dates)
    zscores = pd.DataFrame(z_series, index=dates)
    return spreads, zscores


# ------------------------- the track-a assembly line ----------------------

def main() -> None:
    """Build every track-a artifact: python -m src.representation --track a

    The whole chain, each output feeding the next, every artifact
    written through the schema-checked writers:
      returns -> run_rolling_pca -> factors, meta, residuals, loadings
      loadings -> pivot + standardize per window -> features_by_window
      features_by_window -> cluster_all_windows -> labels, stability
      labels + features -> build_pairs -> pairs_a.csv
      residuals + pairs -> build_spreads -> spreads_a, zscores_a

    --track b runs only the last step: P2's characteristics flow already
    built pairs_b.csv (through the shared clustering + pair-builder), so
    here we read pairs_b.csv + residuals_a and write spreads_b, zscores_b.
    Both tracks trade the same residual spreads — only the grouping differs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="a")
    args = parser.parse_args()
    assert args.track in ("a", "b"), f"unknown track '{args.track}'"

    seed_everything()

    if args.track == "b":
        residuals_file = Path("data/processed/residuals_a.parquet")
        pairs_file = Path("data/pairs/pairs_b.csv")
        assert residuals_file.exists(), "residuals_a.parquet missing: run 'make tracka' first"
        assert pairs_file.exists(), "pairs_b.csv missing: run 'python -m src.characteristics' first"

        residuals = read_parquet(residuals_file)
        pairs = pd.read_csv(pairs_file, parse_dates=["active_from", "active_to"])
        pairs["active_from"] = pairs["active_from"].astype("datetime64[us]")
        pairs["active_to"] = pairs["active_to"].astype("datetime64[us]")
        validate_artifact(pairs, "pairs")

        spreads, zscores = build_spreads(residuals, pairs)
        write_parquet(spreads, "data/spreads/spreads_b.parquet", "spreads")
        write_parquet(zscores, "data/spreads/zscores_b.parquet", "zscores")
        print(f"representation: track b, {zscores.shape[1]} pairs with z-scores over {zscores.shape[0]} days")
        return

    # FOR TRACK A

    returns = pd.read_parquet("data/processed/returns.parquet")

    # 1. rolling PCA
    factors, meta, residuals, loadings = run_rolling_pca(returns)

    # 2. one feature table per window: stocks x betas
    feature_table_by_window = {}
    for window_end, window_loadings in loadings.groupby("date"):
        beta_table = window_loadings.pivot(index="ticker", columns="component", values="beta")
        standardized_beta = (beta_table - beta_table.mean()) / beta_table.std()
        feature_table_by_window[window_end] = standardized_beta

    # 3. cluster every window
    labels, stability = cluster_all_windows(feature_table_by_window, config.K_RANGE["a"])

    # 4. build pairs from clusters
    pairs = build_pairs(labels, feature_table_by_window, "track_a", returns.index)

    # 5. calculate spreads and z-scores from pairs and residuals
    spreads, zscores = build_spreads(residuals, pairs)

    write_parquet(factors, "data/processed/factors_a.parquet", "factors_a")
    write_parquet(meta, "data/processed/pca_meta.parquet", "pca_meta")
    write_parquet(residuals, "data/processed/residuals_a.parquet", "residuals_a")
    write_parquet(loadings, "data/processed/loadings_a.parquet", "loadings_a")
    write_parquet(labels, "data/clusters/labels_a.parquet", "labels")
    write_parquet(stability, "data/clusters/stability_a.parquet", "stability")
    write_validated_csv(pairs, "data/pairs/pairs_a.csv", "pairs")
    write_parquet(spreads, "data/spreads/spreads_a.parquet", "spreads")
    write_parquet(zscores, "data/spreads/zscores_a.parquet", "zscores")
    print(f"representation: {labels['window_end'].nunique()} windows, {len(pairs)} pair rows, {zscores.shape[1]} pairs with z-scores over {zscores.shape[0]} days")


if __name__ == "__main__":
    main()
