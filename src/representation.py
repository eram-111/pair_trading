"""Market structure from returns: rolling PCA, out-of-sample residuals,
clustering, pair building, and spread/z-scores — the whole
returns -> zscores chain in one file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

from src import config


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
