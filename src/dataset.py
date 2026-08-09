"""Build trigger rows, labels, splits and model features"""

import argparse

import pandas as pd

from src import config
from src.contracts import SCHEMAS, read_parquet, write_parquet


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Check that a table has all the columns it needs"""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"{name} is missing column: {col}")


def check_date_index(index: pd.DatetimeIndex, name: str) -> None:
    """Check that a date index is sorted, unique and not empty"""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} needs a DatetimeIndex")
    if index.empty:
        raise ValueError(f"{name} is empty")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be sorted")
    if not index.is_unique:
        raise ValueError(f"{name} has duplicate dates")


def clean_triggers(triggers: pd.DataFrame) -> pd.DataFrame:
    """Copy the trigger table and parse its dates"""
    df = triggers.copy()
    df["trigger_date"] = pd.to_datetime(df["trigger_date"])
    if df["trigger_date"].isna().any():
        raise ValueError("trigger_date has missing values")
    return df


def get_pair_legs(pairs: pd.DataFrame, pair_id: str) -> tuple[str, str]:
    """Look up the two stocks that make up a pair"""
    legs = pairs[pairs["pair_id"] == pair_id]
    if legs.empty:
        raise ValueError(f"pairs is missing the pair: {pair_id}")
    a_vals = legs["stock_a"].unique()
    b_vals = legs["stock_b"].unique()
    if len(a_vals) != 1 or len(b_vals) != 1:
        raise ValueError(f"pair {pair_id} has inconsistent stock legs")
    return a_vals[0], b_vals[0]


def combine_active_windows(pairs: pd.DataFrame,calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Merge pair windows that touch on back to back trading days into runs"""
    needed = ["pair_id", "source", "active_from", "active_to"]
    result_cols = ["pair_id", "source", "run_start", "run_end"]

    if not isinstance(pairs, pd.DataFrame):
        raise TypeError("pairs must be a DataFrame")
    require_columns(pairs, needed, "pairs")
    check_date_index(calendar, "calendar")

    df = pairs.copy()
    df["active_from"] = pd.to_datetime(df["active_from"])
    df["active_to"] = pd.to_datetime(df["active_to"])

    if df.empty:
        return pd.DataFrame(columns=result_cols)

    for col in needed:
        if df[col].isna().any():
            raise ValueError(f"pairs has missing values in {col}")
    if (df["active_from"] > df["active_to"]).any():
        raise ValueError("active_from must not be after active_to")
    for col in ["active_from", "active_to"]:
        if not df[col].isin(calendar).all():
            raise ValueError(f"pairs has a {col} date that is not in the calendar")

    df = df.sort_values(["pair_id", "active_from"])

    result_rows = []
    for pair_id, group in df.groupby("pair_id"):
        group = group.sort_values("active_from")
        if group["source"].nunique() != 1:
            raise ValueError(f"pair {pair_id} has more than one source")

        source = group.iloc[0]["source"]
        run_start = group.iloc[0]["active_from"]
        run_end = group.iloc[0]["active_to"]

        for _, row in group.iloc[1:].iterrows():
            next_start = row["active_from"]
            next_end = row["active_to"]
            if next_start <= run_end:
                raise ValueError(f"pair {pair_id} has overlapping active windows")

            end_pos = calendar.get_loc(run_end)
            next_day = calendar[end_pos + 1]

            if next_start == next_day:
                run_end = next_end
            else:
                result_rows.append({"pair_id": pair_id, "source": source,"run_start": run_start,
                                    "run_end": run_end})
                run_start = next_start
                run_end = next_end

        result_rows.append({"pair_id": pair_id, "source": source, "run_start": run_start, 
                            "run_end": run_end})

    result = pd.DataFrame(result_rows, columns=result_cols)
    result = result.sort_values(["pair_id", "run_start"]).reset_index(drop=True)
    return result


def label_one_trigger(pair_zscores: pd.Series,trigger_date: pd.Timestamp,
                      run_end: pd.Timestamp,horizon: int,reversion_frac: float
                      ) -> tuple[int, pd.Timestamp] | None:
    """Label one trigger, or return None when the future window is incomplete"""
    if not isinstance(pair_zscores, pd.Series):
        raise TypeError("pair_zscores must be a Series")
    check_date_index(pair_zscores.index, "pair_zscores")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if reversion_frac <= 0 or reversion_frac > 1:
        raise ValueError("reversion_frac must be in (0, 1]")

    trigger_date = pd.Timestamp(trigger_date)
    run_end = pd.Timestamp(run_end)
    if trigger_date > run_end:
        raise ValueError("trigger_date is after run_end")
    if trigger_date not in pair_zscores.index:
        raise ValueError("trigger_date is not in the z-score data")

    pos = pair_zscores.index.get_loc(trigger_date)
    trigger_z = pair_zscores.iloc[pos]
    if pd.isna(trigger_z):
        raise ValueError("the z-score on the trigger date is missing")

    end_pos = pos + horizon
    if end_pos >= len(pair_zscores):
        return None
    end_date = pair_zscores.index[end_pos]
    if end_date > run_end:
        return None

    future = pair_zscores.iloc[pos + 1 : end_pos + 1]
    if future.isna().any():
        return None

    target = reversion_frac * abs(trigger_z)
    reverted = (future.abs() <= target).any()
    label = int(reverted)
    return label, pd.Timestamp(end_date)


def detect_triggers(zscores: pd.DataFrame,pairs: pd.DataFrame,z_entry: float = config.TRIGGER_Z,
                    horizon: int = config.LABEL_HORIZON,
                    reversion_frac: float = config.REVERSION_FRACTION) -> pd.DataFrame:
    """Find fresh z-score crossings and label them"""
    result_cols = ["trigger_id", "pair_id", "source", "trigger_date","z_trigger", "label",
                   "horizon_end_date"]

    if not isinstance(zscores, pd.DataFrame):
        raise TypeError("zscores must be a DataFrame")
    if not isinstance(pairs, pd.DataFrame):
        raise TypeError("pairs must be a DataFrame")

    calendar = zscores.index
    check_date_index(calendar, "zscores")
    if not zscores.columns.is_unique:
        raise ValueError("zscores has duplicate pair columns")
    if z_entry <= 0:
        raise ValueError("z_entry must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if reversion_frac <= 0 or reversion_frac > 1:
        raise ValueError("reversion_frac must be in (0, 1]")

    runs = combine_active_windows(pairs, calendar)
    for pair_id in runs["pair_id"].unique():
        if pair_id not in zscores.columns:
            raise ValueError(f"zscores is missing pair column: {pair_id}")

    found = []
    dropped = 0

    for _, run in runs.iterrows():
        pair_id = run["pair_id"]
        source = run["source"]
        run_end = run["run_end"]
        run_z = zscores[pair_id].loc[run["run_start"]:run_end]
        if len(run_z) < 2:
            continue

        armed = True
        last_horizon_end = None
        went_below = False

        for pos in range(1, len(run_z)):
            date = run_z.index[pos]
            prev_z = run_z.iloc[pos - 1]
            cur_z = run_z.iloc[pos]

            if not armed:
                if date >= last_horizon_end and pd.notna(cur_z) and abs(cur_z) < z_entry:
                    went_below = True
                if date > last_horizon_end and went_below:
                    armed = True
            if not armed:
                continue

            if pd.isna(prev_z) or pd.isna(cur_z):
                continue
            crossed = abs(prev_z) < z_entry and abs(cur_z) >= z_entry
            if not crossed:
                continue

            result = label_one_trigger(run_z, date, run_end, horizon, reversion_frac)
            armed = False
            went_below = False

            if result is None:
                dropped += 1
                end_pos = pos + horizon
                if end_pos >= len(run_z):
                    break
                last_horizon_end = run_z.index[end_pos]
                continue

            label, horizon_end = result
            last_horizon_end = horizon_end
            found.append({"trigger_id": f"{pair_id}__{date:%Y%m%d}","pair_id": pair_id,
                          "source": source,"trigger_date": date,"z_trigger": float(cur_z),
                          "label": label,"horizon_end_date": horizon_end})

    triggers = pd.DataFrame(found, columns=result_cols)
    triggers = triggers.astype({"trigger_id": "str", "pair_id": "str", "source": "str",
                                "z_trigger": "float64", "label": "int8"})
    triggers["trigger_date"] = pd.to_datetime(triggers["trigger_date"])
    triggers["horizon_end_date"] = pd.to_datetime(triggers["horizon_end_date"])
    triggers = triggers.sort_values(["trigger_date", "pair_id"]).reset_index(drop=True)

    print(f"detect_triggers kept {len(triggers)} and dropped {dropped} trigger(s)")
    return triggers


def assign_split(trigger_dates: pd.Series,horizon_end_dates: pd.Series,
                 calendar: pd.DatetimeIndex) -> pd.Series:
    """Give each trigger a split: train, val, test, purged or embargo"""
    if not isinstance(trigger_dates, pd.Series):
        raise TypeError("trigger_dates must be a Series")
    if not isinstance(horizon_end_dates, pd.Series):
        raise TypeError("horizon_end_dates must be a Series")
    if not trigger_dates.index.equals(horizon_end_dates.index):
        raise ValueError("trigger_dates and horizon_end_dates need the same index")
    check_date_index(calendar, "calendar")

    trigger_dates = pd.to_datetime(trigger_dates)
    horizon_end_dates = pd.to_datetime(horizon_end_dates)

    if trigger_dates.isna().any():
        raise ValueError("trigger_dates has missing values")
    if horizon_end_dates.isna().any():
        raise ValueError("horizon_end_dates has missing values")
    if (horizon_end_dates <= trigger_dates).any():
        raise ValueError("every horizon_end_date must be after its trigger_date")
    if not trigger_dates.isin(calendar).all():
        raise ValueError("trigger_dates has a date that is not in the calendar")
    if not horizon_end_dates.isin(calendar).all():
        raise ValueError("horizon_end_dates has a date that is not in the calendar")

    train_start = pd.Timestamp(config.TRAIN_START)
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start = pd.Timestamp(config.VAL_START)
    val_end = pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)
    test_end = pd.Timestamp(config.TEST_END)

    split = pd.Series(index=trigger_dates.index, dtype="object", name="split")

    split.loc[trigger_dates.between(train_start, train_end)] = "train"
    split.loc[trigger_dates.between(val_start, val_end)] = "val"
    split.loc[trigger_dates.between(test_start, test_end)] = "test"

    if split.isna().any():
        raise ValueError("a trigger_date is outside the configured split periods")

    val_days = calendar[(calendar >= val_start) & (calendar <= val_end)]
    test_days = calendar[(calendar >= test_start) & (calendar <= test_end)]

    if len(val_days) > 0:
        split.loc[(split == "train") & (horizon_end_dates >= val_days[0])] = "purged"
    if len(test_days) > 0:
        split.loc[(split == "val") & (horizon_end_dates >= test_days[0])] = "purged"

    split.loc[trigger_dates.isin(val_days[: config.EMBARGO_DAYS])] = "embargo"
    split.loc[trigger_dates.isin(test_days[: config.EMBARGO_DAYS])] = "embargo"

    return split.astype("str")


def calculate_spread_volatility(triggers: pd.DataFrame,spreads: pd.DataFrame) -> pd.Series:
    """Calculate the 60 day volatility of daily spread changes at each trigger"""
    require_columns(triggers, ["pair_id", "trigger_date"], "triggers")
    check_date_index(spreads.index, "spreads")

    df = clean_triggers(triggers)

    diffs = spreads.diff()
    rolling_std = diffs.rolling(config.SPREAD_VOL_WINDOW, min_periods=30).std()

    result = pd.Series(index=triggers.index, dtype="float64", name="f_spread_vol_60d")

    for i, row in df.iterrows():
        pair_id = row["pair_id"]
        trigger_date = row["trigger_date"]
        if pair_id not in rolling_std.columns:
            raise ValueError(f"spreads is missing pair column: {pair_id}")
        if trigger_date not in rolling_std.index:
            raise ValueError(f"trigger date {trigger_date} is not in the spread calendar")
        result.loc[i] = rolling_std.loc[trigger_date, pair_id]

    return result


def calculate_residual_momentum(triggers: pd.DataFrame,residuals: pd.DataFrame,
                                pairs: pd.DataFrame) -> pd.Series:
    """Calculate the signed five day residual momentum at each trigger"""
    require_columns(triggers, ["pair_id", "trigger_date", "z_trigger"], "triggers")
    require_columns(pairs, ["pair_id", "stock_a", "stock_b"], "pairs")
    check_date_index(residuals.index, "residuals")

    df = clean_triggers(triggers)

    result = pd.Series(index=triggers.index, dtype="float64", name="f_resid_mom_5d")

    for i, row in df.iterrows():
        pair_id = row["pair_id"]
        trigger_date = row["trigger_date"]
        z_trigger = row["z_trigger"]

        stock_a, stock_b = get_pair_legs(pairs, pair_id)
        if stock_a not in residuals.columns:
            raise ValueError(f"residuals is missing stock: {stock_a}")
        if stock_b not in residuals.columns:
            raise ValueError(f"residuals is missing stock: {stock_b}")
        if trigger_date not in residuals.index:
            raise ValueError(f"trigger date {trigger_date} is not in the residual calendar")
        if pd.isna(z_trigger) or z_trigger == 0:
            raise ValueError(f"pair {pair_id} has an invalid z_trigger")

        pos = residuals.index.get_loc(trigger_date)
        start_pos = pos - config.RESID_MOM_WINDOW + 1
        if start_pos < 0:
            continue

        window = residuals.iloc[start_pos : pos + 1]
        diff = window[stock_a] - window[stock_b]
        if diff.isna().any():
            continue

        sign = 1 if z_trigger > 0 else -1
        result.loc[i] = sign * diff.sum()

    return result


def calculate_market_volatility(triggers: pd.DataFrame,factors: pd.DataFrame) -> pd.Series:
    """Calculate the 20 day volatility of pc_1 at each trigger"""
    require_columns(triggers, ["trigger_date"], "triggers")
    require_columns(factors, ["pc_1"], "factors")
    check_date_index(factors.index, "factors")

    trigger_dates = pd.to_datetime(triggers["trigger_date"])
    if trigger_dates.isna().any():
        raise ValueError("trigger_date has missing values")

    rolling_std = factors["pc_1"].rolling(config.MKT_VOL_WINDOW, 
                                          min_periods=config.MKT_VOL_WINDOW).std()

    result = pd.Series(index=triggers.index, dtype="float64", name="f_mkt_vol_20d")

    for i, trigger_date in trigger_dates.items():
        if trigger_date not in factors.index:
            raise ValueError(f"trigger date {trigger_date} is not in the factor calendar")
        result.loc[i] = rolling_std.loc[trigger_date]

    return result


def calculate_relative_volume(triggers: pd.DataFrame,volume: pd.DataFrame,
                              pairs: pd.DataFrame) -> pd.Series:
    """Compare each pair's volume with its own 20 day average"""
    require_columns(triggers, ["pair_id", "trigger_date"], "triggers")
    require_columns(pairs, ["pair_id", "stock_a", "stock_b"], "pairs")
    check_date_index(volume.index, "volume")

    df = clean_triggers(triggers)

    rolling_mean = volume.rolling(config.REL_VOLUME_WINDOW, 
                                  min_periods=config.REL_VOLUME_WINDOW).mean()

    result = pd.Series(index=triggers.index, dtype="float64", name="f_rel_volume_20d")

    for i, row in df.iterrows():
        pair_id = row["pair_id"]
        trigger_date = row["trigger_date"]

        stock_a, stock_b = get_pair_legs(pairs, pair_id)
        if stock_a not in volume.columns:
            raise ValueError(f"volume is missing stock: {stock_a}")
        if stock_b not in volume.columns:
            raise ValueError(f"volume is missing stock: {stock_b}")
        if trigger_date not in volume.index:
            raise ValueError(f"trigger date {trigger_date} is not in the volume calendar")

        vol_a = volume.loc[trigger_date, stock_a]
        vol_b = volume.loc[trigger_date, stock_b]
        avg_a = rolling_mean.loc[trigger_date, stock_a]
        avg_b = rolling_mean.loc[trigger_date, stock_b]

        if pd.isna(vol_a) or pd.isna(vol_b):
            continue
        if pd.isna(avg_a) or pd.isna(avg_b):
            continue
        if avg_a <= 0 or avg_b <= 0:
            continue

        result.loc[i] = (vol_a / avg_a + vol_b / avg_b) / 2

    return result


def calculate_days_since_trigger(triggers: pd.DataFrame,calendar: pd.DatetimeIndex) -> pd.Series:
    """Count the trading days since the same pair's previous trigger"""
    require_columns(triggers, ["pair_id", "trigger_date"], "triggers")
    check_date_index(calendar, "calendar")

    df = clean_triggers(triggers)
    if df["pair_id"].isna().any():
        raise ValueError("pair_id has missing values")

    max_days = 126

    result = pd.Series(index=triggers.index, dtype="float64", name="f_days_since_trigger")

    for pair_id, group in df.groupby("pair_id"):
        group = group.sort_values("trigger_date")
        prev_pos = None

        for i, row in group.iterrows():
            trigger_date = row["trigger_date"]
            if trigger_date not in calendar:
                raise ValueError(f"trigger date {trigger_date} is not in the calendar")

            pos = calendar.get_loc(trigger_date)
            if prev_pos is None:
                result.loc[i] = max_days
            else:
                gap = pos - prev_pos
                if gap <= 0:
                    raise ValueError(f"pair {pair_id} has duplicate or invalid trigger dates")
                result.loc[i] = min(gap, max_days)
            prev_pos = pos

    return result


def calculate_cluster_stability(triggers: pd.DataFrame,stability: pd.DataFrame) -> pd.Series:
    """Find the latest known co-clustered value for each trigger"""
    require_columns(triggers, ["pair_id", "trigger_date"], "triggers")
    require_columns(stability, ["pair_id", "window_end", "co_clustered"], "stability")

    df = clean_triggers(triggers)
    hist = stability.copy()
    hist["window_end"] = pd.to_datetime(hist["window_end"])

    if df["pair_id"].isna().any():
        raise ValueError("trigger pair_id has missing values")
    if hist["pair_id"].isna().any():
        raise ValueError("stability pair_id has missing values")
    if hist["window_end"].isna().any():
        raise ValueError("window_end has missing values")
    if hist["co_clustered"].isna().any():
        raise ValueError("co_clustered has missing values")
    if not hist.empty and not pd.api.types.is_bool_dtype(hist["co_clustered"]):
        raise TypeError("co_clustered must be boolean")
    if hist.duplicated(["pair_id", "window_end"]).any():
        raise ValueError("stability has duplicate pair and window_end rows")

    result = pd.Series(0.0, index=triggers.index, dtype="float64", name="f_cluster_stability")

    for i, row in df.iterrows():
        past = hist[(hist["pair_id"] == row["pair_id"]) & (hist["window_end"] 
                                                           <= row["trigger_date"])]
        if past.empty:
            continue
        latest = past.sort_values("window_end").iloc[-1]
        if latest["co_clustered"]:
            result.loc[i] = 1.0

    return result


def build_features(triggers: pd.DataFrame,*,zscores: pd.DataFrame,spreads: pd.DataFrame,
                   residuals: pd.DataFrame,factors: pd.DataFrame,volume: pd.DataFrame,
                   stability: pd.DataFrame,pairs: pd.DataFrame) -> pd.DataFrame:
    """Add the seven raw features to a copy of the trigger table"""
    require_columns(triggers, ["z_trigger"], "triggers")
    if triggers["z_trigger"].isna().any():
        raise ValueError("z_trigger has missing values")

    result = triggers.copy()

    result["f_abs_z"] = result["z_trigger"].abs().astype("float64")
    result["f_spread_vol_60d"] = calculate_spread_volatility(result, spreads)
    result["f_resid_mom_5d"] = calculate_residual_momentum(result, residuals, pairs)
    result["f_mkt_vol_20d"] = calculate_market_volatility(result, factors)
    result["f_rel_volume_20d"] = calculate_relative_volume(result, volume, pairs)
    result["f_days_since_trigger"] = calculate_days_since_trigger(result, zscores.index)
    result["f_cluster_stability"] = calculate_cluster_stability(result, stability)

    for f in config.FEATURES:
        result[f] = result[f].astype("float64")

    return result


def fill_missing_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Fill missing features with medians taken from train rows only"""
    require_columns(dataset, ["split"] + list(config.FEATURES), "dataset")

    result = dataset.copy()
    train_rows = result["split"] == "train"
    if not train_rows.any():
        raise ValueError("dataset has no training rows")

    for f in config.FEATURES:
        median = result.loc[train_rows, f].median()
        if pd.isna(median):
            raise ValueError(f"{f} has no usable training median")
        result[f] = result[f].fillna(median).astype("float64")

    return result


def assemble_dataset(track: str) -> pd.DataFrame:
    """Build, check and save the full trigger dataset for one track"""
    if track not in ("a", "b"):
        raise ValueError(f"unknown track: {track}")

    data_dir = config.DATA_DIR

    zscores = read_parquet(data_dir / "spreads" / f"zscores_{track}.parquet")
    pairs = pd.read_csv(data_dir / "pairs" / f"pairs_{track}.csv",
                        parse_dates=["active_from", "active_to"])

    triggers = detect_triggers(zscores, pairs)
    triggers["split"] = assign_split(triggers["trigger_date"],triggers["horizon_end_date"], 
                                     zscores.index)

    spreads = read_parquet(data_dir / "spreads" / f"spreads_{track}.parquet")
    stability = read_parquet(data_dir / "clusters" / f"stability_{track}.parquet")
    residuals = read_parquet(data_dir / "processed" / "residuals_a.parquet")
    factors = read_parquet(data_dir / "processed" / "factors_a.parquet")
    volume = read_parquet(data_dir / "raw" / "volume.parquet")

    dataset = build_features(triggers, zscores=zscores, spreads=spreads,residuals=residuals, 
                             factors=factors, volume=volume,stability=stability, pairs=pairs)
    dataset = fill_missing_features(dataset)

    wanted = SCHEMAS["triggers"].columns
    dataset = dataset[list(wanted)]
    for col, dtype in wanted.items():
        dataset[col] = dataset[col].astype(dtype)

    if dataset["trigger_id"].duplicated().any():
        raise ValueError("trigger_id values are not unique")
    if not dataset["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0 and 1")
    if not dataset["split"].isin(["train", "val", "test", "purged", "embargo"]).all():
        raise ValueError("split has a value that is not allowed")

    write_parquet(dataset, data_dir / "datasets" / f"triggers_{track}.parquet", "triggers")
    return dataset


def run_tracks(track_text: str) -> None:
    """Build the dataset for each track in a comma separated list"""
    tracks = []
    for track in track_text.split(","):
        tracks.append(track.strip())

    for track in tracks:
        if track not in ("a", "b"):
            raise ValueError(f"unknown track: {track}")
    if len(tracks) != len(set(tracks)):
        raise ValueError("track names must not repeat")

    for track in tracks:
        dataset = assemble_dataset(track)
        print(f"track {track}: {len(dataset)} rows")


def main() -> None:
    """Run the dataset build from the command line"""
    parser = argparse.ArgumentParser(description="Build the trigger datasets")
    parser.add_argument("--tracks", default="a,b", help="comma-separated tracks")
    args = parser.parse_args()
    run_tracks(args.tracks)


if __name__ == "__main__":
    main()
