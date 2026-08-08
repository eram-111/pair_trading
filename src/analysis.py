"""Base rate tables and figures per track and the cross track consensus table"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.contracts import read_parquet


def base_rate_table(triggers: pd.DataFrame) -> pd.DataFrame:
    """Calculate base rates overall and by split and year and market regime"""
    for c in ["label", "split", "trigger_date", "f_mkt_vol_20d"]:
        if c not in triggers.columns:
            raise ValueError(f"triggers is missing column: {c}")
        
    groups = [("overall", triggers)]

    for split, group in triggers.groupby("split"):
        groups.append((f"split={split}", group))
    for year, group in triggers.groupby(triggers["trigger_date"].dt.year):
        groups.append((f"year={year}", group))

    median_vol = triggers["f_mkt_vol_20d"].median()
    calm = triggers[triggers["f_mkt_vol_20d"] <= median_vol]
    stressed = triggers[triggers["f_mkt_vol_20d"] > median_vol]
    groups.append(("regime=calm", calm))
    groups.append(("regime=stressed", stressed))

    rows = []
    for name, group in groups:
        rows.append({"group": name,"n_triggers": len(group),"base_rate": group["label"].mean()})

    return pd.DataFrame(rows,columns=["group", "n_triggers", "base_rate"])


def base_rate_by_year_figure(triggers: pd.DataFrame,
                             out_path: str = "results/figures/base_rate_by_year.png") -> None:
    """Plot the base rate and number of triggers for each year"""
    for c in ["label", "trigger_date"]:
        if c not in triggers.columns:
            raise ValueError(f"triggers is missing column: {c}")

    rows = []
    for year, group in triggers.groupby(triggers["trigger_date"].dt.year):
        rows.append({"year": year,"n_triggers": len(group),"base_rate": group["label"].mean()})

    yearly = pd.DataFrame(rows,columns=["year", "n_triggers", "base_rate"])

    fig, count_axis = plt.subplots(figsize=(9, 5))

    bars = count_axis.bar(yearly["year"], yearly["n_triggers"], color="lightgray")
    count_axis.bar_label(bars)
    count_axis.set_xlabel("Year")
    count_axis.set_ylabel("Number of triggers")
    count_axis.set_xticks(yearly["year"])

    rate_axis = count_axis.twinx()
    rate_axis.plot(yearly["year"], yearly["base_rate"], color="blue", marker="o")
    rate_axis.set_ylabel("Base rate")
    rate_axis.set_ylim(0, 1)

    count_axis.set_title("Base Rate and Trigger Count by Year")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def consensus_table(pairs_by_track: dict, triggers_by_track: dict,
                    trades_by_track: dict) -> pd.DataFrame:
    """Compare pairs selected by one track with pairs selected by multiple tracks"""
    if len(pairs_by_track) < 2:
        raise ValueError("consensus table needs at least two tracks")

    rows = []
    for track, pairs in pairs_by_track.items():
        for _, pair in pairs.iterrows():
            quarters = pd.period_range(pair["active_from"],pair["active_to"],freq="Q")
            for quarter in quarters:
                rows.append({"track": track,"pair_id": pair["pair_id"],"quarter": quarter})

    selected = pd.DataFrame(rows).drop_duplicates()
    counts = selected.groupby(["pair_id", "quarter"]).size().reset_index(name="n_tracks")
    selected = selected.merge(counts,on=["pair_id", "quarter"])

    trigger_tables = []
    for track, table in triggers_by_track.items():
        table = table.copy()
        table["track"] = track
        table["quarter"] = table["trigger_date"].dt.to_period("Q")
        trigger_tables.append(table)

    triggers = pd.concat(trigger_tables,ignore_index=True)
    triggers = triggers.merge(selected,on=["track", "pair_id", "quarter"])

    trade_tables = []
    for track, table in trades_by_track.items():
        table = table.copy()
        table["track"] = track
        trade_tables.append(table)

    trades = pd.concat(trade_tables,ignore_index=True)
    trades = trades.merge(triggers[["track", "trigger_id", "n_tracks"]],on=["track", "trigger_id"])
    net_column = f"net_ret_{config.HEADLINE_COST_BPS}bps"

    rows = []
    for n_tracks in sorted(counts["n_tracks"].unique()):
        pair_group = counts[counts["n_tracks"] == n_tracks]
        trigger_group = triggers[triggers["n_tracks"] == n_tracks]
        trade_group = trades[trades["n_tracks"] == n_tracks]
        rows.append({"n_tracks": n_tracks,"n_pair_quarters": len(pair_group),
                     "n_triggers": len(trigger_group),
                     "base_rate": trigger_group["label"].mean(),
                     "mean_net_return": trade_group[net_column].mean()})

    return pd.DataFrame(rows,columns=["n_tracks", "n_pair_quarters", "n_triggers",
                                      "base_rate", "mean_net_return"])


def main() -> None:
    """Create the base rate and consensus results"""
    tables_dir = config.RESULTS_DIR / "tables"
    figures_dir = config.FIGURES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    triggers_by_track = {}
    for track in ["a", "b"]:
        triggers_path = config.DATA_DIR / "datasets" / f"triggers_{track}.parquet"
        if triggers_path.exists():
            triggers_by_track[track] = read_parquet(triggers_path)

    if not triggers_by_track:
        raise ValueError("no trigger datasets found, run src.dataset first")

    for track, triggers in triggers_by_track.items():
        table = base_rate_table(triggers)
        print(f"base rate, track {track}:")
        print(table.to_string(index=False))
        table.to_csv(tables_dir / f"base_rate_{track}.csv",index=False)
        figure_path = figures_dir / f"base_rate_by_year_{track}.png"
        base_rate_by_year_figure(triggers,str(figure_path))

    val_end = pd.Timestamp(config.VAL_END)
    pairs_by_track = {}
    trades_by_track = {}
    consensus_triggers = {}

    for track, triggers in triggers_by_track.items():
        pairs_path = config.DATA_DIR / "pairs" / f"pairs_{track}.csv"
        trades_path = config.RESULTS_DIR / f"trades_{track}_e1_trainval.parquet"
        if not pairs_path.exists() or not trades_path.exists():
            continue
        pairs = pd.read_csv(pairs_path,parse_dates=["active_from", "active_to"])
        pairs = pairs[pairs["active_from"] <= val_end].copy()
        pairs["active_to"] = pairs["active_to"].clip(upper=val_end)
        pairs_by_track[track] = pairs
        trades_by_track[track] = read_parquet(trades_path)
        consensus_triggers[track] = triggers[triggers["split"].isin(["train", "val"])]

    if len(pairs_by_track) < 2:
        print("two completed tracks are required")
        return

    consensus = consensus_table(pairs_by_track,consensus_triggers,trades_by_track)
    print("consensus:")
    print(consensus.to_string(index=False))
    consensus.to_csv(tables_dir / "consensus.csv",index=False)

if __name__ == "__main__":
    main()
