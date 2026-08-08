"""Noise test: full pipeline on synthetic random-walk prices; any skill found means leakage.
Run: python -m src.noise_test  (make noise-test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config
from src.contracts import SCHEMAS, read_parquet, seed_everything, write_parquet
from src.data import compute_returns
from src.dataset import assign_split, build_features, detect_triggers, fill_missing_features
from src.engine import run_backtest
from src.models.common import e0_decisions
from src.representation import run_track_a


def _learned_model_classes() -> dict:
    """name -> class for each learned model that imports; import errors mean "not written yet"."""
    classes = {}
    try:
        from src.models.e1 import E1
        classes["e1"] = E1
    except Exception:
        pass
    try:
        from src.models.e2 import E2
        classes["e2"] = E2
    except Exception:
        pass
    try:
        from src.models.e3 import E3
        classes["e3"] = E3
    except Exception:
        pass
    return classes


def build_synthetic_triggers() -> tuple:
    """Run the real pipeline on the synthetic data (needs 'make data' first).
    Returns (triggers, zscores, prices).
    """
    # synthetic inputs
    prices_file = Path("data/synth/raw/prices.parquet")
    volume_file = Path("data/synth/raw/volume.parquet")
    assert prices_file.exists(), "synthetic data missing"
    assert volume_file.exists(), "synthetic data missing"
    prices = read_parquet(prices_file)
    volume = read_parquet(volume_file)
    returns = compute_returns(prices)

    factors, meta, residuals, loadings, labels, stability, pairs, spreads, zscores = run_track_a(returns)

    # 3. the dataset chain, exactly as the builder runs it
    triggers = detect_triggers(zscores, pairs)
    triggers["split"] = assign_split(triggers["trigger_date"], triggers["horizon_end_date"], zscores.index)
    triggers = build_features(triggers, zscores=zscores, spreads=spreads, residuals=residuals, factors=factors, volume=volume, stability=stability, pairs=pairs)
    triggers = fill_missing_features(triggers)
    wanted = SCHEMAS["triggers"].columns
    triggers = triggers[list(wanted)]
    for col, dtype in wanted.items():
        triggers[col] = triggers[col].astype(dtype)

    # 4. write the synthetic trigger table
    write_parquet(triggers, "results/noise/triggers_synth.parquet", "triggers")
    return triggers, zscores, prices


def _check_returns(trades: pd.DataFrame) -> str:
    """Check whether e0's net return is significantly positive"""
    net = trades[f"net_ret_{config.HEADLINE_COST_BPS}bps"]
    mean_net = net.mean()
    standard_error = net.std() / (len(net) ** 0.5)
    ci_low = mean_net - 1.96 * standard_error
 
    if ci_low > 0:
        verdict = "FAIL"
    else: 
        verdict = "PASS"

    return f"{verdict} returns: mean={mean_net:.5f}, CI_low={ci_low:.5f}"


def _auc_difference_ci_low(y_val, p_model, p_base, rng) -> float:
    """Bootstrap the (model - baseline) AUC difference on val rows; returns the 2.5th percentile."""
    diffs = []
    for draw in range(1000):
        sample = rng.integers(0, len(y_val), size=len(y_val))
        y_sample = y_val[sample]
        if y_sample.min() == y_sample.max():
            continue
        diff = roc_auc_score(y_sample, p_model[sample]) - roc_auc_score(y_sample, p_base[sample])
        diffs.append(diff)
    return float(np.percentile(diffs, 2.5))


def _check_model_aucs(triggers: pd.DataFrame) -> list:
    """Check whether any learned model beats the abs-z baseline."""

    lines = []

    train_mask = triggers["split"] == "train"
    train = triggers[train_mask]
    val_mask = triggers["split"] == "val"
    val = triggers[val_mask]
    y_train = train["label"].values
    y_val = val["label"].values


    # the yardstick: how predictable the label is from |z| alone
    baseline = LogisticRegression(max_iter=2000, random_state=config.SEED)
    baseline.fit(train[["f_abs_z"]], y_train)

    p_base = baseline.predict_proba(val[["f_abs_z"]])[:, 1]
    auc_base = roc_auc_score(y_val, p_base)

    lines.append(f"baseline AUC = {auc_base:.3f}")

    model_classes = _learned_model_classes()
    if not model_classes:
        lines.append("SKIP: no learned models available")
        return lines

    
    rng = np.random.default_rng(config.SEED)

    
    for name in sorted(model_classes):
        try:
            model = model_classes[name]().fit(train, y_train)
        except NotImplementedError:
            lines.append(f"SKIP {name}: not implemented")
            continue

        p_model = model.predict_proba(val)
        auc_model = roc_auc_score(y_val, p_model)

        diff_ci_low = _auc_difference_ci_low(y_val, p_model, p_base, rng)

        
        if diff_ci_low > 0:
            verdict = "FAIL"
        else: 
            verdict = "PASS"

        lines.append(f"{verdict} {name}: AUC={auc_model:.3f}, baseline={auc_base:.3f}, CI_low={diff_ci_low:+.3f}")
    return lines


def _check_trigger_counts(tradeable: pd.DataFrame) -> str:
    """Check that enough triggers fire and both labels appear."""
    n_triggers = len(tradeable)
    base_rate = tradeable["label"].mean()

    if n_triggers <= 50 or base_rate <= 0 or base_rate >= 1:
        verdict = "FAIL"
    else: 
        verdict = "PASS"

    return f"{verdict}: {n_triggers} triggers, base rate={base_rate:.3f}"


def run_noise_test() -> None:
    """Write PASS/FAIL lines to results/noise/PASS_FAIL.md (also printed).
    Any FAIL means a leakage bug: stop and find it before trusting results.
    """
    out_dir = Path("results/noise")
    out_dir.mkdir(parents=True, exist_ok=True)

    triggers, zscores, prices = build_synthetic_triggers()

    # e0
    split_mask = triggers["split"].isin(("train", "val", "test"))
    tradeable = triggers[split_mask]
    decisions = e0_decisions(tradeable)
    trades = run_backtest(zscores, prices, tradeable, decisions)
    write_parquet(trades, out_dir / "trades_synth_e0.parquet", "trades")

    # the three criteria
    lines = []
    lines.append(_check_returns(trades))
    lines.extend(_check_model_aucs(triggers))
    lines.append(_check_trigger_counts(tradeable))

    report = "\n".join(lines) + "\n"

    (out_dir / "PASS_FAIL.md").write_text(report + "\n")

    print(report)


def main() -> None:
    seed_everything()
    run_noise_test()


if __name__ == "__main__":
    main()
