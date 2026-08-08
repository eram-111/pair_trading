"""Tests for the turnover-matched control in src/experiments.py.

Hand fixture: 6 triggers over two quarters; the fake model entered
2 of 4 in Q1 and 1 of 2 in Q2.

Run: pytest -q tests/test_experiments.py
"""
import pandas as pd
import pytest

from src.experiments import matched_control_decisions


def fake_triggers():
    """Only the columns the control reads: trigger_id, trigger_date."""
    return pd.DataFrame({
        "trigger_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "trigger_date": pd.to_datetime([
            "2020-01-10", "2020-02-10", "2020-03-10", "2020-03-20",
            "2020-04-10", "2020-05-10",
        ]),
    })


def fake_decisions():
    """The model entered T1 and T3 (Q1) and T5 (Q2)."""
    return pd.DataFrame({
        "trigger_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "enter": [True, False, True, False, True, False],
        "p_hat": [0.7, 0.3, 0.8, 0.2, 0.9, 0.1],
    })


def entered_count_by_quarter(decisions, triggers):
    """{quarter -> how many the strategy entered}."""
    date_by_id = triggers.set_index("trigger_id")["trigger_date"]
    quarters = date_by_id.loc[decisions["trigger_id"]].dt.to_period("Q").values
    counts = {}
    for quarter, entered in zip(quarters, decisions["enter"]):
        if entered:
            counts[quarter] = counts.get(quarter, 0) + 1
    return counts


def test_control_matches_the_models_quarter_counts():
    triggers = fake_triggers()
    decisions = fake_decisions()
    model_counts = entered_count_by_quarter(decisions, triggers)

    for seed in range(5):
        control = matched_control_decisions(decisions, triggers, seed)
        control_counts = entered_count_by_quarter(control, triggers)
        assert control_counts == model_counts, f"seed {seed}: quarter counts differ"


def test_degenerate_enter_all_model_controls_are_itself():
    # a model that entered everything leaves the control no choice:
    # sampling all of each quarter without replacement picks everything
    triggers = fake_triggers()
    decisions = fake_decisions()
    decisions["enter"] = True

    control = matched_control_decisions(decisions, triggers, seed=0)
    assert control["enter"].all(), "control of an enter-all model must enter everything"


def test_same_seed_reproduces_the_same_control():
    triggers = fake_triggers()
    decisions = fake_decisions()

    first = matched_control_decisions(decisions, triggers, seed=3)
    second = matched_control_decisions(decisions, triggers, seed=3)
    assert first.equals(second), "same seed must reproduce the same control"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
