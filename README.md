# Pairs Trading — CSC311 Final Project (Option 2)

Does machine learning help a pairs-trading strategy? A 2-track × 4-model
comparison scored by identical machinery, with a turnover-matched control.

- **Report:** `report/final_report.pdf` (source: `report/final_report.tex`)
- **Numbers the report cites:** `results/` (tables, figures, ledgers, frozen models)
- **Report-ready bundle:** `output/`
- **Background and citations:** `project_spec_v2.md`
- **Design and ownership:** `IMPLEMENTATION_PLAN.md`

## Quickstart

```
conda env create -f environment.yml
conda activate pair-trading
pytest -q            # 35 tests, must be green
```

## Pipeline

Raw price/characteristic pulls are committed, so nothing needs the network.
Derived data (`data/processed`, `data/spreads`, …) is **not** committed because
it regenerates exactly; `results/` is committed instead, so you can read every
number without rerunning anything.

```
make data        # prices, returns, synthetic fixture, from committed raw pulls
make tracka      # factors -> clusters -> pairs -> z-scores (track a)
make trackb      # characteristics -> pairs, then the shared machinery
make dataset     # triggers_{track}.parquet
make models      # fit + tune + freeze each model x track
make grid        # all 8 cells on train+validation
make metrics     # produce + analyse the validation split (tables, figures)
make noise-test  # whole pipeline on random walks — must find nothing
make test-run    # the ONE held-out test run
make figures     # regenerate every table and figure from test results
make output      # copy report-ready artifacts into output/
```

`make all` runs everything up to (but excluding) the test run. Run it serially,
never with `-j`: evaluation targets deliberately have no build prerequisites so
that evaluating can never silently retrain a frozen model.

## Pre-registered choices

These were fixed **before** any test-set number existed, and are what the
report's "decided in advance" claims refer to.

- **Seed** 311 everywhere; every stochastic step is seeded.
- **Splits** by date: train 2015–2020, validation 2021–2022, test 2023–2024,
  with 5 days of label-overlapping examples purged before each boundary and 10
  days embargoed after.
- **Entry threshold rule (τ).** Per (model, track), τ is chosen from
  {0.40, 0.45, …, 0.80} to maximise validation net P&L at 10 bps, subject to at
  least 25 accepted validation trades; P&L ties (within 1e-6) go to the higher
  τ; if no τ reaches 25 trades, the τ with the most trades wins; if no τ trades
  at all, the cell is degenerate and τ = 0.5. Chosen values live in
  `results/frozen/taus.json`.
- **Headline cost** 10 bps per leg per transaction (40 bps per round trip);
  results are also reported across a 0–50 bps grid.
- **The test split was read once**, after all models and thresholds were frozen.

### Frozen models

`results/frozen/*.joblib` are the exact fitted models the reported test numbers
came from, with `taus.json` holding their thresholds. Re-running `make models`
refits and replaces them — don't, unless you intend to.

One deviation is worth recording: **E2** was first fitted with empirical class
priors, which put every probability below the τ grid so it never traded. That
was a mismatch with how E1 and E3 handle class imbalance rather than a property
of the method; it was spotted on validation, refitted with balanced priors so
all four models are treated alike, and its test cells were re-run afterwards.

## Layout

```
src/            pipeline: data -> representation -> dataset -> models -> evaluation
tests/          35 tests, including golden files for the trade simulator
results/        every number and figure the report cites
report/         final_report.tex and the compiled PDF
scripts/        extra figure generation
```
