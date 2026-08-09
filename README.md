# Pairs Trading — CSC311 Final Project

Does machine learning help a pairs-trading strategy? We test 8 versions:
2 ways of choosing stock pairs x 4 ways of deciding when to trade.

Answer: the models predict slightly better than chance, but none makes
money after trading fees.

## Where things are

```
report/     the report (final_report.pdf) and its LaTeX source
results/    every number and figure the report cites
src/        the pipeline code
```

## Run it

```
conda env create -f environment.yml
conda activate pair-trading
make all          # raw prices -> results, everything except the test run
make noise-test   # runs the pipeline on random data; must find nothing
make test-run     # the held-out test years; we ran this once
```

Raw price data is committed, so nothing needs the network. Derived data is
not committed because `make all` regenerates it exactly.

Everything downstream of the raw prices is a deterministic function of them,
`src/config.py`, and seed 311, so the same environment gives the same numbers.
Run `make all` serially, not with `-j`.

## The 8 versions

| | |
|---|---|
| Track A | pairs of stocks that move alike |
| Track B | pairs of companies that look alike |
| E0 | trade every gap (no learning) |
| E1 | logistic regression |
| E2 | Gaussian discriminant analysis |
| E3 | small neural network |

## Rules we fixed before looking at results

- Seed 311 everywhere.
- Split by date: train 2015-2020, validation 2021-2022, test 2023-2024.
- Trading cost: 10 bps per leg, so 40 bps per round trip.
- Each model's entry threshold comes from one pre-set rule, run on
  validation data only. Chosen values: `results/frozen/taus.json`.
- The test years were read once, after everything was frozen.

`results/frozen/*.joblib` are the exact models behind the reported numbers.
`make models` refits and replaces them, so don't run it unless you mean to.
