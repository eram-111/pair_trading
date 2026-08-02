# Pairs Trading — CSC311 Final Project

Predicting temporary stock-price gap convergence: a 3-track × 4-model factorial
comparison with a turnover-matched control. Spec: `project_spec_v2.md`.
Plan (who/when/how): `IMPLEMENTATION_PLAN.md`. Visual summary: `PLAN_ONEPAGER.html`.

## Quickstart

```
conda env create -f environment.yml
conda activate pair-trading
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pytest -q            # must be green before anything else
```

## Pipeline entry points (the only sanctioned ones)

```
make data        # returns.parquet + both fixtures from committed raw pulls (no network)
make tracka      # factors -> clusters -> pairs -> z-scores (track a)
make trackb      # Track B characteristics, then the shared machinery
make trackc      # Track C partial correlation, then the shared machinery
make dataset     # triggers_{track}.parquet for all tracks
make grid        # all 12 cells on train+validation
make noise-test  # full pipeline on random walks — must find nothing
make test-run    # Day 4 ~noon ONLY, witnessed, guarded
make figures     # all report figures from committed artifacts
```

Rules of the road: every file has exactly one owner (plan §1.2); hand-offs are
artifact contracts (plan §2.2), never imports of half-written code; seed 311
everywhere; the test set is touched exactly once.
