# RUNBOOK — fresh clone to full grid

Owner: P3. Verified on a clean machine Day 2, re-verified Day 4.
After Day 1, no target ever contacts Yahoo or Bloomberg — committed raw pulls
are the roots of the DAG; everything downstream is a deterministic function of
them plus src/config.py and seed 311.

```
1. git clone https://github.com/eram-111/pair_trading.git && cd pair_trading
2. conda env create -f environment.yml
3. conda activate pair-trading
4. pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
5. pytest -q                      # all green before anything else
6. make all                       # raw parquet -> full train+val grid, ~minutes
7. make noise-test                # must find nothing
8. make test-run                  # Day 4, once, all three watching
```

Definition of done: a teammate who has never touched the repo runs steps 1–7 on
a clean machine on Day 3 and gets bit-identical results/metrics_* (same seed,
same lockfile). That dry-run IS the reproducibility claim in the report.
