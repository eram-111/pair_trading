.PHONY: all data tracka trackb trackc dataset grid noise-test test-run figures test

data:            ## returns.parquet, prices.parquet, volume.parquet, spy.parquet, universe.csv
	python -m src.data
	python -m src.make_synthetic
data_pull:       ## downloads new data from yfinance, then works same as make data
	python -m src.data --pull
tracka: data     ## Steps 2-6, track a: factors -> clusters -> pairs -> z-scores, one module
	python -m src.representation --track a
trackb: data     ## Track B approach file, then the shared representation machinery
	python -m src.characteristics
	python -m src.representation --track b
# trackc: tracka   ## Track C approach file (reads corr_windows.npz), then the shared machinery
# 	python -m src.partial_corr
# 	python -m src.representation --track c
dataset: tracka trackb trackc
	python -m src.dataset --tracks a,b,c
grid: dataset    ## E0/E1/E2/E3 x tracks on train+val only (tracks read from config so the Makefile cannot drift from the committed grid)
	python -m src.experiments grid --tracks a,b,c --models e0,e1,e2,e3 --split trainval
noise-test:      ## full pipeline on src/make_synthetic.py output; PASS/FAIL per Part 4
	python -m src.experiments noise
test-run:        ## Day 4 ONLY, witnessed; guarded (--i-am-sure + FREEZE.md + date >= Aug 5 + no results/final/TEST_RUN_COMPLETE marker)
	python -m src.experiments grid --tracks a,b,c --models e0,e1,e2,e3 --split test --i-am-sure
figures: grid
	python -m scripts.figures.make_all
test:
	pytest -q
all: data tracka trackb trackc dataset grid figures
