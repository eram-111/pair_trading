.PHONY: all data tracka trackb dataset models grid metrics noise-test test-run figures output test

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
dataset: tracka trackb
	python -m src.dataset --tracks a,b
models:          ## fit + tune + freeze each model x track: writes results/frozen/{model}_{track}.joblib + taus.json
	python -m src.train --model e1 --track a
	python -m src.train --model e2 --track a
	python -m src.train --model e3 --track a
	python -m src.train --model e1 --track b
	python -m src.train --model e2 --track b
	python -m src.train --model e3 --track b
grid:            ## E0..E3 x tracks on train+val only; evaluates what exists, rebuilds nothing
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split trainval
metrics:         ## produce + analyze the VAL split: cells, report table, controls, breakevens, all figures
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split val
	python -m src.metrics --tracks a,b --split val
	python -m src.analysis
noise-test:      ## full pipeline on src/make_synthetic.py output; PASS/FAIL criteria
	python -m src.noise_test
test-run:        ## the ONE witnessed test-split run (soft freeze: discipline, not code)
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split test
figures:         ## regenerate every table + figure from written results (run after test-run)
	python -m src.metrics --tracks a,b --split test
	python -m src.analysis
output:          ## copy every report-ready artifact into output/ (a leading - ignores a missing group)
	mkdir -p output
	-cp results/tables/*.csv output/
	-cp results/figures/*.png output/
	-cp results/control_*.json output/
	-cp results/noise/PASS_FAIL.md output/
	-cp results/frozen/taus.json output/
test:
	pytest -q
# everything up to (but excluding) the one witnessed test run; run serially, never with -j
all: data tracka trackb dataset models grid metrics noise-test output
