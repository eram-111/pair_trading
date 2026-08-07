.PHONY: all data tracka trackb dataset models grid noise-test test-run figures test

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
models:          ## fit + tune + freeze each model: writes results/frozen/{model}_{track}.joblib + taus.json
	python -m src.models.e1 --track a
	python -m src.models.e3 --track a
grid:            ## E0..E3 x tracks on train+val only; evaluates what exists, rebuilds nothing
	python -m src.experiments grid --tracks a,b --models e0,e1,e2,e3 --split trainval
noise-test:      ## full pipeline on src/make_synthetic.py output; PASS/FAIL criteria
	python -m src.experiments noise
test-run:        ## the ONE witnessed test-split run (soft freeze: discipline, not code)
	python -m src.experiments grid --tracks a,b --models e0,e1,e2,e3 --split test
figures:
	python -m scripts.figures.make_all
test:
	pytest -q
all: data tracka trackb dataset models grid figures
