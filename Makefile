.PHONY: all data tracka trackb dataset models grid metrics noise-test test-run figures output test

data:            ## returns.parquet, prices.parquet, volume.parquet, spy.parquet, universe.csv
	python -m src.data
	python -m src.make_synthetic
data_pull:       ## downloads new data from yfinance, then works same as make data
	python -m src.data --pull
tracka: data    
	python -m src.representation --track a
trackb: data  
	python -m src.characteristics
	python -m src.representation --track b
dataset: tracka trackb
	python -m src.dataset --tracks a,b
models:          ## fit + tune + freeze each model x track: writes results/frozen/{model}_{track}.joblib + taus.json
	python -m src.train --model e1 --track a
	python -m src.train --model e2 --track a
	python -m src.train --model e3 --track a
	python -m src.train --model e1 --track b
	python -m src.train --model e2 --track b
	python -m src.train --model e3 --track b
grid:         
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split trainval
metrics:       
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split val
	python -m src.metrics --tracks a,b --split val
	python -m src.analysis
noise-test:     
	python -m src.noise_test
test-run:      
	python -m src.experiments --tracks a,b --models e0,e1,e2,e3 --split test
figures:   
	python -m src.metrics --tracks a,b --split test
	python -m src.analysis
output:       
	mkdir -p output
	-cp results/tables/*.csv output/
	-cp results/figures/*.png output/
	-cp results/control_*.json output/
	-cp results/noise/PASS_FAIL.md output/
	-cp results/frozen/taus.json output/
test:
	pytest -q
all: data tracka trackb dataset models grid metrics noise-test output
