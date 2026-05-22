.PHONY: install data reproduce-all reproduce-e0 reproduce-e1 reproduce-e2 reproduce-e3 reproduce-e4 reproduce-e4t reproduce-e5 reproduce-e6 figures tables test clean

install:
	pip install -e ".[dev,viz]"
	-pre-commit install

data:
	python -m artifact_limits.data.download_eegdenoisenet
	python -m artifact_limits.data.build_D1prime --alpha-grid 0,0.05,0.1,0.2,0.3,0.5,1.0
	-python -m artifact_limits.data.load_D3_audit

reproduce-all: data
	$(MAKE) reproduce-e0
	$(MAKE) reproduce-e1
	$(MAKE) reproduce-e2
	$(MAKE) reproduce-e3
	$(MAKE) reproduce-e4
	$(MAKE) reproduce-e4t
	$(MAKE) reproduce-e5
	$(MAKE) reproduce-e6
	$(MAKE) figures
	$(MAKE) tables

reproduce-e0:
	python -m experiments.run E0 --seed 42

reproduce-e1:
	for seed in 42 123 2024 7 31337; do python -m experiments.run E1 --seed $$seed; done

reproduce-e2:
	for seed in 42 123 2024 7 31337; do python -m experiments.run E2 --seed $$seed; done

reproduce-e3:
	for seed in 42 123 2024 7 31337; do python -m experiments.run E3 --seed $$seed; done

reproduce-e4:
	for seed in 42 123 2024 7 31337; do python -m experiments.run E4 --seed $$seed; done

reproduce-e4t:
	python -m experiments.run E4t --seed 42

reproduce-e5:
	python -m experiments.run E5 --seed 42

reproduce-e6:
	python -m experiments.run E6 --seed 42

figures:
	python -m artifact_limits.viz.figure1_identifiability
	python -m artifact_limits.viz.figure2_wiener_floor
	python -m artifact_limits.viz.figure3_dependent_mixing
	python -m artifact_limits.viz.figure4_capacity_scaling
	python -m artifact_limits.viz.figure5_in_vivo_audit
	python -m artifact_limits.viz.figure6_pool_audit

tables:
	python results/consolidate.py

test:
	pytest -xvs tests/

clean:
	rm -rf results/figures/* results/tables/*
