.PHONY: install test quickstart sweep estimate view pages docker clean

install:        ## editable install with test extras
	pip install -e ".[test]"

test:           ## unit and integration tests (offline)
	pytest -q tests

quickstart:     ## offline matched experiment with oracle and judge
	python -m survival_island run -c configs/quickstart.toml

sweep:          ## dose-response sweep over red-team intensity and style
	python -m survival_island sweep -c configs/sweep_dose_response.toml

estimate:       ## calls, tokens and cost of the full study
	python -m survival_island estimate -c configs/full_study.toml

view:           ## 3D viewer on the quickstart traces
	python -m survival_island view runs/quickstart

pages:          ## rebuild the GitHub Pages site in docs/
	python scripts/build_pages.py

docker:
	docker build -t survival-island .

clean:
	rm -rf runs .pytest_cache **/__pycache__
