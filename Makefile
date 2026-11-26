.PHONY: install test quickstart sweep estimate view pages docker clean

install:        ## editable install with test extras
	pip install -e ".[test]"

test:           ## unit and integration tests (offline)
	pytest -q tests

quickstart:     ## offline matched experiment with oracle and judge
	python -m hawthornebench run -c configs/quickstart.toml

sweep:          ## dose-response sweep over red-team intensity and style
	python -m hawthornebench sweep -c configs/sweep_dose_response.toml

estimate:       ## calls, tokens and cost of the full study
	python -m hawthornebench estimate -c configs/full_study.toml

view:           ## 3D viewer on the quickstart traces
	python -m hawthornebench view runs/quickstart

pages:          ## rebuild the GitHub Pages site in docs/
	python scripts/build_pages.py

docker:
	docker build -t hawthornebench .

clean:
	rm -rf runs .pytest_cache **/__pycache__
