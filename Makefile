UV ?= uv
PYTHON ?= python3

.PHONY: sync format lint typecheck architecture markdown source-check source-check-strict test test-fast build check check-local clean

sync:
	$(UV) sync --locked --group dev

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

architecture:
	$(UV) run python scripts/check_architecture.py

markdown:
	$(UV) run python scripts/check_markdown_math.py

source-check:
	$(PYTHON) scripts/check_source_checkouts.py

source-check-strict:
	$(PYTHON) scripts/check_source_checkouts.py --require-clean --timeout-seconds 300

test-fast:
	$(UV) run pytest -m "unit or architecture or contract"

test:
	$(UV) run pytest --cov=femx --cov-branch --cov-report=term-missing

build:
	$(UV) build

check: lint typecheck architecture markdown test build

check-local: source-check check

clean:
	rm -rf build dist .coverage coverage.xml htmlcov .mypy_cache .pytest_cache .ruff_cache
