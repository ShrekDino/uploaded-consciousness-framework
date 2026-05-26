.PHONY: install lint check clean run-single run-multi train-lang

# ─── Setup ───

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install ruff mypy

# ─── Quality ───

lint:
	ruff check consciousness/ scripts/ config.py

format:
	ruff format consciousness/ scripts/ config.py

typecheck:
	mypy consciousness/ scripts/ config.py --ignore-missing-imports || true

check: lint typecheck

# ─── Clean ───

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .mypy_cache

# ─── Run ───

run-single:
	python scripts/run.py --single

run-multi:
	python scripts/run.py --nodes 3

train-lang:
	python scripts/run.py --train-lang --train-steps 500

train-lang-full:
	python scripts/run.py --train-lang --train-steps 1000
