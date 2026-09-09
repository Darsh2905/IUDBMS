# IUDBMS — development entry points.
.PHONY: help install test test-fast lint eval demo baseline clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install + dev extras
	python3 -m pip install -e ".[dev]"

test:  ## full pytest suite
	python3 -m pytest

test-fast:  ## skip anything marked slow
	python3 -m pytest -m "not slow"

lint:  ## ruff check (no-op if ruff is absent)
	@command -v ruff >/dev/null && ruff check src tests eval || echo "ruff not installed; skipping"

eval:  ## regenerate every reported table/figure (seeded)
	python3 -m eval.run_all

baseline:  ## re-run the Sprint-0 prototype for comparison (~46 s)
	python3 run_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__
