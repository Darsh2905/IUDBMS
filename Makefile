# IUDBMS — development entry points.
.PHONY: help install test test-fast lint eval report demo demo-hard dashboard baseline clean

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

report:  ## regenerate docs/report/ from the last eval (charts + numbers)
	python3 -m eval.build_report

demo:  ## live replay: watch the loop close in real time (needs `make eval` first)
	python3 scripts/live_replay.py --profile staged_delay --speed 6

demo-hard:  ## the same, on the mimicry adversary (slow, noisy detection)
	python3 scripts/live_replay.py --profile mimicry --speed 6

dashboard:  ## operator dashboard on http://localhost:8000
	python3 -m uvicorn iudbms.api.app:app

baseline:  ## re-run the Sprint-0 prototype for comparison (~46 s)
	python3 run_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__
