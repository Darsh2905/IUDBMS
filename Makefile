# IUDBMS — development entry points.
.PHONY: help install test test-fast lint eval zoo report monitor monitor-replay demo demo-hard dashboard baseline clean

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
	python3 -m eval.run_eval --cache out/corpus_cache.npz --target-fpr 0.0001 --persist

fusion-ablation:  ## pillar N1: fused vs single-modality on single-axis attacks
	python3 -m eval.run_fusion_ablation

baseline-ablation:  ## pillar N2: per-device baseline, with and without device variation
	python3 -m eval.run_baseline_ablation

cross-device:  ## does host context clear the single-shot mimicry bound?
	python3 -m eval.run_cross_device

zoo:  ## compare model families on the selection split (~60 s)
	python3 -m eval.model_zoo --on selection

report:  ## regenerate docs/report/ from the last eval (charts + numbers)
	python3 -m eval.build_report

paper:  ## rebuild the IEEE .docx; every Results number comes from out/*.json
	python3 scripts/build_paper.py

diagrams:  ## architecture + gate diagrams as PNG/PDF (needs `make eval` first)
	python3 -m eval.render_architecture
	cp out/architecture.png out/architecture.pdf out/architecture_ieee.png out/architecture_ieee.pdf \
	   out/gate_timeline.png out/gate_timeline.pdf docs/report/

monitor-replay:  ## the monitor loop driven from a recorded capture (any OS); writes out/live.db
	python3 scripts/live_monitor.py --replay tests/fixtures/usbmon/demo_capture.txt \
		--descriptors tests/fixtures/usbmon/demo_descriptors.json

monitor:  ## LIVE capture on Linux (needs root + usbmon); writes out/live.db
	sudo python3 scripts/live_monitor.py --bus 0

dashboard-live:  ## the dashboard over the monitor's database rather than the evaluation's
	IUDBMS_DB=out/live.db python3 -m uvicorn iudbms.api.app:app

dist:  ## build the installable wheel into dist/
	python3 -m pip wheel . --no-deps -w dist

install-service:  ## install the systemd services on this Linux host (see docs/DEPLOYMENT.md)
	sudo ./deploy/install.sh

verify-service:  ## post-install checks on this Linux host
	sudo ./deploy/verify.sh

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
