# IUDBMS — development entry points.
.PHONY: help install test test-fast lint eval zoo report monitor monitor-replay demo-dashboard lab db-dashboard demo demo-hard dashboard baseline clean

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

paper:  ## rebuild the IEEE .docx; supplied text is verbatim, every number is checked against out/*.json
	python3 scripts/build_paper.py
	python3 scripts/check_paper_text.py
	python3 scripts/check_paper_layout.py

diagrams:  ## architecture + gate diagrams as PNG/PDF (needs `make eval` first)
	python3 -m eval.render_architecture
	cp out/architecture.png out/architecture.pdf out/architecture_ieee.png out/architecture_ieee.pdf \
	   out/gate_timeline.png out/gate_timeline.pdf docs/report/

train-deployed:  ## train the DEPLOYED set on Corpus B, then calibrate its threshold per device-hour (~7 min)
	python3 -m eval.train_deployed
	python3 -m eval.analysis_calibration --adopt

long-sessions:  ## false alarms per device-hour on sessions up to 60 min (~10 min)
	python3 -m eval.analysis_long_sessions

model-card:  ## docs/MODEL_CARD.md from the trained artefacts
	python3 -m eval.model_card

testbed-images:  ## download the four distribution cloud images (~2 GB)
	sh testbed/fetch_images.sh

testbed:  ## real-kernel testbed: 4 distributions, benign 75 min + attacks 30 min each (~2 h)
	python3 -m testbed.human_input fetch
	python3 -m testbed.campaign all --seed 1
	python3 -m testbed.score

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

demo-dashboard:  ## the live demo page on http://localhost:8000/demo (deployed models, streamed window by window)
	python3 -m uvicorn iudbms.api.app:app --port 8000

lab:  ## live lab on http://localhost:8001: boots a Linux VM, scores its real usbmon live (~1 min to boot)
	python3 -m iudbms.api.lab --port 8001

db-dashboard:  ## database explorer on http://localhost:8002 over out/iudbms.db (the simulated corpus; needs `make eval`)
	python3 -m iudbms.api.dbview --port 8002

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
