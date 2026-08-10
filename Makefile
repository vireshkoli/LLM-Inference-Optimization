.DEFAULT_GOAL := help
SHELL := /bin/bash

# GPU 1 is the measurement device. GPU 0 shares chassis airflow with it and is
# treated as a neighbour whose activity is recorded, never as a second worker:
# parallelising the sweep across both cards would reintroduce exactly the
# thermal and PCIe confound this methodology exists to control.
BENCH_GPU ?= 1

.PHONY: help
help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Sync the dev environment (uv, Python 3.12)
	uv sync --all-extras --dev

.PHONY: lint
lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt:  ## Apply ruff formatting and autofixes
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: types
types:  ## mypy --strict
	uv run mypy

.PHONY: test
test:  ## Run the CPU-only test suite (no GPU required)
	uv run pytest

.PHONY: check
check: lint types test  ## Everything CI runs

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
.PHONY: bench-smoke
bench-smoke:  ## ~15 min pipeline validation. The gate before the full sweep.
	uv run llmbench sweep --config configs/smoke.yaml --gpu $(BENCH_GPU)

.PHONY: bench
bench:  ## Full sweep matrix (~18 GPU-hours). Run bench-smoke first.
	uv run llmbench sweep --config configs/sweep.yaml --gpu $(BENCH_GPU)

.PHONY: quality
quality:  ## Perplexity + GSM8K + IFEval across quantization levels
	uv run llmbench quality --config configs/sweep.yaml --gpu $(BENCH_GPU)

.PHONY: report
report:  ## Regenerate every chart and table in README/REPORT from results JSON
	uv run llmbench report --results results/runs --out .

.PHONY: lock-clocks
lock-clocks:  ## Pin GPU clocks for measurement stability (requires sudo)
	sudo ./scripts/lock_clocks.sh $(BENCH_GPU) lock

.PHONY: unlock-clocks
unlock-clocks:  ## Restore default GPU clock behaviour
	sudo ./scripts/lock_clocks.sh $(BENCH_GPU) unlock

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
.PHONY: stack-up
stack-up:  ## Bring up Prometheus + Grafana + DCGM exporter
	docker compose -f docker/compose.full.yml up -d prometheus grafana dcgm-exporter

.PHONY: stack-down
stack-down:  ## Tear down the observability stack
	docker compose -f docker/compose.full.yml down

.PHONY: clean
clean:  ## Remove caches and generated artifacts (keeps committed results)
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
