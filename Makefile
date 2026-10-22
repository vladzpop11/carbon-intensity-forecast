.PHONY: help setup test data data-push data-pull backtest

# DagsHub credentials for DVC come from .env (never committed): DAGSHUB_USER, DAGSHUB_TOKEN.
# DagsHub's S3 endpoint takes the token as both key and secret.
WITH_DAGSHUB = set -a; . ./.env; set +a; \
	AWS_ACCESS_KEY_ID="$$DAGSHUB_TOKEN" AWS_SECRET_ACCESS_KEY="$$DAGSHUB_TOKEN"

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies from uv.lock
	uv sync --locked

test:  ## Ruff and pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

data:  ## Download the raw data, check its contracts, track the new version with DVC
	uv run python -m carbon_intensity_forecast.ingest
	uv run dvc add data/raw

data-push:  ## Upload tracked data versions to the DagsHub remote
	$(WITH_DAGSHUB) uv run dvc push

data-pull:  ## Download the data version this commit points to
	$(WITH_DAGSHUB) uv run dvc pull

backtest:  ## Rolling-origin backtest on the development months (reports/backtest.json)
	uv run python -m carbon_intensity_forecast.backtest
