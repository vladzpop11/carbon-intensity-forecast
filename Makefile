.PHONY: help setup test data data-push data-pull backtest experiments train registry rollback promote flow export-model serve up down kind-up kind-down

# MLflow sends usage telemetry by default; this project turns it off.
export MLFLOW_DISABLE_TELEMETRY := true
export MLFLOW_DISABLE_AGENT_HINT := 1

# Prefect's temporary local server needs about 23 s to start on this Mac (default limit 20 s).
export PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS := 120

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

experiments:  ## Backtest the candidate setups on development months, logged to MLflow (needs `make up`)
	uv run python -m carbon_intensity_forecast.experiments

train:  ## Train, check the promotion gates, register (needs `make up`; reports/training.json)
	uv run python -m carbon_intensity_forecast.train

registry:  ## Show the champion and challenger model versions
	uv run python -m carbon_intensity_forecast.registry status

rollback:  ## Move the champion alias back to the version it replaced
	uv run python -m carbon_intensity_forecast.registry rollback

promote:  ## Make VERSION the champion by hand (recorded on the version): make promote VERSION=3
	uv run python -m carbon_intensity_forecast.registry promote $(VERSION)

flow:  ## The whole pipeline as a Prefect flow: ingest, validate, version, train, register
	uv run python -m carbon_intensity_forecast.flows

up:  ## Start the stack: MLflow :5050, API :8000, Prometheus :9090, Grafana :3000 (needs models/champion)
	docker compose up -d --build --wait

down:  ## Stop the local stack (data stays in its volumes)
	docker compose down

export-model:  ## Export the champion from MLflow to models/champion and track it with DVC
	uv run python -m carbon_intensity_forecast.export_model
	uv run dvc add models/champion

serve:  ## Run the API locally (model from the registry; needs `make up`)
	MLFLOW_TRACKING_URI=http://127.0.0.1:5050 uv run uvicorn carbon_intensity_forecast.api:app --port 8000

kind-up:  ## Local Kubernetes: create a kind cluster, load the API image, deploy (stop compose first: 4 GB)
	kind create cluster --name carbon --config k8s/kind-config.yaml --wait 180s
	docker build --platform=linux/amd64 -t carbon-forecast-api:dev .
	kind load docker-image carbon-forecast-api:dev --name carbon
	kubectl apply -k k8s/overlays/kind
	kubectl -n carbon-forecast rollout status deployment/forecast-api --timeout=240s

kind-down:  ## Delete the kind cluster
	kind delete cluster --name carbon
