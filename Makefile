.PHONY: help setup secrets test data data-push data-pull backtest experiments train registry rollback promote flow export-model serve up down kind-up kind-down monitor drift-history daily

# MLflow sends usage telemetry by default; this project turns it off.
export MLFLOW_DISABLE_TELEMETRY := true
export MLFLOW_DISABLE_AGENT_HINT := 1

# Prefect's temporary local server needs about 23 s to start on this Mac (default limit 20 s).
export PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS := 120

# Secrets live in .env (never committed): DAGSHUB_USER and DAGSHUB_TOKEN (yours), and the
# local passwords `make secrets` generates. DagsHub's S3 endpoint takes the token as both
# key and secret.
WITH_ENV = set -a; . ./.env; set +a;
WITH_DAGSHUB = $(WITH_ENV) AWS_ACCESS_KEY_ID="$$DAGSHUB_TOKEN" AWS_SECRET_ACCESS_KEY="$$DAGSHUB_TOKEN"
LOCAL_SECRETS = POSTGRES_PASSWORD GRAFANA_DB_PASSWORD GRAFANA_ADMIN_PASSWORD

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies from uv.lock
	uv sync --locked

secrets:  ## Add any missing local passwords to .env (random; existing values are kept, nothing printed)
	@touch .env && chmod 600 .env
	@[ -z "$$(tail -c 1 .env)" ] || echo >> .env   # end the last line before appending
	@for key in $(LOCAL_SECRETS); do \
	  grep -q "^$$key=" .env || { echo "$$key=$$(openssl rand -hex 24)" >> .env && echo "added $$key"; }; \
	done

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

flow:  ## The training pipeline as a Prefect flow: ingest, validate, version, train, register
	uv run python -m carbon_intensity_forecast.flows training

daily:  ## The daily flow: ingest, version, monitor; retrain if triggered (needs `make up`)
	uv run python -m carbon_intensity_forecast.flows daily

up: secrets  ## Start the stack: MLflow :5050, API :8000, Prometheus :9090, Grafana :3000, Prefect :4200 (needs models/champion)
	docker compose up -d --wait postgres
	@# Database roles (the app's password, Grafana's SELECT-only role), applied every time.
	@$(WITH_ENV) docker compose exec -T postgres psql -q -U forecast -d forecast \
	  -v app_password="$$POSTGRES_PASSWORD" -v reader_password="$$GRAFANA_DB_PASSWORD" -f /etc/roles.sql
	docker compose up -d --build --wait
	@# Grafana keeps its first admin password in its volume; align it with .env.
	@$(WITH_ENV) docker compose exec -T grafana grafana cli admin reset-admin-password \
	  "$$GRAFANA_ADMIN_PASSWORD" >/dev/null && echo "Grafana admin password: GRAFANA_ADMIN_PASSWORD in .env"

down:  ## Stop the local stack (data stays in its volumes)
	docker compose down

export-model:  ## Export the champion from MLflow to models/champion, track it with DVC and push it
	uv run python -m carbon_intensity_forecast.export_model
	uv run dvc add models/champion
	$(WITH_DAGSHUB) uv run dvc push models/champion.dvc

serve:  ## Run the API from source on :8001 (the compose API holds :8000; model from the registry)
	MLFLOW_TRACKING_URI=http://127.0.0.1:5050 uv run uvicorn carbon_intensity_forecast.api:app --port 8001

kind-up:  ## Local Kubernetes: create a kind cluster, load the API image, deploy (stop compose first: 4 GB)
	kind create cluster --name carbon --config k8s/kind-config.yaml --wait 180s
	docker build --platform=linux/amd64 -t carbon-forecast-api:dev .
	kind load docker-image carbon-forecast-api:dev --name carbon
	kubectl apply -k k8s/overlays/kind
	kubectl -n carbon-forecast rollout status deployment/forecast-api --timeout=240s

kind-down:  ## Delete the kind cluster
	kind delete cluster --name carbon

monitor:  ## Issue, benchmark and score live forecasts; drift report; retraining decision (needs `make up`)
	uv run python -m carbon_intensity_forecast.monitor

drift-history:  ## How often input drift was flagged in past months, and whether it tracked the error
	uv run python -m carbon_intensity_forecast.drift_history
