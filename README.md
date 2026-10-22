# Carbon Intensity Forecast

[![CI](https://github.com/vladzpop11/carbon-intensity-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/vladzpop11/carbon-intensity-forecast/actions/workflows/ci.yml)

Day-ahead forecasts of the carbon intensity of Great Britain's electricity (gCO₂/kWh, every half hour), built as an end-to-end MLOps pipeline: versioned data with contracts, orchestrated training, a model registry with promotion gates and rollback, and (in progress) a served API with monitoring that triggers retraining. The model is scored live against the forecast the grid operator (NESO) publishes a day ahead.

Work in progress: stages 1 (the reproducible pipeline) and 2 (API, compose, CI/CD, Kubernetes) are done; stage 3 (monitoring and retraining) follows.

## The task

Every day at 09:00 UTC, forecast the 48 half hours of the next day (00:00–23:30 UTC, 15 to 38.5 hours ahead), using only what was known at 09:00. This is the decision a flexible electricity user makes: when to run tomorrow.

## Results

Mean absolute error in gCO₂/kWh (lower is better). Rolling origin: each month is forecast by a model trained only on earlier data.

| | Development, Oct 2024 – Dec 2025 | **Final test, Jan – Sep 2026 (run once)** |
| --- | --- | --- |
| **LightGBM with weather forecasts** | 20.65 | **21.12** |
| Same half hour, latest day known at 09:00 | 46.27 | 43.90 |
| Same half hour a week before | 56.85 | 50.87 |

- **52% less error than the best simple rule** on the final test, and better in every month. Errors are largest in winter and in June 2026 (29.5).
- **The weather forecasts carry the model:** without them the development error doubles (43.25).
- **The final test agrees with development** (21.12 against 20.65): the design choices did not overfit the development months.
- **NESO's own day-ahead forecast is the real benchmark,** and only exists from 22 October 2026 (see Data). Its comparison will be added once weeks of it have accumulated.

## Data

- [Carbon Intensity API](https://carbonintensity.org.uk) (NESO), CC BY 4.0: actual carbon intensity every half hour since 2018.
- [Open-Meteo](https://open-meteo.com) Previous Runs API, CC BY 4.0: weather forecasts (wind at 100 m, solar radiation, temperature) for ten points that drive GB carbon intensity (wind farms, the sunny south, large cities), as forecast one and two days ahead. Available from March 2024, which is where modelling starts.
- **Why NESO's forecast is archived here:** the API keeps only the latest revision of each forecast, made shortly before the half hour (its error is about 10, against 40 for "same time yesterday"), so its history is not a day-ahead benchmark, and NESO publishes no older versions. A scheduled GitHub Action ([archive.yml](.github/workflows/archive.yml)) saves, every morning, NESO's forecast for the next day as published, and the weather forecasts as served at that moment, to the `archive` branch.
- **Data contracts** ([schemas.py](src/carbon_intensity_forecast/schemas.py), Pandera): a batch with broken structure is refused; values outside physical limits are set to missing and counted in [reports/data_quality.json](reports/data_quality.json), and more than 1% of them refuses the batch. The source has some: zeros during estimation outages (2023, 2026) and forecasts up to 13,579 gCO₂/kWh (2018–19).
- **Versioned with DVC** on [DagsHub](https://dagshub.com/vladzpop11/carbon-intensity-forecast); `make data-pull` gets the version a commit points to.

## How it avoids hindsight

A day-ahead forecast is easy to make look better than it is. Each feature is built from what existed at 09:00 ([features.py](src/carbon_intensity_forecast/features.py)):

- **Actual values** only up to an hour before (each half hour is published at its end; 30 minutes' margin).
- **Weather forecasts** at the shortest lead already published. Open-Meteo's "forecast from a day earlier" for a time t is the latest run made at least 24 hours before t; with 6 hours' margin for publication, tomorrow 00:00–03:00 uses day-ahead forecasts and the rest two-day-ahead ones. (The daily archive will check this against what was really served.)
- **A test proves it** ([test_features.py](tests/test_features.py)): changing every value published after 09:00 leaves the features identical, and changing known values does change them.

## How decisions were made

Everything below was written into [configs/config.yaml](configs/config.yaml) and committed before the run it governs:

1. **Evaluation periods:** development months for every choice; the final-test months scored once when the design was frozen; then live scoring.
2. **Experiments** ([experiments.py](src/carbon_intensity_forecast/experiments.py), [reports/experiments.json](reports/experiments.json)): three candidate setups and one ablation, each logged to MLflow with its git commit and DVC data version. Rule: lowest development error, keeping the original unless beaten by more than 0.2. Chosen: the L1 objective (20.65 against 21.00).
3. **The final test** ([reports/final_test.json](reports/final_test.json)) was run once. Its first report scored the baselines as `nan` (a faulty value set to missing left a baseline without input on 2 half hours); the scoring was fixed and the test rerun as a recorded revision, with identical model predictions. Per-row predictions are now saved, so a report can always be rescored without rerunning.

## The pipeline

```
make flow   (Prefect)
  ingest      download both sources, check the contracts       ingest.py, schemas.py
  version     dvc add + push to DagsHub                         flows.py
  train       evaluation twin on all but the last 28 days       train.py
  gates       vs baseline (≥ 30% skill) and vs the champion
  register    refit on all data, MLflow version, alias champion
```

**Promotion gates** ([train.py](src/carbon_intensity_forecast/train.py)): the challenger must beat the baseline by at least 30% and have a strictly lower error than the current champion on the same 28 holdout days. The comparison is kept fair: a champion trained before the window is compared as it is; one that has seen part of the window has its settings retrained on the challenger's data; a routine refresh of the champion's own settings with newer data passes. `make rollback` moves `champion` back to the version it replaced; `make promote VERSION=n` is a manual, recorded override.

## Serving

- **The API** ([api.py](src/carbon_intensity_forecast/api.py), [serving.py](src/carbon_intensity_forecast/serving.py)): `GET /forecast` returns the forecast issued at 09:00 UTC for the next day's 48 half hours. It fetches the live inputs, passes them through the same data contracts and the same feature code as training, and only uses what was published by 09:00, so asking later in the day gives the same answer. Results are cached per model version and issue day. Inputs that fail a contract give a 503, counted in the metrics. `GET /health`, `GET /metrics` (Prometheus).
- **The model** is the registry's champion (`MODEL_URI=models:/carbon-intensity-day-ahead@champion`); the API checks every 10 minutes whether the alias has moved and loads the new champion. Where MLflow is not reachable (Kubernetes, CI) it serves the champion exported into the image (`make export-model`, versioned with DVC).
- **`make up`** starts MLflow (:5050), the API (:8000), Prometheus (:9090) and Grafana (:3000), with Grafana's data source and dashboard provisioned from [monitoring/](monitoring/). Measured: 1.07 GB for the four, within Docker's 4 GB on this Mac.
- **Kubernetes** ([k8s/](k8s/)): a kustomize base (non-root, read-only root filesystem, probes, limits) and a kind overlay. `make kind-up` creates a local cluster and deploys (cluster 54 s, rollout 23 s, 1.36 GB).
- **CI** ([ci.yml](.github/workflows/ci.yml)): lint, tests and a dependency audit; then the image is built with the champion pulled from DagsHub, smoke-tested with a live forecast, deployed to a throwaway kind cluster, and published from `main` to [GHCR](https://github.com/vladzpop11/carbon-intensity-forecast/pkgs/container/carbon-intensity-forecast) (`docker pull ghcr.io/vladzpop11/carbon-intensity-forecast:latest`).

## How to run

Python 3.12 via [uv](https://docs.astral.sh/uv/), Docker, and a `.env` with `DAGSHUB_USER` and `DAGSHUB_TOKEN` (never committed).

```bash
make setup         # dependencies
make data-pull     # the data and the exported champion this commit points to
make up            # MLflow :5050, API :8000, Prometheus :9090, Grafana :3000
make test          # Ruff and pytest
make backtest      # development backtest (reports/backtest.json)
make flow          # the whole pipeline: ingest, validate, version, train, register
make registry      # champion and challenger versions
make export-model  # export the champion for the image (then `make data-push`)
make kind-up       # deploy to a local kind cluster (stop compose first); make kind-down
```

## Mac notes

Built on a 2018 MacBook Air (2 cores, 8 GB, Docker limited to 4 GB):

- **MLflow server:** `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` and `--workers 1` (1.1 GB otherwise; 364 MB with both), host port 5050 (AirPlay holds 5000).
- **Models are logged in LightGBM's native format** (the booster), not as a pickle: loading a model never runs stored code.
- **Prefect 3.9:** 3.8.8 accepts FastAPI 0.159, which breaks its local server.
- **GitHub's scheduler can skip runs** (both slots on 22 October 2026), so the archive job has five idempotent slots.

## Security

- **Secrets:** the DagsHub token lives in `.env` locally (git ignores every `.env` variant) and in the repository secret `DAGSHUB_TOKEN` for CI; DVC receives it only for the command that needs it.
- **Models** are stored in LightGBM's native text format, not as pickles: loading one never runs stored code.
- **The API image** has no development tools, runs as a non-root user, and on Kubernetes with a read-only root filesystem and no Linux capabilities.
- **CI audits** every installed dependency with `pip-audit`. One advisory is accepted: PYSEC-2026-2447 in `diskcache` (no fixed release), used by DVC's local cache. Exploiting it needs write access to that cache as the same user, and DVC is not in the served image.

## Licence

Code: [MIT](LICENSE). Data: NESO Carbon Intensity API and Open-Meteo, both CC BY 4.0.
