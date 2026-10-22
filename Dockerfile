# syntax=docker/dockerfile:1
# The forecast API. Base images are pinned by digest; bump them deliberately.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336

FROM ${UV_IMAGE} AS uv

# --- build stage: runtime dependencies only: no dependency group (the pipeline and dev
# groups hold DVC, Prefect, Evidently and the test tools) -----------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

# --- pipeline stage (compose `scheduler`): the daily Prefect flow --------------------
# The pipeline dependency group (Prefect, DVC, Evidently; no test tools), git for
# provenance; the code is read from folders mounted at /project, so it matches the checkout.
FROM ${PYTHON_IMAGE} AS pipeline
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /project \
    && useradd --create-home --uid 1000 app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /build
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-default-groups --group pipeline
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/project/src \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /project
USER app
CMD ["python", "-m", "carbon_intensity_forecast.flows", "serve"]

# --- runtime stage (the API; the default build target) --------------------------------
FROM ${PYTHON_IMAGE}
# LightGBM needs the OpenMP runtime; pip is not needed at run time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y -q pip
RUN useradd --create-home --uid 1000 app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_URI=/app/models/champion
WORKDIR /app

COPY --from=build /app /app
COPY configs ./configs
# The exported champion (make export-model, or dvc pull): served when MODEL_URI is not a
# registry alias, e.g. on Kubernetes. Compose points MODEL_URI at the registry instead.
COPY models/champion ./models/champion

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

CMD ["uvicorn", "carbon_intensity_forecast.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--limit-concurrency", "32", "--timeout-keep-alive", "5"]
