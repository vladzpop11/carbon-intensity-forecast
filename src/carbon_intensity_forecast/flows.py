"""The pipeline as a Prefect flow: ingest -> validate -> version the data -> train -> evaluate
-> register. Validation is part of ingestion (the data contracts): a batch that breaks them
stops the flow before anything is trained on it.

The data version is DVC's checksum of data/raw, pushed to the DagsHub remote and recorded
on every MLflow run and model version (data_md5), so a model's exact data can be pulled
again by writing that checksum into data/raw.dvc. The flow does not commit to git.

Needs the MLflow server (make up) and DagsHub credentials in .env.

python -m carbon_intensity_forecast.flows [--no-train]
"""

import argparse
import os
import subprocess

import yaml
from prefect import flow, get_run_logger, task

from carbon_intensity_forecast import ingest, train
from carbon_intensity_forecast.config import project_path


def dagshub_env() -> dict:
    """Environment for DVC with the DagsHub token from .env (read, never printed)."""
    values = {}
    for line in project_path(".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip():
            values[key.strip()] = value.strip()
    token = values["DAGSHUB_TOKEN"]
    return {**os.environ, "AWS_ACCESS_KEY_ID": token, "AWS_SECRET_ACCESS_KEY": token}


@task(retries=2, retry_delay_seconds=300)
def ingest_and_validate() -> None:
    """Download, check the contracts, save data/raw (raises if a contract fails)."""
    ingest.main()


@task(retries=2, retry_delay_seconds=60)
def version_data() -> str:
    """Track the new data/raw with DVC and push it to the remote; -> its checksum."""
    cwd = project_path(".")
    subprocess.run(["dvc", "add", "data/raw"], cwd=cwd, check=True, capture_output=True)
    subprocess.run(["dvc", "push"], cwd=cwd, check=True, capture_output=True, env=dagshub_env())
    return yaml.safe_load((cwd / "data/raw.dvc").read_text())["outs"][0]["md5"]


@task
def train_and_register() -> None:
    train.main()


@flow(name="carbon-intensity-pipeline", log_prints=True)
def pipeline(train_model: bool = True) -> None:
    logger = get_run_logger()
    ingest_and_validate()
    md5 = version_data()
    logger.info(f"data version {md5}")
    if train_model:
        train_and_register()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="ingest and version only")
    pipeline(train_model=not parser.parse_args().no_train)
