"""Run the load test (loadtest/locustfile.py) against the API and check it against the target
fixed in configs/config.yaml (loadtest) before the first run.

python -m carbon_intensity_forecast.loadtest [URL]   (default http://127.0.0.1:8000)
-> reports/loadtest.json; exit status 1 if the target is missed
"""

import csv
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.config import load_config, project_path


def main() -> None:
    cfg = load_config()["loadtest"]
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    # The first forecast of the day fetches the sources (seconds); the test measures serving.
    httpx.get(f"{url}/forecast", timeout=120).raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "run"
        subprocess.run(
            ["locust", "-f", str(project_path("loadtest/locustfile.py")), "--headless",
             "--host", url, "--users", str(cfg["users"]), "--spawn-rate", str(cfg["users"]),
             "--run-time", f"{cfg['seconds']}s", "--csv", str(prefix), "--only-summary"],
            check=True, capture_output=True,
        )  # fmt: skip
        with open(f"{prefix}_stats.csv") as f:
            stats = {row["Name"]: row for row in csv.DictReader(f)}
    rows = {name: {"requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
                   "median_ms": float(r["50%"]), "p95_ms": float(r["95%"]),
                   "p99_ms": float(r["99%"]), "requests_per_second": float(r["Requests/s"])}
            for name, r in stats.items()}  # fmt: skip
    forecast = rows["/forecast"]
    passed = forecast["p95_ms"] < cfg["p95_ms_below"] and rows["Aggregated"]["failures"] == 0
    write_json(project_path("reports/loadtest.json"), {
        "run_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "url": url, "client_machine": f"{platform.system()} {platform.machine()}",
        "target": cfg, "passed": passed, "results": rows,
    })  # fmt: skip
    print(f"/forecast: {forecast['requests']} requests, median {forecast['median_ms']:.0f} ms, "
          f"p95 {forecast['p95_ms']:.0f} ms, {forecast['requests_per_second']:.1f}/s; failures "
          f"{rows['Aggregated']['failures']} -> {'PASS' if passed else 'FAIL'}")  # fmt: skip
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
