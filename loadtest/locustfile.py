"""Load test of the forecast API (make loadtest): users who read the forecast, and a health
check now and then, as a dashboard or a scheduling client would."""

from locust import HttpUser, between, task


class ForecastReader(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def forecast(self) -> None:
        with self.client.get("/forecast", catch_response=True) as response:
            if response.status_code == 200 and len(response.json()["half_hours"]) != 48:
                response.failure("not 48 half hours")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
