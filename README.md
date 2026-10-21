# Carbon Intensity Forecast

Day-ahead forecasts of the carbon intensity of Great Britain's electricity (gCO₂/kWh, every half hour), scored live against the forecast the grid operator (NESO) publishes a day ahead. Built as an end-to-end MLOps pipeline: versioned and validated data, orchestrated training, a model registry with promotion gates, a served API, and monitoring that triggers retraining.

Work in progress.

## Data

- [Carbon Intensity API](https://carbonintensity.org.uk) (NESO), CC BY 4.0: actual carbon intensity every half hour since 2018, and NESO's forecast.
- [Open-Meteo](https://open-meteo.com) Previous Runs API, CC BY 4.0: weather forecasts (wind at 100 m, solar radiation, temperature) for ten points that drive GB carbon intensity, as they were forecast one and two days ahead. Archived from March 2024.
- The Carbon Intensity API keeps only the latest revision of each forecast, made shortly before each half hour, so its history is not a day-ahead benchmark. A scheduled GitHub Action ([archive.yml](.github/workflows/archive.yml)) therefore saves, every morning, NESO's forecast for the next day as published, and the weather forecasts the model uses as served at that moment. They go to the `archive` branch. NESO's archived forecast is the benchmark.
- Data contracts ([schemas.py](src/carbon_intensity_forecast/schemas.py)): a batch with broken structure is refused; values outside physical limits are set to missing and counted in [reports/data_quality.json](reports/data_quality.json). The source has some: zeros during estimation outages in 2023 and 2026, and forecasts up to 13,579 gCO₂/kWh in 2018–19.
