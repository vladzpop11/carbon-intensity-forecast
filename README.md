# Carbon Intensity Forecast

Day-ahead forecasts of the carbon intensity of Great Britain's electricity (gCO₂/kWh, every half hour), scored live against the forecast the grid operator (NESO) publishes a day ahead. Built as an end-to-end MLOps pipeline: versioned and validated data, orchestrated training, a model registry with promotion gates, a served API, and monitoring that triggers retraining.

Work in progress.

## Data

- [Carbon Intensity API](https://carbonintensity.org.uk) (NESO), CC BY 4.0: actual carbon intensity every half hour since 2018, and NESO's forecast.
- The API keeps only the latest revision of each forecast, made shortly before each half hour. A scheduled GitHub Action ([archive.yml](.github/workflows/archive.yml)) therefore saves the day-ahead forecast as published every morning, to the `archive` branch. That is the benchmark.
