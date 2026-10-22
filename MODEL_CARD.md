# Model card: day-ahead carbon intensity forecast

One page on what the model is for, what it learned from, how well it does, and where it fails. The evidence for every number is in the [README](README.md) and [reports/](reports/).

## The model

- **Task:** at 09:00 UTC, forecast the carbon intensity of Great Britain's electricity (gCO₂/kWh) for each of the next day's 48 half hours, 15 to 38.5 hours ahead.
- **Model:** LightGBM regression with an L1 objective; one row per (issue day, target half hour), the lead as a feature. 50 inputs: the calendar, carbon intensity published by 09:00, and day-ahead weather forecasts (wind at 100 m, solar radiation, temperature) at ten points.
- **Served** from the MLflow registry (`carbon-intensity-day-ahead@champion`) by a FastAPI service, and exported into the API image for Kubernetes. Stored in LightGBM's text format, not as a pickle.

## Intended use

- **Suitable for:** planning flexible electricity use a day ahead (when to run tomorrow), and as a teaching example of an honestly evaluated, monitored forecasting system.
- **Not suitable for:** grid operation or trading decisions; carbon accounting (it forecasts, it does not measure); regional or sub-half-hour decisions; any use needing a stated uncertainty per half hour (it gives point forecasts only).

## Training data

| | Source | Licence | Period |
| --- | --- | --- | --- |
| Carbon intensity (target and lags) | NESO Carbon Intensity API | CC BY 4.0 | since 2018; modelling from March 2024 |
| Weather forecasts, one and two days ahead | Open-Meteo Previous Runs API | CC BY 4.0 | from 8 March 2024 |

Data contracts set impossible values to missing (zeros during estimation outages, forecasts up to 13,579) and refuse broken batches. Features use only what was published by 09:00 (README, How it avoids hindsight).

## Results

Mean absolute error in gCO₂/kWh, rolling origin, 95% intervals resampling whole days.

| | Model | Same half hour, latest day | A week before | Skill |
| --- | --- | --- | --- | --- |
| Final test, Jan–Sep 2026 (run once) | **21.12** (19.89–22.40) | 43.90 | 50.87 | 51.9% (47.5–56.1%) |
| Development, Jan–Sep 2025 | 19.58 (18.53–20.73) | | | 54.4% |
| Development, Oct 2024–Dec 2025 | 20.65 (19.71–21.62) | 46.27 | 56.85 | 55.4% |

- **Weather is most of the skill:** without it the development error is 43.25.
- **Hardest month:** June 2026 (29.5): biased low for weeks (mean error −27), skill 28%.
- **Against NESO's own day-ahead forecast:** live scoring from 23 October 2026; no result yet.

## Monitoring and retraining

Every live forecast is scored the day after, next to NESO's archived day-ahead forecast and the baseline (Grafana, "Forecast quality"). Retraining starts when the error over the last 14 scored days exceeds 26.8 (1.3 × development) or the model is more than 28 days old; a new model is served only if it beats the baseline by 30% and the current champion. Input drift is reported, not used as a trigger: over 18 past months it was flagged in half or more of them without tracking the error.

## Limitations

- No live comparison with NESO yet; one nine-month final test.
- Actual values are revised after publication, and the weather history's availability was checked on two days only: hindsight is limited, not proven absent.
- No inputs for outages, interconnectors or demand; ten weather points for the whole of Great Britain; data from March 2024 only.
- Point forecasts only; national level; UTC days.

## Licence

Code: MIT. Data: NESO Carbon Intensity API and Open-Meteo, both CC BY 4.0, with attribution.
