# ADR-0008: Point-in-time features and a true 24h-ahead target

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** Ede Szarka
- **Implemented by:** spec 019
- **Supersedes:** ADR-0002 in part (Prophet is removed, not kept as a benchmark)

## Context

- Training built features at an anchor t for the target t + 24h from silver, whose weather gaps were filled from later hours and whose rolling windows include load at t, which arrives ≥ 1.5h later.
- Serving built a different feature set in the notebook for 24 future anchors (run+1…run+24) whose load was never observed, with last-value fallbacks, a one-week-old weather proxy, UTC calendar fields and a hard-coded holiday list. The model was trained on one distribution and served another.
- The promotion gate compared the model with a 24h persistence baseline that assumes zero data latency, while serving falls back to T − 168h.
- The owner's rule: a forecast for tomorrow 10:00 issued today at 10:00 may use only data that has arrived by today 10:00, including weather forecasts.

## Decision

1. A forecast is issued at hour τ for exactly T = τ + 24h; one target per hourly run.
2. One feature function (`src/pit_features.build_features`) for training, evaluation and serving.
3. Load features use only hours ≤ τ − 24h (22h of ingestion slack for 0.16 pp of MAPE).
4. Weather comes from as-issued forecasts: Open-Meteo Previous Runs API, `icon_seamless`, `*_previous_day2`, valid hours ≤ T only.
5. Model selection and promotion use a rolling-origin backtest with a 24h purge; model, weekly naive (T − 168h) and daily naive (T − 48h) are scored on the same rows.
6. Runs carry a `feature_set` tag; readers accept only the current feature set.
7. Prophet is removed.

## Alternatives considered

- **`previous_day1` weather.** Fresher, but for hour T it uses the run initialised up to T − 24h = τ, which is not published at τ: a leak of up to the publication delay.
- **Reconstruct "latest published run at τ" from single runs.** Exact, but the Single Runs API has no history for 2024–2025.
- **Load lag of 2h.** 0.16 pp better in backtest, but with the observed ingestion gaps the model would often lose its inputs and fall back to naive (8.7%).
- **Keep the anchor-based silver features and fix serving only.** Two feature paths would remain; skew would come back with the next change.
- **Keep Prophet as a benchmark with fixed inputs.** More code for a benchmark the naive baselines already provide on the same rows.

## Consequences

- Backtest MAPE 3.95% vs 8.71% for the served fallback; the model is promotable on honest evidence.
- A new bronze table and ingestion step (`bronze_weather_forecast`, `01b_ingest_weather_forecast`), fed by GitHub Actions like the other sources.
- All pre-019 runs are unusable for serving, shadowing and promotion; the first retrain after deploy is a `first_run`.
- Weather forecasts are 48h+ old at training and serving; archiving our own live forecasts could shorten that later.
