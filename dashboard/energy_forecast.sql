-- ============================================================
-- QUERY NAME: Pipeline Health Counter
-- DASHBOARD PANEL: Pipeline Status
-- VISUALIZATION: Counter
-- REFRESH: Auto (1 hour)
-- PURPOSE: Monitor the freshness of data ingestion from ENTSO-E.
-- ============================================================
WITH latest_ingest AS (
  SELECT 
    MAX(fetched_at) AS last_fetch,
    TIMESTAMPDIFF(HOUR, MAX(fetched_at), current_timestamp()) AS hours_since_last_ingest
  FROM workspace.energy_forecasting.bronze_load
)
SELECT 
  hours_since_last_ingest,
  CASE 
    WHEN hours_since_last_ingest < 2 THEN 'HEALTHY'
    WHEN hours_since_last_ingest < 6 THEN 'WARNING'
    ELSE 'CRITICAL' 
  END AS pipeline_status
FROM latest_ingest;


-- ============================================================
-- QUERY NAME: Actual vs Forecast Line Chart (24h model)
-- DASHBOARD PANEL: Actual vs Forecast — 24h Model (Last 7 Days)
-- VISUALIZATION: Line chart, one series per `series`
-- REFRESH: Auto (1 hour)
-- PURPOSE: Visual validation of the served forecast. Seasonal naive is written
--          every hour and the LightGBM run may be scored in shadow (spec 017),
--          so each series says whether it was served.
-- ============================================================
WITH ranked_forecasts AS (
  SELECT 
    f.timestamp,
    f.model_name,
    CONCAT(f.model_name, CASE WHEN COALESCE(f.is_served, true) THEN ' (served)' ELSE ' (shadow)' END) AS series,
    f.predicted_mwh,
    f.model_version,
    f.forecast_run_at,
    s.value_mwh AS actual_mwh,
    ROW_NUMBER() OVER (PARTITION BY f.timestamp, f.model_name ORDER BY f.forecast_run_at DESC) as rnk
  FROM workspace.energy_forecasting.gold_forecasts f
  LEFT JOIN workspace.energy_forecasting.silver_features s ON f.timestamp = s.timestamp
  WHERE f.horizon_hours = 24 
    AND f.model_name IN ('energy_lgbm_24h', 'seasonal_naive')
    AND NOT f.is_backfilled
    AND f.timestamp >= current_timestamp() - INTERVAL 7 DAYS
)
SELECT 
  timestamp,
  series,
  model_name,
  actual_mwh,
  predicted_mwh,
  model_version,
  ABS(actual_mwh - predicted_mwh) AS abs_error_mwh
FROM ranked_forecasts
WHERE rnk = 1
ORDER BY timestamp ASC, model_name ASC;


-- ============================================================
-- QUERY NAME: Live Forecast Error (7-day window)
-- DASHBOARD PANEL: Forecast Accuracy — Served vs Naive vs Shadow
-- VISUALIZATION: Line chart of mape by stream; table for skill and alerts
-- REFRESH: Auto (1 hour)
-- PURPOSE: Written hourly by 09_error_monitor (spec 017). relative_skill =
--          1 - model MAPE / naive MAPE on the same target hours (> 0 beats naive).
-- ============================================================
SELECT
  check_timestamp,
  stream,
  model_name,
  ROUND(mape, 2) AS mape_7d,
  ROUND(naive_mape_paired, 2) AS naive_mape_same_hours,
  ROUND(relative_skill, 3) AS relative_skill,
  n,
  ROUND(coverage, 2) AS coverage,
  ARRAY_JOIN(alerts, ', ') AS alerts
FROM workspace.energy_forecasting.forecast_error_monitor
WHERE window_hours = 168
  AND check_timestamp >= current_timestamp() - INTERVAL 30 DAYS
ORDER BY check_timestamp ASC, stream ASC;


-- ============================================================
-- QUERY NAME: Active Forecast Error Alerts
-- DASHBOARD PANEL: Forecast Error Alerts
-- VISUALIZATION: Counter (alert_count) with the alert list as label
-- REFRESH: Auto (1 hour)
-- PURPOSE: Alerts of the latest check (spec 017 T5). A failing alert also
--          fails the hourly job once, which sends the failure email.
-- ============================================================
SELECT
  check_timestamp,
  model_name AS served_forecaster,  -- latest served row: model or a naive fallback
  SIZE(alerts) AS alert_count,
  ARRAY_JOIN(alerts, ', ') AS alerts
FROM workspace.energy_forecasting.forecast_error_monitor
WHERE window_hours = 168 AND stream = 'served'
ORDER BY check_timestamp DESC
LIMIT 1;


-- ============================================================
-- QUERY NAME: Seasonal-Naive Accuracy History
-- DASHBOARD PANEL: Seasonal Naive — Weekly MAPE since 2024
-- VISUALIZATION: Line chart
-- REFRESH: Auto (1 day)
-- PURPOSE: The floor every model must beat. Includes the naive history
--          reconstructed from silver (spec 017 T7, is_backfilled = true).
-- ============================================================
SELECT
  DATE_TRUNC('week', timestamp) AS week_start,
  AVG(ABS(actual_mwh - predicted_mwh) / NULLIF(actual_mwh, 0) * 100) AS weekly_mape,
  COUNT(*) AS n_observations
FROM workspace.energy_forecasting.gold_forecasts
WHERE model_name = 'seasonal_naive'
  AND horizon_hours = 24
  AND actual_mwh IS NOT NULL
GROUP BY 1
HAVING n_observations >= 24
ORDER BY week_start ASC;


-- ============================================================
-- QUERY NAME: Drift Monitoring Heatmap
-- DASHBOARD PANEL: Feature Drift Scores — Last 30 Days
-- VISUALIZATION: Table
-- REFRESH: Auto (1 hour)
-- PURPOSE: Identifies specific features causing data drift signals. Diagnostic
--          only: since spec 018 data drift never triggers retraining.
-- ============================================================
WITH exploded_drift AS (
  SELECT 
    DATE(check_timestamp) AS drift_date,
    EXPLODE(
      CASE 
        WHEN drifted_features IS NULL OR drifted_features = '' THEN ARRAY('none') 
        ELSE SPLIT(drifted_features, ',') 
      END
    ) AS feature_name,
    drift_score_value_mwh,
    drift_score_temp,
    CAST(retrain_triggered AS INT) as retrain_val
  FROM workspace.energy_forecasting.drift_control
  WHERE check_timestamp >= current_timestamp() - INTERVAL 30 DAYS
)
SELECT 
  drift_date,
  feature_name,
  COUNT(*) AS hours_drifted,
  MAX(drift_score_value_mwh) AS max_drift_score_target,
  MAX(drift_score_temp) AS max_drift_score_temp,
  MAX(retrain_val) AS any_retrain_triggered
FROM exploded_drift
GROUP BY 1, 2
ORDER BY drift_date DESC, hours_drifted DESC;


-- ============================================================
-- QUERY NAME: Model Registry Status Table
-- DASHBOARD PANEL: Model Registry — Current Production Models
-- VISUALIZATION: Table
-- REFRESH: Auto (1 hour)
-- PURPOSE: Audit of active model versions and their age.
-- ============================================================
WITH latest_promotions AS (
  SELECT 
    model_name,
    champion_version,
    challenger_mape,
    promoted_at,
    promotion_reason,
    drift_triggered,
    ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY promoted_at DESC) as rnk
  FROM workspace.energy_forecasting.promotion_log
  -- Note: promotion_log contains the status from the moment of promotion
)
SELECT 
  model_name,
  champion_version AS current_production_version,
  challenger_mape AS current_mape,
  promoted_at AS last_promoted,
  promotion_reason,
  drift_triggered AS was_drift_triggered,
  DATEDIFF(current_timestamp(), promoted_at) AS days_in_production,
  CASE 
    WHEN DATEDIFF(current_timestamp(), promoted_at) > 30 THEN 'STALE — consider retraining'
    WHEN DATEDIFF(current_timestamp(), promoted_at) > 14 THEN 'AGING'
    ELSE 'FRESH' 
  END AS model_age_status
FROM latest_promotions
WHERE rnk = 1
ORDER BY model_name ASC;


-- ============================================================
-- QUERY NAME: Retraining History Timeline
-- DASHBOARD PANEL: Retraining Events — Last 90 Days
-- VISUALIZATION: Table
-- REFRESH: Auto (1 hour)
-- PURPOSE: Full audit trail of all automated MLOps promotion decisions.
-- ============================================================
SELECT 
  promoted_at,
  CASE 
    WHEN first_run THEN '🆕'
    WHEN should_promote THEN '✅'  
    ELSE '⏭️' 
  END AS outcome,
  model_name,
  CASE 
    WHEN first_run THEN 'FIRST RUN'
    WHEN should_promote THEN 'PROMOTED'  
    ELSE 'SKIPPED' 
  END AS decision,
  challenger_version,
  challenger_mape,
  champion_version,
  champion_mape,
  ROUND((champion_mape - challenger_mape) / NULLIF(champion_mape, 0) * 100, 2) AS mape_change_pct,
  promotion_reason,
  drift_triggered,
  drifted_features,
  promoted_by
FROM workspace.energy_forecasting.promotion_log
WHERE promoted_at >= current_timestamp() - INTERVAL 90 DAYS
ORDER BY promoted_at DESC;
