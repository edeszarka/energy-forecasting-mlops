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
-- VISUALIZATION: Line chart
-- REFRESH: Auto (1 hour)
-- PURPOSE: Visual validation of the primary operational model's accuracy.
-- ============================================================
WITH ranked_forecasts AS (
  SELECT 
    f.timestamp,
    f.predicted_mwh,
    f.model_version,
    f.forecast_run_at,
    s.value_mwh AS actual_mwh,
    ROW_NUMBER() OVER (PARTITION BY f.timestamp ORDER BY f.forecast_run_at DESC) as rnk
  FROM workspace.energy_forecasting.gold_forecasts f
  LEFT JOIN workspace.energy_forecasting.silver_features s ON f.timestamp = s.timestamp
  WHERE f.horizon_hours = 24 
    AND f.model_name LIKE '%lgbm%'
    AND f.timestamp >= current_timestamp() - INTERVAL 7 DAYS
)
SELECT 
  timestamp,
  actual_mwh,
  predicted_mwh,
  model_version,
  ABS(actual_mwh - predicted_mwh) AS abs_error_mwh
FROM ranked_forecasts
WHERE rnk = 1
ORDER BY timestamp ASC;


-- ============================================================
-- QUERY NAME: 7-Day Forecast Forward View
-- DASHBOARD PANEL: 7-Day Forecast (168h Model)
-- VISUALIZATION: Line chart
-- REFRESH: Auto (1 hour)
-- PURPOSE: Future-looking strategic view of Hungarian energy demand.
-- ============================================================
WITH latest_run AS (
  SELECT MAX(forecast_run_at) as max_run FROM workspace.energy_forecasting.gold_forecasts WHERE horizon_hours = 168
),
forecast_data AS (
  SELECT 
    f.timestamp,
    f.predicted_mwh,
    CASE WHEN f.timestamp > current_timestamp() THEN true ELSE false END AS is_future
  FROM workspace.energy_forecasting.gold_forecasts f
  JOIN latest_run lr ON f.forecast_run_at = lr.max_run
  WHERE f.horizon_hours = 168
)
SELECT 
  fd.timestamp,
  fd.predicted_mwh,
  fd.is_future,
  s.value_mwh AS actual_mwh
FROM forecast_data fd
LEFT JOIN workspace.energy_forecasting.silver_features s ON fd.timestamp = s.timestamp
ORDER BY fd.timestamp ASC;


-- ============================================================
-- QUERY NAME: Rolling MAPE Table
-- DASHBOARD PANEL: Forecast Accuracy — Rolling 7-Day MAPE
-- VISUALIZATION: Line chart
-- REFRESH: Auto (1 hour)
-- PURPOSE: Tracks model performance degradation or improvement over time.
-- ============================================================
WITH daily_metrics AS (
  SELECT 
    DATE_TRUNC('week', timestamp) AS week_start,
    horizon_hours,
    ABS(actual_mwh - predicted_mwh) / NULLIF(actual_mwh, 0) * 100 AS mape_per_row
  FROM workspace.energy_forecasting.gold_forecasts
  WHERE actual_mwh IS NOT NULL
)
SELECT 
  week_start,
  horizon_hours,
  AVG(mape_per_row) AS weekly_mape,
  COUNT(*) AS n_observations,
  5.0 AS target_mape -- 5% benchmark for European load forecasting
FROM daily_metrics
GROUP BY 1, 2
HAVING n_observations >= 24
ORDER BY week_start ASC, horizon_hours ASC;


-- ============================================================
-- QUERY NAME: Drift Monitoring Heatmap
-- DASHBOARD PANEL: Feature Drift Scores — Last 30 Days
-- VISUALIZATION: Table
-- REFRESH: Auto (1 hour)
-- PURPOSE: Identifies specific features causing data drift signals.
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
