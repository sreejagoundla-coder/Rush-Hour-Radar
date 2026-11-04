# Dashboard data

Files exported by notebook section 13 (Drive folder `rush-hour-radar/dashboard_data/`).
Put them in this folder:

- `dashboard_predictions.parquet` (required, committed to git; about 1-2 MB)
- `backtest_metrics.csv` (Backtest results tab)
- `backtest_coverage.csv` (80% band coverage table)
- `zone_names.csv` (optional, shows zone names instead of IDs)

Then run `streamlit run app.py` from the repository root.
