# AI-Driven Sales Forecasting & Business Intelligence Dashboard

An interactive dashboard that combines **business intelligence** (KPIs, trends, regional and category analysis)
with **AI-based sales forecasting** (three models compared on a hold-out set).

## Features
- KPI cards: revenue, units, customers, average order value, period growth
- Trend charts (daily / weekly / monthly), category bars, region donut, weekday x month heatmap
- Forecasting with model comparison: Seasonal Naive (baseline), Holt-Winters, Gradient Boosting (lag, rolling and calendar features)
- Metrics: MAE, RMSE, MAPE on a validation window you choose
- Future forecast with an 80% prediction band, downloadable as CSV
- **Segment forecasting:** separate forecasts per category or per region, with a ranked table of expected growth or decline
- **Anomaly detection:** Robust Z-score and Isolation Forest detectors, an adjustable sensitivity slider, and a downloadable list of flagged spike/drop days
- Auto-generated business insights
- Upload your own CSV from the sidebar

## Project structure
```
app.py            Streamlit dashboard (UI)
forecasting.py    Models, feature engineering, evaluation
generate_data.py  Synthetic dataset generator (3 years, 4 regions, 5 categories)
data/sales.csv    Dataset (replace with real data)
requirements.txt
```

## Run
```
pip install -r requirements.txt
python generate_data.py      # optional, data/sales.csv already included
streamlit run app.py
```

## Data format
`date, category, region, units_sold, unit_price, discount_pct, revenue, customers`
(`discount_pct` is optional; at least ~400 days of history are needed for the ML model.)

## How anomaly detection works
- **Robust Z-score:** each day is compared with a 29-day rolling median (so the gradual festive-season rise is not flagged),
  adjusted for weekday effect, then scored using the median absolute deviation. Days beyond the threshold are flagged.
- **Isolation Forest:** looks at revenue, units, customers and average price together, so it can catch unusual combinations.
- The bundled data contains 6 injected events (3 stock-out drops, 3 flash-sale spikes); the Z-score method finds all six.
  A few extra random days may be flagged too; raise the threshold to reduce them.

## How the forecasting works
1. Revenue is aggregated to a daily series for the selected filters.
2. The last N days are held out; each model trains on the rest and predicts those N days.
3. Models are ranked by MAPE. Gradient Boosting uses lags (1, 7, 14, 28, 365), rolling means (7, 28),
   day-of-week, month, festive-season flag and Fourier-style yearly terms, and forecasts recursively.
4. The chosen model is refit on all data to forecast the future horizon.

## Notes and limitations
- The bundled data is **synthetic**, so the accuracy numbers show the pipeline works, not real-world performance.
- Prediction intervals are a simple approximation, not a rigorous probabilistic forecast.
- The festive-season window is hard-coded (15 Oct to 15 Nov); adjust it in `forecasting.py` for your market.

## Future work
Prophet/LSTM models, inventory recommendations, email alerts for anomalies, database connection.
