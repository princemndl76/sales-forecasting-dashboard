"""Forecasting engine: baseline, Holt-Winters, and ML (Gradient Boosting).
All models work on a daily revenue series and are compared on a hold-out set.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LAGS = [1, 7, 14, 28, 365]
ROLL = [7, 28]


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE %": mape}


# ---------- 1. Seasonal naive baseline ----------
def seasonal_naive(train: pd.Series, horizon: int, season=7):
    last = train.values[-season:]
    return np.array([last[i % season] for i in range(horizon)])


# ---------- 2. Holt-Winters ----------
def holt_winters(train: pd.Series, horizon: int):
    model = ExponentialSmoothing(
        train.values, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit()
    return model.forecast(horizon)


# ---------- 3. Gradient Boosting with engineered features ----------
def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["doy"] = idx.dayofyear
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["festive"] = (((idx.month == 10) & (idx.day >= 15)) | ((idx.month == 11) & (idx.day <= 15))).astype(int)
    f["sin_y"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["cos_y"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return f


def _lag_features(history: pd.Series, idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = _calendar(idx)
    for l in LAGS:
        f[f"lag_{l}"] = history.reindex(idx - pd.Timedelta(days=l)).values
    shifted = history.shift(1)
    for w in ROLL:
        f[f"roll_{w}"] = shifted.rolling(w).mean().reindex(idx).values
    return f


def gradient_boosting(train: pd.Series, horizon: int):
    """Train on lag features, then forecast recursively day by day."""
    feats = _lag_features(train, train.index).dropna()
    y = train.loc[feats.index]
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42
    ).fit(feats, y)

    hist = train.copy()
    future_idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = []
    for d in future_idx:
        hist.loc[d] = np.nan
        row = _lag_features(hist, pd.DatetimeIndex([d]))
        # rolling stats must ignore the NaN placeholder for day d
        p = float(model.predict(row.fillna(0))[0])
        hist.loc[d] = p
        preds.append(p)
    importances = pd.Series(model.feature_importances_, index=feats.columns).sort_values(ascending=False)
    return np.array(preds), importances


# ---------- Orchestration ----------
def evaluate_models(series: pd.Series, test_days: int = 60):
    """Hold out the last `test_days`, fit every model, return metrics + predictions."""
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    out = {}
    out["Seasonal Naive"] = seasonal_naive(train, test_days)
    out["Holt-Winters"] = holt_winters(train, test_days)
    gb_pred, imp = gradient_boosting(train, test_days)
    out["Gradient Boosting"] = gb_pred
    table = pd.DataFrame({k: metrics(test.values, v) for k, v in out.items()}).T.round(2)
    return table.sort_values("MAPE %"), out, test, imp


def forecast_future(series: pd.Series, horizon: int, model: str):
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if model == "Seasonal Naive":
        pred = seasonal_naive(series, horizon)
    elif model == "Holt-Winters":
        pred = holt_winters(series, horizon)
    else:
        pred, _ = gradient_boosting(series, horizon)
    # simple residual-based 80% interval from in-sample 7-day naive error
    resid = (series - series.shift(7)).dropna()
    sigma = resid.std() / np.sqrt(2)
    width = 1.28 * sigma * np.sqrt(1 + np.arange(horizon) / 30)
    return pd.DataFrame({"forecast": pred, "lower": np.maximum(pred - width, 0), "upper": pred + width}, index=idx)


# =====================================================================
# Per-category forecasting
# =====================================================================
def forecast_by_group(df: pd.DataFrame, group_col: str, horizon: int, model: str):
    """Forecast daily revenue separately for every value of `group_col`.
    Returns (wide forecast DataFrame, summary DataFrame)."""
    fcs, rows = {}, []
    for g, sub in df.groupby(group_col):
        s = sub.groupby("date").revenue.sum().asfreq("D").fillna(0)
        if len(s) < 400:
            continue
        fc = forecast_future(s, horizon, model)
        fcs[g] = fc["forecast"]
        last = s.iloc[-horizon:].sum()
        nxt = fc["forecast"].sum()
        rows.append({
            group_col: g,
            f"Last {horizon}d revenue": last,
            f"Next {horizon}d forecast": nxt,
            "Change %": (nxt / last - 1) * 100 if last else np.nan,
        })
    wide = pd.DataFrame(fcs)
    summary = pd.DataFrame(rows).sort_values(f"Next {horizon}d forecast", ascending=False)
    return wide, summary


# =====================================================================
# Anomaly detection
# =====================================================================
def detect_anomalies_zscore(series: pd.Series, threshold: float = 3.5, window: int = 29):
    """Robust detector: compares each day with a rolling median (so gradual seasonal
    changes such as the festive season are NOT flagged), corrects for weekday effects,
    then scores the log-ratio with a MAD-based z-score."""
    s = series.clip(lower=1)
    base = s.rolling(window, center=True, min_periods=window // 2).median()
    ratio = np.log(s / base)
    dow_effect = ratio.groupby(s.index.dayofweek).transform("median")
    resid = ratio - dow_effect
    mad = np.median(np.abs(resid - resid.median()))
    z = (resid - resid.median()) / (1.4826 * mad if mad > 0 else 1)
    out = pd.DataFrame({"revenue": series, "expected": base * np.exp(dow_effect), "score": z})
    out["anomaly"] = out["score"].abs() > threshold
    out["type"] = np.where(out["anomaly"], np.where(out["score"] > 0, "Spike", "Drop"), "")
    return out


def detect_anomalies_iforest(daily: pd.DataFrame, contamination: float = 0.01):
    """Multivariate detector (Isolation Forest) on revenue, units, customers and
    average price, after removing each column's weekday-adjusted rolling trend."""
    from sklearn.ensemble import IsolationForest
    feats = pd.DataFrame(index=daily.index)
    for c in ["revenue", "units_sold", "customers"]:
        s = daily[c].clip(lower=1)
        base = s.rolling(29, center=True, min_periods=14).median()
        feats[c] = np.log(s / base)
    feats["avg_price"] = np.log((daily.revenue / daily.units_sold.clip(lower=1)).clip(lower=0.01))
    feats["avg_price"] = feats["avg_price"] - feats["avg_price"].rolling(29, center=True, min_periods=14).median()
    feats = feats.fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42).fit(feats)
    out = pd.DataFrame({"revenue": daily["revenue"]})
    out["score"] = -iso.score_samples(feats)
    out["anomaly"] = iso.predict(feats) == -1
    return out
