"""Generate a realistic synthetic retail sales dataset (3 years, daily).
Run:  python generate_data.py
Replace data/sales.csv with your own file (same columns) to use real data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

categories = {
    "Electronics": {"base": 120, "price": 450, "season": "festive"},
    "Clothing":    {"base": 200, "price": 60,  "season": "festive"},
    "Groceries":   {"base": 400, "price": 12,  "season": "flat"},
    "Home & Kitchen": {"base": 90, "price": 85, "season": "summer"},
    "Sports":      {"base": 60,  "price": 70,  "season": "summer"},
}
regions = {"North": 1.0, "South": 0.9, "East": 0.7, "West": 1.2}

# Indian festive windows (approx.) that lift demand
festive_days = set()
for y in (2022, 2023, 2024):
    for d in pd.date_range(f"{y}-10-15", f"{y}-11-15"):
        festive_days.add(d)

rows = []
for cat, cfg in categories.items():
    for reg, rmult in regions.items():
        t = np.arange(len(dates))
        trend = 1 + 0.0004 * t                                   # slow growth
        weekly = 1 + 0.12 * np.isin(dates.dayofweek, [5, 6])     # weekend lift
        doy = dates.dayofyear.values
        if cfg["season"] == "summer":
            seas = 1 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)
        elif cfg["season"] == "festive":
            seas = 1 + 0.10 * np.sin(2 * np.pi * (doy - 200) / 365)
        else:
            seas = np.ones(len(dates))
        fest = np.array([1.45 if d in festive_days else 1.0 for d in dates])
        if cfg["season"] == "flat":
            fest = 1 + (fest - 1) * 0.3
        promo = rng.random(len(dates)) < 0.08                    # 8% promo days
        promo_lift = np.where(promo, rng.uniform(1.15, 1.40, len(dates)), 1.0)
        discount = np.where(promo, rng.choice([10, 15, 20, 25], len(dates)), 0)
        mean_units = cfg["base"] * rmult * trend * weekly * seas * fest * promo_lift / 10
        units = rng.poisson(np.maximum(mean_units, 1))
        price = cfg["price"] * (1 - discount / 100) * rng.normal(1, 0.02, len(dates))
        rows.append(pd.DataFrame({
            "date": dates, "category": cat, "region": reg,
            "units_sold": units,
            "unit_price": price.round(2),
            "discount_pct": discount,
            "revenue": (units * price).round(2),
            "customers": (units * rng.uniform(0.55, 0.8, len(dates))).astype(int),
        }))

df = pd.concat(rows, ignore_index=True)

# --- Inject a few known anomalies so the anomaly detector has something to find ---
# (stock-outs = sudden drops, viral/flash-sale events = sudden spikes)
events = [
    ("2022-05-17", 0.25), ("2022-09-09", 2.6), ("2023-02-14", 0.30),
    ("2023-07-21", 2.8), ("2024-03-05", 0.20), ("2024-08-12", 2.9),
]
for day, factor in events:
    m = df["date"] == day                      # affects all categories/regions that day
    for col in ("units_sold", "customers"):
        df.loc[m, col] = (df.loc[m, col] * factor).round().astype(int)
    df.loc[m, "revenue"] = (df.loc[m, "revenue"] * factor).round(2)
df.to_csv("data/sales.csv", index=False)
print(f"Saved data/sales.csv  ->  {len(df):,} rows, {df.date.min().date()} to {df.date.max().date()}")
