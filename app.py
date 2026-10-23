"""AI-Driven Sales Forecasting & Business Intelligence Dashboard
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import (detect_anomalies_iforest, detect_anomalies_zscore, evaluate_models,
                         forecast_by_group, forecast_future)

st.set_page_config(page_title="Sales Forecasting & BI", page_icon="📈", layout="wide")


@st.cache_data
def load(path_or_file):
    df = pd.read_csv(path_or_file, parse_dates=["date"])
    return df


@st.cache_data
def run_eval(series: pd.Series, test_days: int):
    return evaluate_models(series, test_days)


@st.cache_data
def run_forecast(series: pd.Series, horizon: int, model: str):
    return forecast_future(series, horizon, model)


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controls")
up = st.sidebar.file_uploader("Upload sales CSV (optional)", type="csv")
df = load(up if up else "data/sales.csv")
st.sidebar.caption("Required columns: date, category, region, units_sold, unit_price, revenue, customers")

dmin, dmax = df.date.min().date(), df.date.max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
cats = st.sidebar.multiselect("Category", sorted(df.category.unique()), default=sorted(df.category.unique()))
regs = st.sidebar.multiselect("Region", sorted(df.region.unique()), default=sorted(df.region.unique()))

if len(date_range) == 2:
    df = df[(df.date >= pd.Timestamp(date_range[0])) & (df.date <= pd.Timestamp(date_range[1]))]
df = df[df.category.isin(cats) & df.region.isin(regs)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

@st.cache_data
def run_group_forecast(data: pd.DataFrame, group_col: str, horizon: int, model: str):
    return forecast_by_group(data, group_col, horizon, model)


st.title("📈 AI-Driven Sales Forecasting & Business Intelligence Dashboard")

tab1, tab2, tab4, tab5, tab3 = st.tabs(
    ["📊 Business Intelligence", "🔮 AI Forecast", "🗂️ Segment Forecast", "🚨 Anomaly Detection", "💡 Insights"]
)

# ---------------- Tab 1: BI ----------------
with tab1:
    rev, units, cust = df.revenue.sum(), df.units_sold.sum(), df.customers.sum()
    half = df.date.min() + (df.date.max() - df.date.min()) / 2
    prev, curr = df[df.date <= half].revenue.sum(), df[df.date > half].revenue.sum()
    growth = (curr / prev - 1) * 100 if prev else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{rev/1e6:,.2f} M", f"{growth:+.1f}% (2nd half vs 1st)")
    c2.metric("Units Sold", f"{units:,}")
    c3.metric("Customers", f"{cust:,}")
    c4.metric("Avg Order Value", f"₹{rev/max(cust,1):,.0f}")

    freq = st.radio("Trend granularity", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
    trend = df.set_index("date").revenue.resample(rule).sum().reset_index()
    st.plotly_chart(px.line(trend, x="date", y="revenue", title=f"{freq} Revenue Trend"), width="stretch")

    a, b = st.columns(2)
    by_cat = df.groupby("category", as_index=False).revenue.sum().sort_values("revenue", ascending=False)
    a.plotly_chart(px.bar(by_cat, x="category", y="revenue", title="Revenue by Category", color="category"), width="stretch")
    by_reg = df.groupby("region", as_index=False).revenue.sum()
    b.plotly_chart(px.pie(by_reg, names="region", values="revenue", title="Revenue by Region", hole=0.45), width="stretch")

    heat = df.assign(month=df.date.dt.month_name(), dow=df.date.dt.day_name())
    pv = heat.pivot_table(index="dow", columns="month", values="revenue", aggfunc="sum")
    order_d = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    order_m = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    pv = pv.reindex(index=order_d, columns=[m for m in order_m if m in pv.columns])
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="Blues", title="Revenue Heatmap: Weekday × Month"), width="stretch")

# ---------------- Tab 2: Forecast ----------------
with tab2:
    series = df.groupby("date").revenue.sum().asfreq("D").fillna(0)
    if len(series) < 400:
        st.warning("Need at least ~400 days of data for the ML model (365-day lag). Widen the date range.")
        st.stop()

    cA, cB = st.columns(2)
    horizon = cA.slider("Forecast horizon (days)", 7, 90, 30)
    test_days = cB.slider("Validation window (days)", 30, 120, 60)

    with st.spinner("Training and comparing models…"):
        table, preds, test, imp = run_eval(series, test_days)

    st.subheader("Model comparison (hold-out validation)")
    st.dataframe(table.style.highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#c6f6d5"), width="stretch")
    best = table.index[0]
    st.success(f"Best model on validation: **{best}** (MAPE {table.loc[best, 'MAPE %']:.1f}%)")

    model_choice = st.selectbox("Model used for the future forecast", list(table.index), index=0)
    fc = run_forecast(series, horizon, model_choice)

    hist = series.iloc[-120:]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#2b6cb0"))
    fig.add_scatter(x=fc.index, y=fc.forecast, name="Forecast", line=dict(color="#dd6b20"))
    fig.add_scatter(x=fc.index, y=fc.upper, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=fc.index, y=fc.lower, fill="tonexty", fillcolor="rgba(221,107,32,0.2)", line=dict(width=0), name="80% interval")
    fig.update_layout(title=f"{horizon}-day Revenue Forecast ({model_choice})", yaxis_title="Revenue")
    st.plotly_chart(fig, width="stretch")

    k1, k2 = st.columns(2)
    k1.metric(f"Expected revenue, next {horizon} days", f"₹{fc.forecast.sum()/1e6:,.2f} M")
    k2.metric("Avg daily forecast", f"₹{fc.forecast.mean():,.0f}")

    with st.expander("Validation plot & feature importance"):
        v = go.Figure()
        v.add_scatter(x=test.index, y=test.values, name="Actual", line=dict(color="black"))
        for name, p in preds.items():
            v.add_scatter(x=test.index, y=p, name=name)
        st.plotly_chart(v, width="stretch")
        st.bar_chart(imp.head(10))

    st.download_button("⬇️ Download forecast CSV", fc.round(2).to_csv().encode(), "forecast.csv", "text/csv")

# ---------------- Tab 3: Insights ----------------
with tab3:
    st.subheader("Auto-generated business insights")
    ins = []
    top = by_cat.iloc[0]
    ins.append(f"**{top.category}** is the top category, contributing {top.revenue/rev*100:.1f}% of revenue.")
    r = df.groupby("region").revenue.sum()
    ins.append(f"**{r.idxmax()}** is the strongest region ({r.max()/rev*100:.1f}%); **{r.idxmin()}** is the weakest ({r.min()/rev*100:.1f}%).")
    wk = df.groupby(df.date.dt.dayofweek).revenue.mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ins.append(f"Best weekday on average is **{names[wk.idxmax()]}**, weakest is **{names[wk.idxmin()]}**.")
    mo = df.groupby(df.date.dt.month).revenue.sum()
    ins.append(f"Peak month is **{pd.Timestamp(2000, mo.idxmax(), 1).strftime('%B')}**, lowest is **{pd.Timestamp(2000, mo.idxmin(), 1).strftime('%B')}**.")
    if "discount_pct" in df.columns:
        promo = df[df.discount_pct > 0].groupby("date").revenue.sum().mean()
        nopromo = df[df.discount_pct == 0].groupby("date").revenue.sum().mean()
        if pd.notna(promo) and nopromo:
            ins.append(f"Days with active promotions show {(promo/nopromo-1)*100:+.1f}% revenue vs non-promo days (correlation, not proof of causation).")
    ins.append(f"Overall growth between first and second half of the selected period: **{growth:+.1f}%**.")
    for i in ins:
        st.markdown(f"- {i}")

    st.subheader("Top 10 category-region combinations")
    combo = df.groupby(["category", "region"], as_index=False).revenue.sum().sort_values("revenue", ascending=False).head(10)
    st.dataframe(combo, width="stretch", hide_index=True)


# ---------------- Tab 4: Per-category / per-region forecast ----------------
with tab4:
    st.subheader("Forecast every segment separately")
    g1, g2, g3 = st.columns(3)
    group_col = g1.selectbox("Segment by", ["category", "region"])
    seg_h = g2.slider("Horizon (days)", 7, 90, 30, key="seg_h")
    seg_model = g3.selectbox("Model", ["Gradient Boosting", "Holt-Winters", "Seasonal Naive"], key="seg_m")

    with st.spinner(f"Forecasting each {group_col}…"):
        wide, summ = run_group_forecast(df, group_col, seg_h, seg_model)

    if wide.empty:
        st.warning("Not enough history (~400 days) in the selected range.")
    else:
        long = wide.reset_index().melt(id_vars="index", var_name=group_col, value_name="forecast")
        st.plotly_chart(
            px.line(long, x="index", y="forecast", color=group_col, title=f"Daily revenue forecast by {group_col}",
                    labels={"index": "date"}),
            width="stretch",
        )
        st.dataframe(
            summ.style.format({c: "{:,.0f}" for c in summ.columns[1:3]} | {"Change %": "{:+.1f}%"}),
            width="stretch", hide_index=True,
        )
        gain, loss = summ.iloc[summ["Change %"].argmax()], summ.iloc[summ["Change %"].argmin()]
        st.info(
            f"Fastest expected growth: **{gain[group_col]}** ({gain['Change %']:+.1f}%). "
            f"Weakest outlook: **{loss[group_col]}** ({loss['Change %']:+.1f}%) versus the previous {seg_h} days."
        )
        st.download_button("⬇️ Download segment forecast CSV", wide.round(2).to_csv().encode(), "segment_forecast.csv", "text/csv")

# ---------------- Tab 5: Anomaly detection ----------------
with tab5:
    st.subheader("Detect unusual sales days")
    method = st.radio("Method", ["Robust Z-score (revenue)", "Isolation Forest (revenue + units + customers + price)"], horizontal=True)
    daily = df.groupby("date").agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"), customers=("customers", "sum"))
    daily = daily.asfreq("D").fillna(0)

    if method.startswith("Robust"):
        thr = st.slider("Sensitivity threshold (lower = more alerts)", 2.5, 8.0, 3.5, 0.5)
        res = detect_anomalies_zscore(daily.revenue, thr)
    else:
        cont = st.slider("Expected share of anomalous days (%)", 0.2, 5.0, 1.0, 0.2) / 100
        res = detect_anomalies_iforest(daily, cont)
        res["expected"] = res["revenue"].rolling(29, center=True, min_periods=14).median()
        res["type"] = res.apply(lambda r: ("Spike" if r.revenue > r.expected else "Drop") if r.anomaly else "", axis=1)

    flagged = res[res.anomaly]
    fig = go.Figure()
    fig.add_scatter(x=res.index, y=res.revenue, name="Daily revenue", line=dict(color="#2b6cb0", width=1))
    fig.add_scatter(x=res.index, y=res.expected, name="Expected", line=dict(color="gray", dash="dot", width=1))
    for t, color in [("Spike", "#38a169"), ("Drop", "#e53e3e")]:
        p = flagged[flagged.type == t]
        fig.add_scatter(x=p.index, y=p.revenue, mode="markers", name=t, marker=dict(color=color, size=10, symbol="diamond"))
    fig.update_layout(title="Daily revenue with detected anomalies", yaxis_title="Revenue")
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalous days", len(flagged))
    m2.metric("Spikes", int((flagged.type == "Spike").sum()))
    m3.metric("Drops", int((flagged.type == "Drop").sum()))

    if len(flagged):
        tbl = flagged[["revenue", "expected", "type"]].copy()
        tbl["deviation %"] = (tbl.revenue / tbl.expected - 1) * 100
        tbl = tbl.sort_values("deviation %", key=abs, ascending=False).reset_index().rename(columns={"index": "date"})
        tbl["date"] = tbl["date"].dt.date
        st.dataframe(tbl.style.format({"revenue": "{:,.0f}", "expected": "{:,.0f}", "deviation %": "{:+.0f}%"}),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ Download anomalies CSV", tbl.to_csv(index=False).encode(), "anomalies.csv", "text/csv")
        st.caption("Large drops often mean stock-outs, outages or data gaps; spikes often mean flash sales or viral demand. "
                   "Always verify flagged days against business events before acting.")
    else:
        st.success("No anomalies at this sensitivity.")
