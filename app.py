"""Rush Hour Radar: dashboard for NYC taxi demand forecasts.

Replays hourly forecasts (with 80% bands) from a rolling backtest, i.e. months the
model had not been trained on. It is not a live forecast.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rush Hour Radar", page_icon="🚕", layout="wide")

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, RED = "#1f5fa8", "#e67e22", "#c0392b"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "dashboard_predictions.parquet")
    df = df.rename(columns={"PULocationID": "zone"})
    df["hour"] = pd.to_datetime(df["hour"])
    df["date"] = df["hour"].dt.normalize()
    return df


@st.cache_data
def load_csv(name: str):
    path = DATA / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def zone_labels(zone_ids: tuple) -> dict:
    names = load_csv("zone_names.csv")
    labels = {z: f"Zone {z}" for z in zone_ids}
    if names is not None and {"LocationID", "Zone"} <= set(names.columns):
        for _, r in names.iterrows():
            if r["LocationID"] in labels:
                borough = f", {r['Borough']}" if "Borough" in names.columns else ""
                labels[r["LocationID"]] = f"{r['Zone']}{borough} (zone {r['LocationID']})"
    return labels


if not (DATA / "dashboard_predictions.parquet").exists():
    st.error("Missing data/dashboard_predictions.parquet. Run the notebook's export cell "
             "and copy the dashboard_data folder to data/ next to app.py.")
    st.stop()

df = load_predictions()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🚕 Rush Hour Radar")
st.sidebar.caption("Hourly taxi demand forecasts for the 20 busiest NYC zones, with an 80% band. "
                   "Forecasts shown here come from a rolling backtest: each month was predicted "
                   "by a model trained only on earlier data.")

order = df.groupby("zone")["trips"].sum().sort_values(ascending=False).index.tolist()
labels = zone_labels(tuple(order))
zone = st.sidebar.selectbox("Zone", order, format_func=lambda z: labels[z])

dmin, dmax = df["date"].min().date(), df["date"].max().date()
day = st.sidebar.date_input("Date", value=dmax, min_value=dmin, max_value=dmax)
show_band = st.sidebar.checkbox("Show 80% band", value=True)

st.title("Rush Hour Radar")
tab_day, tab_cal, tab_res, tab_about = st.tabs(
    ["Day view", "Calendar view", "Backtest results", "About"])

# ---------------------------------------------------------------- day view
with tab_day:
    d = df[(df["zone"] == zone) & (df["date"] == pd.Timestamp(day))].sort_values("hour")
    st.subheader(f"{labels[zone]}  |  {pd.Timestamp(day):%A, %d %B %Y}")
    if d.empty:
        st.info("No forecasts for this date. Pick a date inside the backtest range "
                f"({dmin} to {dmax}).")
    else:
        actual, pred = d["trips"].sum(), d["pred"].sum()
        err_pct = (actual - pred) / pred * 100 if pred else np.nan
        mae = (d["trips"] - d["pred"]).abs().mean()
        inside = ((d["trips"] >= d["lo"]) & (d["trips"] <= d["hi"])).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual trips (day)", f"{actual:,.0f}")
        c2.metric("Forecast trips (day)", f"{pred:,.0f}", f"{err_pct:+.1f}% actual vs forecast",
                  delta_color="off")
        c3.metric("Average error per hour", f"{mae:.1f} trips")
        c4.metric("Hours inside the 80% band", f"{inside:.0%}")

        lines = d.melt(id_vars=["hour"], value_vars=["trips", "pred"],
                       var_name="series", value_name="value")
        lines["series"] = lines["series"].map({"trips": "Actual", "pred": "Forecast"})
        line = alt.Chart(lines).mark_line(point=True).encode(
            x=alt.X("hour:T", title="Hour of day"),
            y=alt.Y("value:Q", title="Trips per hour"),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["Actual", "Forecast"], range=[BLUE, ORANGE])),
            strokeDash=alt.StrokeDash("series:N", legend=None,
                                      scale=alt.Scale(domain=["Actual", "Forecast"],
                                                      range=[[1, 0], [5, 3]])),
            tooltip=["hour:T", "series:N", alt.Tooltip("value:Q", format=".0f")])
        chart = line
        if show_band:
            band = alt.Chart(d).mark_area(opacity=0.2, color=ORANGE).encode(
                x="hour:T", y=alt.Y("lo:Q", title="Trips per hour"), y2="hi:Q")
            chart = band + line
        st.altair_chart(chart.properties(height=380), width="stretch")
        st.caption("Shaded area: the range in which actual demand should fall about 80% of the time.")

# ---------------------------------------------------------------- calendar view
with tab_cal:
    scope = st.radio("Scope", ["Selected zone", "All zones combined"], horizontal=True)
    src = df if scope == "All zones combined" else df[df["zone"] == zone]
    daily = (src.groupby("date").agg(actual=("trips", "sum"), forecast=("pred", "sum"))
                .reset_index())
    daily["error_pct"] = (daily["actual"] - daily["forecast"]) / daily["forecast"] * 100

    st.subheader("Daily forecast error (actual vs forecast)")
    bars = alt.Chart(daily).mark_bar().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("error_pct:Q", title="Error, % of forecast"),
        color=alt.condition(alt.datum.error_pct > 0, alt.value(BLUE), alt.value(RED)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("actual:Q", format=",.0f"),
                 alt.Tooltip("forecast:Q", format=",.0f"),
                 alt.Tooltip("error_pct:Q", format="+.1f")])
    rules = alt.Chart(pd.DataFrame({"y": [-8, 8]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((bars + rules).properties(height=300), width="stretch")
    st.caption("Dashed lines mark +/-8%. Positive = more trips than forecast.")

    st.subheader("Ten days with the largest error")
    top = (daily.assign(abs_error=daily["error_pct"].abs()).nlargest(10, "abs_error")
                .sort_values("date"))
    st.dataframe(
        top.assign(date=top["date"].dt.strftime("%a %d %b %Y"))[
            ["date", "actual", "forecast", "error_pct"]]
        .rename(columns={"date": "Date", "actual": "Actual", "forecast": "Forecast",
                         "error_pct": "Error %"})
        .style.format({"Actual": "{:,.0f}", "Forecast": "{:,.0f}", "Error %": "{:+.1f}"}),
        hide_index=True, width="stretch")

# ---------------------------------------------------------------- results
with tab_res:
    metrics = load_csv("backtest_metrics.csv")
    coverage = load_csv("backtest_coverage.csv")
    if metrics is None:
        st.info("backtest_metrics.csv not found in data/.")
    else:
        st.subheader("Rolling backtest: error by month")
        st.caption("Each month is predicted by a model trained only on earlier data. "
                   "Lower is better. v3 adds observed hourly weather to v2.")
        pv = metrics.pivot(index="month", columns="model", values="MAE").round(2)
        st.dataframe(pv, width="stretch")
        long = metrics.rename(columns={"MAE": "MAE (trips per zone-hour)"})
        st.altair_chart(
            alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("month:N", title="Test month"),
                y=alt.Y("MAE (trips per zone-hour):Q"),
                color=alt.Color("model:N", title=None),
                tooltip=["month", "model", alt.Tooltip("MAE (trips per zone-hour):Q", format=".2f")]
            ).properties(height=320), width="stretch")
    if coverage is not None:
        st.subheader("80% band: how often actual demand fell inside")
        st.dataframe(coverage.rename(columns={"month": "Month", "coverage": "Coverage",
                                              "mean_width": "Mean band width (trips)"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**What this is.** A replay of out-of-sample hourly forecasts for NYC yellow-taxi pickups in the
20 busiest zones (NYC TLC trip data, 2023 to 2024). Models are LightGBM with lagged demand,
calendar flags and, in v3, observed hourly weather; the band comes from 10th and 90th
percentile models.

**How to read it.** The forecast is made a day ahead from demand at least 24 hours old.
Each month shown was predicted by a model trained only on data before that month.

**Limits.**
- Not a live forecast: it replays stored backtest predictions.
- In the backtest, weather (v3) did not clearly beat v2 (better in 4 of 9 months), so treat it as a tested
  idea rather than an improvement. The forecasts shown here are v3.
- Weather in v3 is *observed* weather. In real use you would feed a weather forecast, so the
  real-world gain would likely be smaller.
- One weather point for the whole city, two years of data, one city.
- Model features and settings were chosen while looking at 2024 results, so no month is a
  perfectly clean test.
- Local events (for example Election Day) are not modeled.
""")
