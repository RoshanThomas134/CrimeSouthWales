# ─── Enhanced UK Crime Dashboard ───

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pydeck as pdk

# ─── CONFIG ───
st.set_page_config(page_title="UK Crime Dashboard", layout="wide")

# ─── LOAD DATA ───
@st.cache_data
def load_data():
    df = pd.read_excel('Cleaned_DA_Assessment_final.xlsb')
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df.dropna(subset=["Month"])

df = load_data()

# ─── SIDEBAR ───
st.sidebar.title("🔍 Filter Crime Data")
available_months = sorted(df["Month"].dt.to_period("M").astype(str).unique())
selected_month = st.sidebar.selectbox("Select Month", available_months)
selected_month_dt = pd.to_datetime(selected_month)

# Get current and previous periods
current_period = selected_month_dt.to_period("M")
previous_period = current_period - 1

filtered_types = df[df["Month"].dt.to_period("M") == current_period]["Crime type"].dropna().unique()
selected_crime_type = st.sidebar.multiselect("Select Crime Type", filtered_types, default=filtered_types)

# ─── FILTER DATA ───
current_df = df[(df["Month"].dt.to_period("M") == current_period) & (df["Crime type"].isin(selected_crime_type))]
previous_df = df[(df["Month"].dt.to_period("M") == previous_period) & (df["Crime type"].isin(selected_crime_type))]

# ─── TITLE ───
st.title(":bar_chart: UK Crime Data Dashboard")
st.markdown("Explore crime trends, hotspots, and predictions with interactive filtering.")

# ─── KPIs ───
def calculate_delta(current, previous):
    if previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 2)

col1, col2, col3 = st.columns(3)

col1.metric("Total Crimes", len(current_df), f"{calculate_delta(len(current_df), len(previous_df)):+}%" if len(previous_df) > 0 else "N/A")
col2.metric("Unique Locations", current_df["Location"].nunique(), f"{calculate_delta(current_df['Location'].nunique(), previous_df['Location'].nunique()):+}%" if len(previous_df) > 0 else "N/A")
col3.metric("Police Forces", current_df["Falls within"].nunique(), f"{calculate_delta(current_df['Falls within'].nunique(), previous_df['Falls within'].nunique()):+}%" if len(previous_df) > 0 else "N/A")

# ─── BAR CHART ───
st.subheader("📊 Crime Count by Police Force and Type")
if not current_df.empty:
    bar_data = current_df.groupby(["Falls within", "Crime type"]).size().reset_index(name="Count")
    fig_bar = px.bar(bar_data, y="Falls within", x="Count", color="Crime type", orientation="h", title="Crime Counts by Police Force and Type")
    fig_bar.update_layout(yaxis_title="Police Force", xaxis_title="Crime Count", barmode="group", height=600)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── CLUSTERING & HEATMAP ───
st.subheader("📍 Crime Clustering and Heatmap")
map_df = current_df.dropna(subset=["Latitude", "Longitude"])
if not map_df.empty:
    coords = map_df[["Latitude", "Longitude"]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
    map_df["Cluster"] = kmeans.fit_predict(coords_scaled)

    fig_clusters = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        zoom=9,
        mapbox_style="carto-positron",
        title="Crime Hotspots - Clustered"
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    st.subheader("🔥 Crime Location Heatmap")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=map_df["Latitude"].mean(),
            longitude=map_df["Longitude"].mean(),
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position='[Longitude, Latitude]',
                radius=200,
                threshold=0.3
            )
        ],
    ))

# ─── TOP HOTSPOTS ───
st.subheader("📍 Top Crime Hotspots")
if not current_df.empty and "Location" in current_df.columns:
    top_locations = current_df["Location"].value_counts().head(10).reset_index()
    top_locations.columns = ["Location", "Count"]
    st.dataframe(top_locations, use_container_width=True)

# ─── CRIME TREND ───
st.subheader("📈 Monthly Crime Trend")
trend_df = df[df["Crime type"].isin(selected_crime_type)]
monthly_trend = trend_df.groupby(df["Month"].dt.to_period("M")).size().reset_index(name="Total Crimes")
monthly_trend["Month"] = monthly_trend["Month"].astype(str)

fig_trend = px.line(monthly_trend, x="Month", y="Total Crimes", markers=True, title="Crime Trend Over Time")
fig_trend.update_xaxes(tickangle=45)
st.plotly_chart(fig_trend, use_container_width=True)

# ─── FORECASTING ───
st.subheader("🔮 Crime Forecast (Next 6 Months)")
forecast_data = trend_df.groupby("Month").size().reset_index(name="Crime Count")
forecast_data["Day_Index"] = (forecast_data["Month"] - forecast_data["Month"].min()).dt.days

X = forecast_data["Day_Index"].values.reshape(-1, 1)
y = forecast_data["Crime Count"].values
model = LinearRegression().fit(X, y)

future_months = pd.date_range(start=forecast_data["Month"].max() + pd.offsets.MonthBegin(), periods=6, freq='MS')
future_index = (future_months - forecast_data["Month"].min()).days.values.reshape(-1, 1)
future_preds = model.predict(future_index)

forecast_df = pd.DataFrame({"Month": future_months, "Predicted Crimes": future_preds.astype(int)})
combined_df = pd.concat([
    forecast_data[["Month", "Crime Count"]],
    forecast_df.rename(columns={"Predicted Crimes": "Crime Count"})
])

fig_forecast = px.line(combined_df, x="Month", y="Crime Count", title="Crime Forecast: Next 6 Months")
fig_forecast.add_scatter(x=future_months, y=future_preds, mode='lines+markers', name='Forecast', line=dict(color='red'))
st.plotly_chart(fig_forecast, use_container_width=True)

# ─── EXPORT ───
st.subheader("📄 Download Forecast Report")
report = forecast_df.copy()
report["Month"] = report["Month"].dt.strftime('%Y-%m')
st.download_button("Download CSV", report.to_csv(index=False), file_name="crime_forecast.csv")

# ─── FOOTER ───
st.markdown("---")
st.caption("Developed by Roshan Thomas | Enhanced with KPIs and Forecasting | Streamlit + Plotly + scikit-learn")
