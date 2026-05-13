import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Advanced Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #0E1117;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 1rem;
            background-color: #1E1E1E;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Dashboard Settings")

forecast_period = st.sidebar.slider(
    "Forecast Days",
    min_value=30,
    max_value=365,
    value=365,
)

# -----------------------------
# Generate Sample Sales Data
# -----------------------------
np.random.seed(42)

dates = pd.date_range(start="2022-01-01", end="2025-12-31", freq="D")

sales = (
    200
    + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 40
    + np.random.normal(0, 12, len(dates))
    + np.linspace(0, 60, len(dates))
)

regions = np.random.choice(["North", "South", "East", "West"], len(dates))
products = np.random.choice(["Electronics", "Furniture", "Clothing", "Food"], len(dates))

sales_df = pd.DataFrame(
    {
        "Date": dates,
        "Sales": sales,
        "Region": regions,
        "Product": products,
    }
)

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

selected_region = st.sidebar.multiselect(
    "Select Region",
    options=sales_df["Region"].unique(),
    default=sales_df["Region"].unique(),
)

selected_product = st.sidebar.multiselect(
    "Select Product",
    options=sales_df["Product"].unique(),
    default=sales_df["Product"].unique(),
)

filtered_df = sales_df[
    (sales_df["Region"].isin(selected_region))
    & (sales_df["Product"].isin(selected_product))
]

# -----------------------------
# Dashboard Header
# -----------------------------
st.title("📈 Advanced Sales Analytics Dashboard")
st.markdown("Real-time business insights with AI-powered 1-Year Forecasting")

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Revenue",
        f"${filtered_df['Sales'].sum():,.0f}",
        "+12.4%",
    )

with col2:
    st.metric(
        "Average Daily Sales",
        f"${filtered_df['Sales'].mean():.0f}",
        "+4.8%",
    )

with col3:
    st.metric(
        "Best Region",
        filtered_df.groupby("Region")["Sales"].sum().idxmax(),
    )

with col4:
    st.metric(
        "Best Product",
        filtered_df.groupby("Product")["Sales"].sum().idxmax(),
    )

# -----------------------------
# Sales Trend Chart
# -----------------------------
st.subheader("📊 Sales Trend Analysis")

daily_sales = filtered_df.groupby("Date")["Sales"].sum().reset_index()

trend_fig = px.line(
    daily_sales,
    x="Date",
    y="Sales",
    title="Daily Sales Performance",
)

trend_fig.update_layout(template="plotly_dark", height=450)

st.plotly_chart(trend_fig, use_container_width=True)

# -----------------------------
# Region and Product Analysis
# -----------------------------
col5, col6 = st.columns(2)

with col5:
    st.subheader("🌍 Regional Sales Distribution")

    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    region_fig = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        hole=0.5,
    )

    region_fig.update_layout(template="plotly_dark")

    st.plotly_chart(region_fig, use_container_width=True)

with col6:
    st.subheader("🛍️ Product Performance")

    product_sales = (
        filtered_df.groupby("Product")["Sales"]
        .sum()
        .reset_index()
    )

    product_fig = px.bar(
        product_sales,
        x="Product",
        y="Sales",
        text_auto=True,
    )

    product_fig.update_layout(template="plotly_dark")

    st.plotly_chart(product_fig, use_container_width=True)

# -----------------------------
# Forecasting Section
# -----------------------------
st.subheader("🤖 AI-Powered Sales Forecast (1 Year)")

forecast_data = daily_sales.rename(columns={"Date": "ds", "Sales": "y"})

# Prophet Model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

model.fit(forecast_data)

future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)

# Forecast Chart
forecast_fig = go.Figure()

forecast_fig.add_trace(
    go.Scatter(
        x=forecast_data["ds"],
        y=forecast_data["y"],
        mode="lines",
        name="Actual Sales",
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecast",
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='Confidence Interval',
    )
)

forecast_fig.update_layout(
    template="plotly_dark",
    title="Sales Forecast for Next 12 Months",
    height=550,
)

st.plotly_chart(forecast_fig, use_container_width=True)

# -----------------------------
# Forecast Insights
# -----------------------------
future_forecast = forecast.tail(forecast_period)

expected_sales = future_forecast["yhat"].sum()
peak_day = future_forecast.loc[
    future_forecast["yhat"].idxmax(),
    "ds",
]

low_day = future_forecast.loc[
    future_forecast["yhat"].idxmin(),
    "ds",
]

col7, col8, col9 = st.columns(3)

with col7:
    st.success(
        f"📌 Expected Revenue (Next Year): ${expected_sales:,.0f}"
    )

with col8:
    st.info(
        f"🚀 Highest Predicted Sales Day: {peak_day.strftime('%d %b %Y')}"
    )

with col9:
    st.warning(
        f"📉 Lowest Predicted Sales Day: {low_day.strftime('%d %b %Y')}"
    )

# -----------------------------
# Raw Data Section
# -----------------------------
with st.expander("📄 View Sales Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# Download Forecast
# -----------------------------
forecast_download = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

csv = forecast_download.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv",
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit • Plotly • Prophet • Machine Learning Forecasting")
