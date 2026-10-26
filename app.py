import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Online Retail Analytics", page_icon="🛍️", layout="wide")
BASE = Path(__file__).parent

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

st.title("🛍️ Online Retail Analytics Dashboard")
st.caption("Sales insights • Customer segmentation • 30-day purchase prediction")

clean_path = BASE / "outputs" / "cleaned_online_retail.csv"
segments_path = BASE / "outputs" / "customer_segments.csv"
predictions_path = BASE / "outputs" / "purchase_predictions.csv"
results_path = BASE / "outputs" / "model_results.csv"

missing = [p.name for p in [clean_path, segments_path, predictions_path, results_path] if not p.exists()]
if missing:
    st.error("Required data files are missing: " + ", ".join(missing))
    st.info("Upload the matching CSV files into the outputs/ folder, then rerun the app.")
    st.stop()

sales = load_csv(clean_path)
segments = load_csv(segments_path)
preds = load_csv(predictions_path)
model_results = load_csv(results_path)

sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"], errors="coerce")
sales["Revenue"] = pd.to_numeric(sales["Revenue"], errors="coerce")
sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")
sales["CustomerID"] = pd.to_numeric(sales["CustomerID"], errors="coerce")
sales = sales.dropna(subset=["InvoiceDate", "Revenue", "CustomerID"])
sales["Month"] = sales["InvoiceDate"].dt.to_period("M").astype(str)

st.sidebar.header("Filters")
countries = sorted(sales["Country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries[:min(5, len(countries))])
if not selected_countries:
    selected_countries = countries
filtered = sales[sales["Country"].isin(selected_countries)]

min_date = sales["InvoiceDate"].min().date()
max_date = sales["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("Invoice date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[(filtered["InvoiceDate"].dt.date >= date_range[0]) & (filtered["InvoiceDate"].dt.date <= date_range[1])]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue (£)", f"£{filtered['Revenue'].sum():,.2f}")
c2.metric("Invoices", f"{filtered['InvoiceNo'].nunique():,}")
c3.metric("Customers", f"{filtered['CustomerID'].nunique():,}")
c4.metric("Products", f"{filtered['StockCode'].nunique():,}")

st.subheader("Sales trends")
monthly = filtered.groupby(filtered["InvoiceDate"].dt.to_period("M").astype(str), as_index=False)["Revenue"].sum()
monthly.columns = ["Month", "Revenue"]
if not monthly.empty:
    st.line_chart(monthly.set_index("Month")["Revenue"])
else:
    st.info("No sales data for the selected filters.")

left, right = st.columns(2)
with left:
    st.subheader("Top products by quantity")
    products = filtered.groupby("Description", dropna=True)["Quantity"].sum().nlargest(10).sort_values()
    st.bar_chart(products)
with right:
    st.subheader("Revenue by country")
    country_rev = filtered.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
    st.bar_chart(country_rev)

st.subheader("Customer segments")
if "Segment" in segments.columns:
    segment_counts = segments["Segment"].value_counts()
    st.bar_chart(segment_counts)
    st.dataframe(segments.head(100), use_container_width=True)
else:
    st.warning("Segment column not found in customer_segments.csv.")

st.subheader("30-day purchase prediction")
if "RandomForestProbability" in preds.columns:
    preds["RandomForestProbability"] = pd.to_numeric(preds["RandomForestProbability"], errors="coerce")
    if "SnapshotDate" in preds.columns:
        preds["SnapshotDate"] = pd.to_datetime(preds["SnapshotDate"], errors="coerce")
    snapshots = sorted(preds["SnapshotDate"].dropna().dt.strftime("%Y-%m-%d").unique()) if "SnapshotDate" in preds.columns else []
    if snapshots:
        chosen_snapshot = st.selectbox("Prediction snapshot date", snapshots, index=len(snapshots)-1)
        view = preds[preds["SnapshotDate"].dt.strftime("%Y-%m-%d") == chosen_snapshot].copy()
    else:
        view = preds.copy()
    view = view.sort_values("RandomForestProbability", ascending=False)
    view["Predicted return probability"] = (view["RandomForestProbability"] * 100).round(1)
    st.dataframe(view.head(100), use_container_width=True)
    st.download_button("Download prediction table", view.to_csv(index=False).encode("utf-8"), "purchase_predictions_filtered.csv", "text/csv")
else:
    st.warning("RandomForestProbability column not found in purchase_predictions.csv.")

st.subheader("Model evaluation")
st.dataframe(model_results.round(3), use_container_width=True)

st.subheader("Business recommendations")
total_revenue = filtered["Revenue"].sum()
top_country = filtered.groupby("Country")["Revenue"].sum().idxmax() if not filtered.empty else "N/A"
if "Segment" in segments.columns:
    inactive_count = int((segments["Segment"].astype(str).str.contains("Inactive|At-Risk", case=False, regex=True)).sum())
else:
    inactive_count = 0
st.markdown(f"""
- **Sales focus:** Use the monthly trend to plan inventory and promotions around stronger sales periods.
- **Geographic focus:** The highest-revenue country in the current filters is **{top_country}**. Validate profitability before increasing spend there.
- **Customer retention:** The segmentation file identifies **{inactive_count:,}** customers labelled inactive/at-risk. Consider a targeted re-engagement campaign.
- **Prediction use:** Treat predicted probabilities as decision support, not certainty. Test campaigns and monitor actual outcomes.
""")
st.caption("Dataset: UCI Online Retail. Historical data only; model performance may change on new customers or periods.")
