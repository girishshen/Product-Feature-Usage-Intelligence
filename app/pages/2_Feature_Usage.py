import streamlit as st
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.viz import feature_usage_bar, feature_usage_timeseries

RAW_PATH = Path("data/raw/synthetic_feature_usage.csv")

@st.cache_data
def load_raw():
    return pd.read_csv(RAW_PATH, parse_dates=["signup_date", "event_date"])


def main():
    st.title("📌 Feature Usage")

    df = load_raw()

    # Filters
    st.sidebar.header("Filters")

    all_features = sorted(df["feature_name"].unique().tolist())
    selected_features = st.sidebar.multiselect(
        "Select Features", all_features, default=all_features
    )

    min_date = df["event_date"].min()
    max_date = df["event_date"].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Apply filters
    df_filtered = df.copy()
    if selected_features:
        df_filtered = df_filtered[df_filtered["feature_name"].isin(selected_features)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df_filtered = df_filtered[
            (df_filtered["event_date"] >= pd.to_datetime(start))
            & (df_filtered["event_date"] <= pd.to_datetime(end))
        ]

    st.subheader("Total Usage per Feature")
    fig_bar = feature_usage_bar(df_filtered)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Usage Over Time")
    fig_ts = feature_usage_timeseries(df_filtered, feature_filter=selected_features)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown(
        """
**Tip for interview:**  
You can explain how this page helps Product Managers quickly see which features
are gaining or losing traction over time and by which date ranges.
"""
    )


if __name__ == "__main__":
    main()