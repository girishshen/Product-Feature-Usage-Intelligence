import streamlit as st
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.rfm import load_rfm_with_clusters

CLEAN_PATH = Path("data/processed/clean_feature_usage.csv")

@st.cache_data
def load_clean_and_rfm():
    clean = pd.read_csv(CLEAN_PATH)
    rfm = load_rfm_with_clusters()
    return clean, rfm


def main():
    st.title("📌 Overview")

    clean, rfm = load_clean_and_rfm()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", clean.shape[0])
    col2.metric("Avg Events per User", f"{clean['total_events'].mean():.2f}")
    col3.metric("Avg Active Days per User", f"{clean['active_days'].mean():.2f}")

    st.markdown("---")

    st.subheader("RFM Summary by Cluster")

    # Simple RFM cluster summary table
    summary = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .round(2)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

    st.markdown(
        """
**How to read this:**  
- Lower *recency* = users are more recent/active  
- Higher *frequency* = more active days  
- Higher *monetary* = more total feature usage  
"""
    )


if __name__ == "__main__":
    main()