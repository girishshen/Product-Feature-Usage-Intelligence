import streamlit as st
from pathlib import Path
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.rfm import load_rfm_with_clusters
from src.viz import rfm_cluster_3d, cluster_distribution_bar

def main():
    st.title("📌 RFM Segments")

    rfm = load_rfm_with_clusters()

    st.sidebar.header("Cluster Filter")
    clusters = sorted(rfm["cluster"].unique().tolist())
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters", clusters, default=clusters
    )

    rfm_filtered = rfm[rfm["cluster"].isin(selected_clusters)]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("RFM 3D Scatter")
        fig_3d = rfm_cluster_3d(rfm_filtered)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Cluster Size")
        fig_bar = cluster_distribution_bar(rfm_filtered)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
**How to talk about this in interviews:**  

- Each point is a user, plotted by Recency, Frequency, and Monetary value.  
- Clusters group users with similar behavior.  
- This kind of view helps identify **power users**, **regulars**, and potential **at-risk** users.
"""
    )


if __name__ == "__main__":
    main()