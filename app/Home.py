import streamlit as st

st.set_page_config(
    page_title="Product Feature Usage Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Product Feature Usage Intelligence Dashboard")

st.markdown(
    """
Welcome to the **Feature Usage Intelligence Dashboard**.

This app simulates how a SaaS / digital product company can:

- Track which features are most used
- Understand user engagement with RFM (Recency, Frequency, Monetary)
- Segment users into usage-based clusters

Use the pages on the left to explore:

1. **Overview** – High-level KPIs  
2. **Feature Usage** – Which features are used and when  
3. **RFM Segments** – Behavioral clusters of users  

Built with **Python, Pandas, scikit-learn, Plotly, and Streamlit**.
"""
)