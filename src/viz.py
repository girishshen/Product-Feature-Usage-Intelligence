import pandas as pd
import plotly.express as px


def feature_usage_bar(df_events: pd.DataFrame):
    """
    Total events per feature (bar chart).
    Expects: feature_name, events_count
    """
    agg = (
        df_events.groupby("feature_name")["events_count"]
        .sum()
        .reset_index()
        .sort_values("events_count", ascending=False)
    )

    fig = px.bar(
        agg,
        x="feature_name",
        y="events_count",
        title="Total Feature Usage",
        labels={"feature_name": "Feature", "events_count": "Total Events"},
        text="events_count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(showgrid=False))
    return fig


def feature_usage_timeseries(df_events: pd.DataFrame, feature_filter=None):
    """
    Daily events over time, optionally filtered by feature.
    """
    df = df_events.copy()
    if feature_filter:
        df = df[df["feature_name"].isin(feature_filter)]

    daily = (
        df.groupby(["event_date", "feature_name"])["events_count"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily,
        x="event_date",
        y="events_count",
        color="feature_name",
        title="Feature Usage Over Time",
        labels={"event_date": "Date", "events_count": "Total Events", "feature_name": "Feature"},
    )
    fig.update_layout(xaxis_rangeslider_visible=True)
    return fig


def rfm_cluster_3d(rfm: pd.DataFrame):
    """
    3D scatter plot of RFM clusters.
    Expects: recency, frequency, monetary, cluster
    """
    fig = px.scatter_3d(
        rfm,
        x="recency",
        y="frequency",
        z="monetary",
        color="cluster",
        title="RFM Segments (3D View by Cluster)",
        labels={
            "recency": "Recency (days since last use)",
            "frequency": "Frequency (active days)",
            "monetary": "Monetary (total events)",
            "cluster": "Cluster",
        },
    )
    return fig


def cluster_distribution_bar(rfm: pd.DataFrame):
    """
    Bar chart of user count per cluster.
    """
    agg = rfm["cluster"].value_counts().reset_index()
    agg.columns = ["cluster", "user_count"]
    fig = px.bar(
        agg,
        x="cluster",
        y="user_count",
        title="Users per Cluster",
        text="user_count",
        labels={"cluster": "Cluster", "user_count": "Number of Users"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(showgrid=False))
    return fig