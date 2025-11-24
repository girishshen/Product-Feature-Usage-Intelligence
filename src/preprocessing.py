import pandas as pd

def clean_data(df):
    df = df.dropna(subset=["user_id", "event_date", "feature_name"])
    df["events_count"] = df["events_count"].clip(lower=1)
    return df

def aggregate_user_level(df):
    agg = df.groupby("user_id").agg(
        signup_date=("signup_date", "min"),
        last_event_date=("event_date", "max"),
        total_events=("events_count", "sum"),
        active_days=("event_date", "nunique"),
    ).reset_index()

    return agg