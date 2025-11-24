import pandas as pd
from pathlib import Path
import joblib

RFM_PATH = Path("data/processed/rfm_features.csv")
MODEL_PATH = Path("models/kmeans_rfm.pkl")


def compute_rfm(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM metrics from a user-level aggregated dataframe.
    Expects columns: user_id, last_event_date, active_days, total_events
    """
    reference_date = df_users["last_event_date"].max()

    df_users["recency"] = (reference_date - df_users["last_event_date"]).dt.days
    df_users["frequency"] = df_users["active_days"]
    df_users["monetary"] = df_users["total_events"]

    return df_users[["user_id", "recency", "frequency", "monetary"]]


def load_rfm_with_clusters(
    rfm_path: Path = RFM_PATH, model_path: Path = MODEL_PATH
) -> pd.DataFrame:
    """
    Load RFM features and attach KMeans cluster labels.
    Returns a dataframe with: user_id, recency, frequency, monetary, cluster
    """
    rfm = pd.read_csv(rfm_path)
    model, scaler = joblib.load(model_path)

    X = rfm[["recency", "frequency", "monetary"]]
    X_scaled = scaler.transform(X)

    rfm["cluster"] = model.predict(X_scaled)
    return rfm