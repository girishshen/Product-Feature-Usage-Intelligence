import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

RFM_PATH = Path("data/processed/rfm_features.csv")
MODEL_PATH = Path("models/kmeans_rfm.pkl")

if __name__ == "__main__":
    df = pd.read_csv(RFM_PATH)

    X = df[["recency", "frequency", "monetary"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42, n_init="auto")
    model.fit(X_scaled)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, scaler), MODEL_PATH)

    print(f"Saved model → {MODEL_PATH}")