"""
Evaluate the KMeans segmentation model using:
- Silhouette Score
- Inertia
- Cluster summary stats

Outputs:
- Printed evaluation results
- CSV file: data/processed/cluster_summary.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
import joblib

RFM_PATH = Path("data/processed/rfm_features.csv")
MODEL_PATH = Path("models/kmeans_rfm.pkl")
OUTPUT_SUMMARY = Path("data/processed/cluster_summary.csv")


def evaluate_kmeans():
    """Evaluate KMeans clustering on RFM features."""

    # ---------------------------
    # Load data & model
    # ---------------------------
    if not RFM_PATH.exists():
        raise FileNotFoundError("RFM file not found. Run build_rfm.py first.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")

    rfm = pd.read_csv(RFM_PATH)
    model, scaler = joblib.load(MODEL_PATH)

    X = rfm[["recency", "frequency", "monetary"]]

    # Scale the data
    X_scaled = scaler.transform(X)

    # Predict clusters
    rfm["cluster"] = model.predict(X_scaled)

    # ---------------------------
    # Evaluation Metrics
    # ---------------------------
    silhouette = silhouette_score(X_scaled, rfm["cluster"])
    inertia = model.inertia_

    print("\n==========================")
    print("📊 Model Evaluation Report")
    print("==========================\n")
    print(f"Silhouette Score : {silhouette:.4f}")
    print(f"Inertia          : {inertia:.4f}")

    # ---------------------------
    # Cluster Summary Stats
    # ---------------------------
    summary = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .round(2)
    )
    print("\nCluster Summary:")
    print(summary)

    # Save summary
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY)

    print(f"\nCluster summary saved → {OUTPUT_SUMMARY}")
    print("\n✔ Evaluation complete.\n")


if __name__ == "__main__":
    evaluate_kmeans()