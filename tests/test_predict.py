import pytest
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/kmeans_rfm.pkl")


def load_model():
    model, scaler = joblib.load(MODEL_PATH)
    return model, scaler


def predict_segment(recency, frequency, monetary, model, scaler):
    """Simple prediction function for testing."""

    # -----------------------------
    # Input validation added
    # -----------------------------
    if recency < 0 or frequency < 0 or monetary < 0:
        raise ValueError("RFM values cannot be negative")

    df = pd.DataFrame([{
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary
    }])

    X_scaled = scaler.transform(df)
    cluster = model.predict(X_scaled)[0]
    return cluster


# ------------------------------------------------------------
#                      TEST CASES
# ------------------------------------------------------------

def test_model_file_exists():
    assert MODEL_PATH.exists(), "Model file not found. Train the model first."


def test_model_loads_correctly():
    model, scaler = load_model()
    assert model is not None
    assert scaler is not None


def test_predict_valid_user_value():
    model, scaler = load_model()

    cluster = predict_segment(
        recency=10,
        frequency=15,
        monetary=300,
        model=model,
        scaler=scaler
    )

    # Allow numpy types
    assert isinstance(cluster, (int, float, np.integer, np.floating))
    assert cluster >= 0


def test_predict_low_activity_user():
    model, scaler = load_model()

    cluster = predict_segment(
        recency=120,
        frequency=1,
        monetary=5,
        model=model,
        scaler=scaler
    )

    assert isinstance(cluster, (int, float, np.integer, np.floating))


def test_input_validation_negative_values():
    model, scaler = load_model()

    with pytest.raises(ValueError):
        predict_segment(
            recency=-5,
            frequency=-10,
            monetary=-1,
            model=model,
            scaler=scaler
        )