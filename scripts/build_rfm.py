import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rfm import compute_rfm

CLEAN = Path("data/processed/clean_feature_usage.csv")
RFM_PATH = Path("data/processed/rfm_features.csv")

if __name__ == "__main__":
    df = pd.read_csv(CLEAN, parse_dates=["last_event_date", "signup_date"])
    rfm = compute_rfm(df)

    RFM_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(RFM_PATH, index=False)
    print(f"Saved RFM file → {RFM_PATH}")