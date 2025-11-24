import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_data, aggregate_user_level

RAW = Path("data/raw/synthetic_feature_usage.csv")
CLEAN = Path("data/processed/clean_feature_usage.csv")

if __name__ == "__main__":
    df = pd.read_csv(RAW, parse_dates=["signup_date", "event_date"])
    df = clean_data(df)
    user_df = aggregate_user_level(df)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    user_df.to_csv(CLEAN, index=False)
    print(f"Saved cleaned user-level file → {CLEAN}")