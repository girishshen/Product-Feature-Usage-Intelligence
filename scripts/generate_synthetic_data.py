import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/synthetic_feature_usage.csv")

def generate_data(n_users=1500, n_days=120):
    np.random.seed(42)

    users = []
    for user_id in range(1, n_users + 1):
        signup_offset = np.random.randint(0, 60)
        signup_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=signup_offset)

        for day in range(n_days):
            event_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)

            if event_date < signup_date:
                continue

            # Simulate feature usage patterns
            features = ["Dashboard", "Search", "Export", "API", "Integrations"]
            for f in features:
                events = np.random.poisson(lam=np.random.uniform(0.1, 1.2))
                if events > 0:
                    users.append([
                        user_id,
                        signup_date,
                        event_date,
                        f,
                        events
                    ])

    df = pd.DataFrame(users, columns=["user_id", "signup_date", "event_date", "feature_name", "events_count"])
    return df

if __name__ == "__main__":
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(RAW_PATH, index=False)
    print(f"Generated synthetic data → {RAW_PATH}")