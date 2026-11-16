import numpy as np
import pandas as pd


class SyntheticBrokerData:

    def __init__(self, n_clients=10000, seed=42):
        self.n_clients = n_clients
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.start = pd.Timestamp("2025-01-01")
        self.end = pd.Timestamp("2025-12-31")

        self.clients = None
        self.trades = None
        self.logins = None

    def _make_clients(self):
        n = self.n_clients

        client_id = np.arange(1, n + 1)

        # signed up between 1 day and 5 years before 1 Jan 2025
        days_before = self.rng.integers(1, 5 * 365, size=n)
        signup_date = self.start - pd.to_timedelta(days_before, unit="D")

        segment = self.rng.choice(
            ["Active", "Regular", "Occasional"],
            size=n,
            p=[0.20, 0.40, 0.40],
        )
        age_band = self.rng.choice(
            ["18-25", "26-35", "36-45", "46-60", "60+"],
            size=n,
            p=[0.15, 0.35, 0.25, 0.18, 0.07],
        )
        city = self.rng.choice(
            ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Kolkata", "Other"],
            size=n,
            p=[0.20, 0.15, 0.15, 0.10, 0.08, 0.07, 0.25],
        )

        self.clients = pd.DataFrame({
            "client_id": client_id,
            "signup_date": signup_date,
            "segment": segment,
            "age_band": age_band,
            "city": city,
        })