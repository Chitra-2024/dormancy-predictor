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

    def _decay(self, days_to_quit, window, floor):
        slide = floor + (1 - floor) * days_to_quit / window
        return np.where(days_to_quit <= 0, 0.0,
                        np.where(days_to_quit < window, slide, 1.0))

    def _make_rates(self):
        n = self.n_clients
        self.days = pd.date_range(self.start, self.end, freq="D")
        day_idx = np.arange(len(self.days))         

        # 1. base daily rate from segment (trades per month / 30)
        monthly = self.clients["segment"].map(
            {"Active": 8, "Regular": 3, "Occasional": 1}
        ).values
        self.base_rate = monthly / 30 * self.rng.lognormal(0, 0.3, size=n)

        # 2. who will go dormant, and when
        self.will_quit = self.rng.random(n) < 0.25
        quit_day = self.rng.integers(59, 334, size=n)   # ~1 Mar to ~30 Nov
        self.quit_day = np.where(self.will_quit, quit_day, 10**6)

        days_to_quit = self.quit_day[:, None] - day_idx[None, :]

        # 3. decay multipliers (logins start falling earlier than trades)
        self.trade_mult = self._decay(days_to_quit, window=60, floor=0.1)
        self.login_mult = self._decay(days_to_quit, window=75, floor=0.15)

        # 4. noise: some NON-quitters get a temporary quiet spell
        has_spell = (~self.will_quit) & (self.rng.random(n) < 0.20)
        spell_start = self.rng.integers(0, 320, size=n)
        spell_len = self.rng.integers(20, 46, size=n)
        in_spell = (
            (day_idx[None, :] >= spell_start[:, None])
            & (day_idx[None, :] < (spell_start + spell_len)[:, None])
            & has_spell[:, None]
        )
        quiet = np.where(in_spell, 0.15, 1.0)
        self.trade_mult = self.trade_mult * quiet
        self.login_mult = self.login_mult * quiet

    def _make_trades(self):
        rate = self.base_rate[:, None] * self.trade_mult      # 10000 x 365 grid
        counts = self.rng.poisson(rate)                       # trades per client per day

        # turn the grid into one row per trade
        client_pos, day_pos = np.nonzero(counts)              # cells with >= 1 trade
        reps = counts[client_pos, day_pos]                    # how many trades in each cell
        client_pos = np.repeat(client_pos, reps)
        day_pos = np.repeat(day_pos, reps)
        n_trades = len(client_pos)

        self.trades = pd.DataFrame({
            "trade_id": np.arange(1, n_trades + 1),
            "client_id": self.clients["client_id"].values[client_pos],
            "trade_date": self.days[day_pos],
            "product": self.rng.choice(
                ["Equity", "F&O", "Mutual Fund", "IPO"],
                size=n_trades, p=[0.50, 0.25, 0.20, 0.05],
            ),
            "value": np.round(
                self.rng.lognormal(mean=np.log(5000), sigma=1.0, size=n_trades), 2
            ),
        })
    
    def _make_logins(self):
        login_rate = (0.1 + 1.5 * self.base_rate)[:, None] * self.login_mult
        counts = self.rng.poisson(login_rate)

        client_pos, day_pos = np.nonzero(counts)
        reps = counts[client_pos, day_pos]
        client_pos = np.repeat(client_pos, reps)
        day_pos = np.repeat(day_pos, reps)

        self.logins = pd.DataFrame({
            "login_id": np.arange(1, len(client_pos) + 1),
            "client_id": self.clients["client_id"].values[client_pos],
            "login_date": self.days[day_pos],
        })

    def generate(self):
        self._make_clients()
        self._make_rates()
        self._make_trades()
        self._make_logins()
        return self