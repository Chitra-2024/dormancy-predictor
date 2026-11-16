import os
import pandas as pd


class Pipeline:
    """Moves data Bronze -> Silver -> Gold."""

    START = pd.Timestamp("2025-01-01")
    END = pd.Timestamp("2025-12-31")

    def __init__(self, bronze_dir="data/bronze", silver_dir="data/silver",
                 gold_dir="data/gold"):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        self.counts = {}      # before/after/rejected per table, filled later
    
    def _parse_dates(self, s):
        """Accept 2025-03-15 or 15/03/2025. Anything else becomes NaT (missing)."""
        iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        slash = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        return iso.fillna(slash)

    def _apply_rules(self, df, rules):
        """rules = list of (mask, reason). The FIRST rule a row breaks wins."""
        reason = pd.Series("", index=df.index)
        for mask, text in rules:
            reason = reason.mask((reason == "") & mask, text)
        return reason

    def _clean_events(self, raw, id_col, date_col, valid_ids, extra_rules=None):
        df = raw.copy()
        df[date_col] = self._parse_dates(df[date_col])

        rules = [
            (df[id_col].isna(), f"missing {id_col}"),
            (df["client_id"].isna(), "missing client_id"),
            (df[id_col].duplicated(keep="first"), f"duplicate {id_col}"),
        ]
        rules += extra_rules or []
        rules += [
            (df[date_col].isna(), "unparseable date"),
            ((df[date_col] < self.START) | (df[date_col] > self.END),
             "date outside 2025"),
            (~df["client_id"].isin(valid_ids), "client_id not in clients"),
        ]

        reason = self._apply_rules(df, rules)
        bad = reason != ""

        rejected = raw[bad].copy()           # keep the ORIGINAL text for evidence
        rejected["reason"] = reason[bad]

        clean = df[~bad].copy()
        clean[id_col] = clean[id_col].astype("int64")
        clean["client_id"] = clean["client_id"].astype("int64")
        return clean, rejected

    def bronze_to_silver(self):
        os.makedirs(self.silver_dir, exist_ok=True)
        ids = {"trade_id": "Int64", "login_id": "Int64", "client_id": "Int64"}

        def read(name):
            return pd.read_csv(os.path.join(self.bronze_dir, f"{name}.csv"), dtype=ids)

        # ---- clients ----
        clients_raw = read("clients")
        clients = clients_raw.copy()
        clients["signup_date"] = pd.to_datetime(clients["signup_date"], errors="coerce")
        bad = (clients["client_id"].isna() | clients["client_id"].duplicated()
               | clients["signup_date"].isna())
        clients_rej = clients_raw[bad].copy()
        clients_rej["reason"] = "invalid client row"
        clients = clients[~bad].copy()
        clients["client_id"] = clients["client_id"].astype("int64")
        valid_ids = set(clients["client_id"])

        # ---- trades ----
        trades_raw = read("trades")
        value_rule = [(trades_raw["value"].isna() | (trades_raw["value"] <= 0),
                       "value not positive")]
        trades, trades_rej = self._clean_events(
            trades_raw, "trade_id", "trade_date", valid_ids, value_rule)

        # ---- logins ----
        logins_raw = read("logins")
        logins, logins_rej = self._clean_events(
            logins_raw, "login_id", "login_date", valid_ids)

        # ---- write Silver (overwrites: idempotent) ----
        for name, df in [("clients", clients), ("trades", trades), ("logins", logins)]:
            df.to_parquet(os.path.join(self.silver_dir, f"{name}.parquet"), index=False)

        rejected = pd.concat(
            [clients_rej.assign(source_table="clients"),
             trades_rej.assign(source_table="trades"),
             logins_rej.assign(source_table="logins")],
            ignore_index=True)
        rejected.to_csv(os.path.join(self.silver_dir, "rejected_rows.csv"), index=False)

        self.counts = {
            "clients": {"before": len(clients_raw), "after": len(clients), "rejected": len(clients_rej)},
            "trades": {"before": len(trades_raw), "after": len(trades), "rejected": len(trades_rej)},
            "logins": {"before": len(logins_raw), "after": len(logins), "rejected": len(logins_rej)},
        }
        return self.counts