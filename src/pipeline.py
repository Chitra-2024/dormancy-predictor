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
        results = self.run_quality_checks()
        self.write_quality_report(results)      # write the evidence first
        failed = [name for name, r in results.items() if not r["passed"]]
        if failed:
            raise ValueError(f"Critical quality checks failed: {failed}")
        return self.counts

    def run_quality_checks(self):
        """Inspect the Silver files. Returns {check name: {passed, count}}."""
        read = lambda n: pd.read_parquet(os.path.join(self.silver_dir, f"{n}.parquet"))
        clients, trades, logins = read("clients"), read("trades"), read("logins")
        results = {}

        def record(name, bad_count):
            results[name] = {"passed": int(bad_count) == 0, "count": int(bad_count)}

        record("no null client_id or trade_id",
               clients["client_id"].isna().sum()
               + trades["trade_id"].isna().sum() + trades["client_id"].isna().sum()
               + logins["login_id"].isna().sum() + logins["client_id"].isna().sum())

        record("trade value greater than 0",
               trades["value"].isna().sum() + (trades["value"] <= 0).sum())

        record("no duplicate primary keys",
               clients["client_id"].duplicated().sum()
               + trades["trade_id"].duplicated().sum()
               + logins["login_id"].duplicated().sum())

        record("all foreign keys resolve",
               (~trades["client_id"].isin(clients["client_id"])).sum()
               + (~logins["client_id"].isin(clients["client_id"])).sum())

        def out_of_range(s):
            return ((s < self.START) | (s > self.END)).sum()

        record("all dates within 2025-01-01 to 2025-12-31",
               out_of_range(trades["trade_date"]) + out_of_range(logins["login_date"]))

        # reconciliation: before = after + rejected, and after = real file length
        actual = {"clients": len(clients), "trades": len(trades), "logins": len(logins)}
        mismatches = 0
        for table, c in self.counts.items():
            if c["before"] != c["after"] + c["rejected"] or c["after"] != actual[table]:
                mismatches += 1
        record("row counts reconcile (before = after + rejected)", mismatches)

        return results

    def write_quality_report(self, results):
        lines = ["QUALITY REPORT", "=" * 50, ""]
        for name, r in results.items():
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"[{status}] {name}  (problem rows: {r['count']})")

        lines += ["", "ROW COUNTS", "-" * 50]
        for table, c in self.counts.items():
            lines.append(f"{table}: before={c['before']}  after={c['after']}  "
                         f"rejected={c['rejected']}")

        path = os.path.join(self.silver_dir, "quality_report.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))