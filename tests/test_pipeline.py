import pandas as pd


def test_negative_value_is_quarantined(built):
    pipe, counts, silver = built
    rejected = pd.read_csv(f"{silver}/rejected_rows.csv")
    assert (rejected["reason"] == "value not positive").sum() > 0
    trades = pd.read_parquet(f"{silver}/trades.parquet")
    assert (trades["value"] > 0).all()


def test_bronze_to_silver_is_idempotent(built):
    pipe, counts, silver = built
    again = pipe.bronze_to_silver()
    assert again == counts


def test_features_are_valid_and_before_snapshot(built):
    pipe, counts, silver = built
    df = pipe.silver_to_gold("2025-06-01")
    for col in ["trades_30d", "trades_60d", "trades_90d",
                "logins_30d", "distinct_products_90d"]:
        assert (df[col] >= 0).all()
    # a trade or login ON the snapshot day would give 0 days
    assert df["days_since_last_trade"].min() >= 1
    assert df["days_since_last_login"].min() >= 1