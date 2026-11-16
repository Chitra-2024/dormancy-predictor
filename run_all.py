import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # draw charts without opening a window
import matplotlib.pyplot as plt

from src.data_generator import SyntheticBrokerData
from src.pipeline import Pipeline
from src.model import DormancyModel

TRAIN_SNAP = "2025-06-01"
TEST_SNAP = "2025-09-01"
COST, SAVE_RATE, VALUE = 50, 0.30, 2000
CALL_LIST_MODEL = "logistic"


def net(r):
    return r["tp"] * SAVE_RATE * VALUE - (r["tp"] + r["fp"]) * COST


def main():
    print("1/5 Generating Bronze CSVs ...")
    SyntheticBrokerData().generate().write_bronze("data/bronze")

    print("2/5 Bronze -> Silver (with quality checks) ...")
    pipe = Pipeline()
    pipe.bronze_to_silver()

    print("3/5 Silver -> Gold ...")
    for snap in (TRAIN_SNAP, TEST_SNAP):
        pipe.silver_to_gold(snap)

    print("4/5 Training, choosing thresholds, evaluating ...")
    tr = f"data/gold/features_{TRAIN_SNAP}.parquet"
    te = f"data/gold/features_{TEST_SNAP}.parquet"
    models, rows, thresholds = {}, {}, {}
    for kind, label in [("logistic", "logistic regression"),
                        ("forest", "random forest")]:
        m = DormancyModel(kind)
        m.load_features(tr, te)
        m.fit()
        t = m.best_threshold(COST, SAVE_RATE, VALUE)   # chosen on TRAIN only
        r = m.evaluate(t)
        r["threshold"], r["net_value"] = t, net(r)
        models[kind], rows[label], thresholds[kind] = m, r, t
        t500 = m.best_threshold(500, SAVE_RATE, VALUE)
        print(f"   sensitivity: {label} threshold {t:.2f} at cost 50, "
              f"{t500:.2f} at cost 500")

    base = models["forest"].baseline()
    base["threshold"], base["net_value"] = np.nan, net(base)
    table = pd.DataFrame({"baseline (>30 days)": base, **rows}).T

    os.makedirs("reports", exist_ok=True)
    table.to_csv("reports/results.csv")
    cols = ["threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "net_value"]
    print(table[cols].round(3).to_string())

    print("5/5 Chart and RM call list ...")
    y = models["forest"].y_test.values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, m in models.items():
        sw = m.sweep(y, m.predict_proba(), COST, SAVE_RATE, VALUE)
        ax.plot(sw["threshold"], sw["net_value"] / 1000, label=kind)
        ax.axvline(thresholds[kind], linestyle="--", alpha=0.4)
    ax.axhline(base["net_value"] / 1000, color="grey", linestyle=":", label="baseline rule")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("net value (₹ thousand)")
    ax.set_title("Net value vs threshold on the test snapshot (cost = ₹50)")
    ax.legend()
    fig.savefig("reports/threshold_chart.png", dpi=150, bbox_inches="tight")

    test_df = pd.read_parquet(te)
    out = test_df[["client_id", "segment", "trades_30d", "days_since_last_trade"]].copy()
    out["risk_score"] = models[CALL_LIST_MODEL].predict_proba().round(3)
    out = out[out["risk_score"] >= thresholds[CALL_LIST_MODEL]]
    out = out.sort_values("risk_score", ascending=False)
    out.to_csv("reports/rm_call_list.csv", index=False)
    print(f"   call list: {len(out)} clients, highest risk first")


if __name__ == "__main__":
    main()