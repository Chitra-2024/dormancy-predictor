import pandas as pd
from run_all import TRAIN_SNAP, TEST_SNAP
from src.model import DormancyModel


def test_test_snapshot_is_after_train_label_window():
    # label window is [snapshot, snapshot + 90 days), so the last day is +89
    train_label_last_day = pd.Timestamp(TRAIN_SNAP) + pd.Timedelta(days=89)
    assert pd.Timestamp(TEST_SNAP) > train_label_last_day


def test_probabilities_between_0_and_1(built):
    pipe, counts, silver = built
    pipe.silver_to_gold(TRAIN_SNAP)
    pipe.silver_to_gold(TEST_SNAP)
    m = DormancyModel("logistic")
    m.load_features(f"{pipe.gold_dir}/features_{TRAIN_SNAP}.parquet",
                    f"{pipe.gold_dir}/features_{TEST_SNAP}.parquet")
    m.fit()
    p = m.predict_proba()
    assert p.min() >= 0 and p.max() <= 1