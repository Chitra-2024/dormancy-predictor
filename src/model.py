import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DormancyModel:
    """Trains and evaluates a dormancy classifier."""

    NUMERIC = ["trades_30d", "trades_60d", "trades_90d", "trade_trend",
               "days_since_last_trade", "avg_trade_value_90d",
               "distinct_products_90d", "logins_30d", "login_trend",
               "days_since_last_login", "tenure_days"]
    CATEGORICAL = ["segment"]
    LABEL = "dormant_next_90d"

    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.pipeline = self._build()

    def _build(self):
        onehot = OneHotEncoder(handle_unknown="ignore")
        if self.model_type == "logistic":
            prep = ColumnTransformer([("num", StandardScaler(), self.NUMERIC),
                                      ("cat", onehot, self.CATEGORICAL)])
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        elif self.model_type == "forest":
            prep = ColumnTransformer([("num", "passthrough", self.NUMERIC),
                                      ("cat", onehot, self.CATEGORICAL)])
            clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1)
        else:
            raise ValueError("model_type must be 'logistic' or 'forest'")
        return SkPipeline([("prep", prep), ("model", clf)])

    def load_features(self, train_path, test_path):
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        cols = self.NUMERIC + self.CATEGORICAL
        self.X_train, self.y_train = train[cols], train[self.LABEL]
        self.X_test, self.y_test = test[cols], test[self.LABEL]

    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def predict_proba(self, X=None):
        X = self.X_test if X is None else X
        return self.pipeline.predict_proba(X)[:, 1]