"""Logistic pass-success models that share one family and differ by features."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticSuccess:
    def __init__(self, feature_fn: Callable[[list[dict]], tuple[np.ndarray, np.ndarray]], names: tuple[str, ...]) -> None:
        self.feature_fn = feature_fn
        self.names = names
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, rows: list[dict]) -> None:
        features, labels = self.feature_fn(rows)
        self.model.fit(self.scaler.fit_transform(features), labels)

    def predict(self, rows: list[dict]) -> np.ndarray:
        features, _labels = self.feature_fn(rows)
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]

    def coefficients(self) -> dict[str, float]:
        scaled = {name: float(value) for name, value in zip(self.names, self.model.coef_[0])}
        scaled["intercept"] = float(self.model.intercept_[0])
        return scaled
