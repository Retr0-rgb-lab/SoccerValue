"""Ball-location logistic baseline for ground-pass success."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def design_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        pitch_x, pitch_y = row["pitch"]
        bx, by = row["ball"]
        ex, ey = row["end"]
        distance = ((ex - bx) ** 2 + (ey - by) ** 2) ** 0.5
        diagonal = (pitch_x**2 + pitch_y**2) ** 0.5
        features.append([bx / pitch_x, by / pitch_y, distance / diagonal])
        labels.append(row["success"])
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


class BallLocationBaseline:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500)

    def fit(self, rows: list[dict]) -> None:
        features, labels = design_matrix(rows)
        self.model.fit(self.scaler.fit_transform(features), labels)

    def predict(self, rows: list[dict]) -> np.ndarray:
        features, _labels = design_matrix(rows)
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]
