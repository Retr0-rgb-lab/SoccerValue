"""State value as two shot-opportunity heads. Not a pass-success map."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from soccervalue.raster import raster


def _nearest(origin: tuple[float, float], points: list[tuple[float, float]], fallback: float) -> float:
    if not points:
        return fallback
    return min(((point[0] - origin[0]) ** 2 + (point[1] - origin[1]) ** 2) ** 0.5 for point in points)


def spatial_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple state features plus four meter counts. No extra model capacity."""
    base, y_for, y_against = simple_features(rows)
    extra = []
    for row in rows:
        pitch_x, pitch_y = row["pitch"]
        ball = row["ball"]
        diagonal = (pitch_x**2 + pitch_y**2) ** 0.5
        opponents = row["opponent"]
        extra.append(
            [
                _nearest(ball, opponents, diagonal) / diagonal,
                (pitch_x - ball[0]) / pitch_x,
                float(sum(point[0] > ball[0] for point in row["team"])),
                float(sum(_nearest(ball, [point], diagonal) <= 5.0 for point in opponents)),
            ]
        )
    return np.column_stack([base, np.asarray(extra, dtype=np.float64)]), y_for, y_against


def simple_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    y_for = []
    y_against = []
    for row in rows:
        pitch_x, pitch_y = row["pitch"]
        bx, by = row["ball"]
        features.append(
            [
                bx / pitch_x,
                by / pitch_y,
                bx / pitch_x,
                (pitch_x - bx) / pitch_x,
                float(row["possession_by_t"]),
            ]
        )
        y_for.append(row["y_for"])
        y_against.append(row["y_against"])
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(y_for, dtype=np.int64),
        np.asarray(y_against, dtype=np.int64),
    )


def _fit_or_constant(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return float(np.mean(labels))
    model.fit(features, labels)
    return None


def _predict_head(model: LogisticRegression, constant: float | None, features: np.ndarray) -> np.ndarray:
    if constant is not None:
        return np.full(len(features), constant)
    return model.predict_proba(features)[:, 1]


def constant_rates(train_rows: list[dict]) -> tuple[float, float]:
    if not train_rows:
        return 0.0, 0.0
    return (
        float(np.mean([row["y_for"] for row in train_rows])),
        float(np.mean([row["y_against"] for row in train_rows])),
    )


class SimpleStateValue:
    """Separate logistic heads for team T and the opponent. No class balancing."""

    def __init__(self, feature_fn=simple_features) -> None:
        self.feature_fn = feature_fn
        self.scaler = StandardScaler()
        self.for_model = LogisticRegression(max_iter=1000)
        self.against_model = LogisticRegression(max_iter=1000)
        self.for_constant = None
        self.against_constant = None

    def fit(self, rows: list[dict]) -> None:
        features, y_for, y_against = self.feature_fn(rows)
        scaled = self.scaler.fit_transform(features)
        self.for_constant = _fit_or_constant(self.for_model, scaled, y_for)
        self.against_constant = _fit_or_constant(self.against_model, scaled, y_against)

    def predict(self, rows: list[dict]) -> np.ndarray:
        features, _for, _against = self.feature_fn(rows)
        scaled = self.scaler.transform(features)
        return np.column_stack(
            [
                _predict_head(self.for_model, self.for_constant, scaled),
                _predict_head(self.against_model, self.against_constant, scaled),
            ]
        )


class StateValueNet(nn.Module):
    """One pair of probabilities for the whole state, not a pass-destination map."""

    def __init__(self) -> None:
        super().__init__()
        self.fine = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
        )
        self.coarse = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
        )
        self.fuse = nn.Conv2d(48, 16, 1)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, frames: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        fine = self.fine(frames)
        coarse = self.coarse(fine)
        up = nn.functional.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        fused = torch.relu(self.fuse(torch.cat([fine, up], dim=1)))
        pooled = fused.mean(dim=(2, 3))
        return torch.sigmoid(self.mlp(torch.cat([pooled, context], dim=1)))


def two_head_loss(predicted: torch.Tensor, y_for: torch.Tensor, y_against: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(predicted[:, 0], y_for) + nn.functional.binary_cross_entropy(
        predicted[:, 1], y_against
    )


def _context(rows: list[dict]) -> np.ndarray:
    values = []
    for row in rows:
        pitch_x, pitch_y = row["pitch"]
        bx, by = row["ball"]
        values.append([bx / pitch_x, by / pitch_y, float(row["possession_by_t"])])
    return np.asarray(values, dtype=np.float32)


def _frames(rows: list[dict]) -> np.ndarray:
    return np.stack([raster(row["team"], row["opponent"], row["ball"], row["pitch"]) for row in rows])


def predict_state_value(model: StateValueNet, rows: list[dict], device: torch.device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            frames = torch.tensor(_frames(batch), device=device)
            context = torch.tensor(_context(batch), device=device)
            outputs.append(model(frames, context).cpu().numpy())
    if not outputs:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(outputs, axis=0).astype(np.float64)


def train_state_value(
    train_rows: list[dict],
    val_rows: list[dict],
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 4,
    seed: int = 0,
) -> tuple[StateValueNet, list[dict]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_score = float("inf")
    waited = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = [train_rows[int(index)] for index in order[start : start + batch_size]]
            frames = torch.tensor(_frames(batch), device=device)
            context = torch.tensor(_context(batch), device=device)
            y_for = torch.tensor([row["y_for"] for row in batch], dtype=torch.float32, device=device)
            y_against = torch.tensor([row["y_against"] for row in batch], dtype=torch.float32, device=device)
            loss = two_head_loss(model(frames, context), y_for, y_against)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        val_prob = predict_state_value(model, val_rows, device, batch_size)
        val_for = np.asarray([row["y_for"] for row in val_rows], dtype=np.float64)
        val_against = np.asarray([row["y_against"] for row in val_rows], dtype=np.float64)
        score = float(np.mean((val_prob[:, 0] - val_for) ** 2) + np.mean((val_prob[:, 1] - val_against) ** 2)) / 2
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_brier_mean": score})
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, history
