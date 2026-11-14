"""Small fully convolutional success map.

The network sees the whole pitch and is scored only at the pass destination.
That is the SoccerMap training pattern reduced to one target: completion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from soccervalue.raster import GRID_X, GRID_Y, cell_index, raster


class SoccerMapSuccess(nn.Module):
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
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        fine = self.fine(frames)
        coarse = self.coarse(fine)
        up = nn.functional.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        return torch.sigmoid(self.head(torch.relu(self.fuse(torch.cat([fine, up], dim=1)))))


def destination_loss(predicted: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    """Binary cross-entropy at each pass's end cell only."""
    selected = []
    labels = []
    for index, row in enumerate(rows):
        pitch_x, pitch_y = row["pitch"]
        ix, iy = cell_index(row["end"][0], row["end"][1], pitch_x, pitch_y)
        selected.append(predicted[index, 0, iy, ix])
        labels.append(row["success"])
    picked = torch.stack(selected)
    target = torch.tensor(labels, dtype=picked.dtype, device=picked.device)
    return nn.functional.binary_cross_entropy(picked, target)


def frames_from_rows(rows: list[dict]) -> np.ndarray:
    return np.stack([raster(row["attack"], row["defense"], row["ball"], row["pitch"]) for row in rows])


def predict_destination(model: SoccerMapSuccess, rows: list[dict], device: torch.device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            frames = torch.tensor(frames_from_rows(batch), device=device)
            predicted = model(frames)
            for index, row in enumerate(batch):
                pitch_x, pitch_y = row["pitch"]
                ix, iy = cell_index(row["end"][0], row["end"][1], pitch_x, pitch_y)
                outputs.append(float(predicted[index, 0, iy, ix].cpu()))
    return np.asarray(outputs, dtype=np.float64)


def train_soccermap(
    train_rows: list[dict],
    val_rows: list[dict],
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 4,
    seed: int = 0,
) -> tuple[SoccerMapSuccess, list[dict]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoccerMapSuccess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_brier = float("inf")
    waited = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = [train_rows[int(i)] for i in order[start : start + batch_size]]
            frames = torch.tensor(frames_from_rows(batch), device=device)
            loss = destination_loss(model(frames), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        val_prob = predict_destination(model, val_rows, device, batch_size)
        val_y = np.asarray([row["success"] for row in val_rows])
        val_brier = float(np.mean((val_prob - val_y) ** 2))
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_brier": val_brier})
        if val_brier < best_brier:
            best_brier = val_brier
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
