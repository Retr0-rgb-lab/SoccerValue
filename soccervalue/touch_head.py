"""Who touches the ball next, given the real ball path and players at the kick.

Players are extrapolated at constant velocity. The baseline picks the earliest
intercept. A linear score may reorder them. It does not see player positions
from the future.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def player_features(segment: dict, ball: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    future = segment["future"] if ball is None else ball
    rows = []
    roster = segment["roster"]
    possessor = min(roster, key=lambda player: np.linalg.norm(player["xy"] - segment["origin"]))
    for player in roster:
        distance = float(np.linalg.norm(player["xy"] - segment["origin"]))
        intercept = _intercept_time(player, future, segment["dt"])
        rows.append(
            [
                distance,
                intercept,
                1.0 if player["team"] == possessor["team"] else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64), roster


def baseline_index(features: np.ndarray) -> int:
    return int(np.argmin(features[:, 1]))


def _intercept_time(player: dict, future: np.ndarray, dt: float, radius: float = 2.0) -> float:
    times = np.arange(len(future)) * dt
    places = player["xy"] + player["velocity"] * times[:, None]
    distance = np.linalg.norm(places - future, axis=1)
    hits = np.flatnonzero(distance <= radius)
    if len(hits) == 0:
        return 10.0 + float(np.min(distance))
    return float(hits[0] * dt)


class TouchRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features).squeeze(-1)


def fit_touch(segments: list[dict], steps: int = 200, seed: int = 0) -> tuple[TouchRanker, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    packed = [_pack(segment) for segment in segments]
    packed = [item for item in packed if item is not None]
    if len(packed) < 8:
        raise RuntimeError("not enough labelled touches to fit a ranker")
    mean, std = _standardise(packed)
    model = TouchRanker()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        order = np.random.default_rng(seed).permutation(len(packed))
        for index in order:
            features, label = packed[int(index)]
            scaled = torch.tensor((features - mean) / std, dtype=torch.float32)
            loss = nn.functional.cross_entropy(model(scaled)[None, :], torch.tensor([label]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, mean, std


def predict_index(model: TouchRanker, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> int:
    model.eval()
    scaled = torch.tensor((features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return int(torch.argmax(model(scaled)).cpu())


def team_probability(model: TouchRanker, features: np.ndarray, roster: list[dict], mean: np.ndarray, std: np.ndarray, team: str) -> float:
    model.eval()
    scaled = torch.tensor((features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        weights = torch.softmax(model(scaled), dim=0).cpu().numpy()
    return float(sum(weight for weight, player in zip(weights, roster) if player["team"] == team))


def score_touch(segments: list[dict], model: TouchRanker | None, mean: np.ndarray | None, std: np.ndarray | None, ball_from: str) -> dict:
    top1_model = []
    top1_base = []
    team_model = []
    team_base = []
    for segment in segments:
        if segment["toucher"] is None:
            continue
        ball = None if ball_from == "real" else _ball_from_model(segment, ball_from)
        features, roster = player_features(segment, ball)
        label = next((index for index, player in enumerate(roster) if player["id"] == segment["toucher"]["id"]), None)
        if label is None:
            continue
        base = baseline_index(features)
        top1_base.append(base == label)
        possessor = min(roster, key=lambda player: np.linalg.norm(player["xy"] - segment["origin"]))
        same = segment["toucher"]["team"] == possessor["team"]
        team_base.append((roster[base]["team"] == possessor["team"]) == same)
        if model is None:
            continue
        chosen = predict_index(model, features, mean, std)
        top1_model.append(chosen == label)
        team_model.append((roster[chosen]["team"] == possessor["team"]) == same)
    return {
        "n": len(top1_base),
        "baseline_top1": float(np.mean(top1_base)) if top1_base else None,
        "model_top1": float(np.mean(top1_model)) if top1_model else None,
        "baseline_team_accuracy": float(np.mean(team_base)) if team_base else None,
        "model_team_accuracy": float(np.mean(team_model)) if team_model else None,
    }


def _ball_from_model(segment: dict, kind: str) -> np.ndarray:
    from soccervalue.ball_flight import roll_position

    drag = 0.0 if kind == "constant" else float(kind)
    times = np.arange(len(segment["future"])) * segment["dt"]
    return roll_position(segment["origin"], segment["velocity"], times, drag)


def _pack(segment: dict) -> tuple[np.ndarray, int] | None:
    if segment["toucher"] is None:
        return None
    features, roster = player_features(segment)
    label = next((index for index, player in enumerate(roster) if player["id"] == segment["toucher"]["id"]), None)
    if label is None:
        return None
    return features, label


def _standardise(packed: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([features for features, _label in packed], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std
