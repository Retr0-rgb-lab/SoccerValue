"""One deterministic step of every player, conditioned on the ball.

The same weights are shared across players. One forward pass is the prediction
that gets scored. There is no sample of 20 futures.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from soccervalue.world_segments import DT, roll_position

SCALE_M = 50.0
DRAG = 0.3


class PlayerMotion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64))
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.out = nn.Linear(64, 2)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        updated, _weights = self.attn(hidden, hidden, hidden, key_padding_mask=pad_mask, need_weights=False)
        return self.out(hidden + updated)


def nearest_to_path(origin_players: np.ndarray, ball_path: np.ndarray, count: int = 5) -> np.ndarray:
    """Indices of players whose starting point lies closest to the ball path."""
    distance = np.linalg.norm(origin_players[:, None, :] - ball_path[None, :, :], axis=2)
    nearest = np.nanmin(distance, axis=1)
    order = np.argsort(np.where(np.isfinite(nearest), nearest, 1e9))
    return order[:count]


def constant_player(player: dict, steps: int, dt: float = DT) -> np.ndarray:
    times = np.arange(steps) * dt
    return player["xy"] + player["velocity"] * times[:, None]


def drag_ball(segment: dict, drag: float = DRAG) -> np.ndarray:
    times = np.arange(len(segment["future"])) * segment["dt"]
    return roll_position(segment["origin"], segment["velocity"], times, drag)


def displacement_error(predicted: np.ndarray, truth: np.ndarray, chosen: np.ndarray) -> float | None:
    """Mean error of the chosen players. predicted and truth are (T, P, 2)."""
    if len(chosen) == 0 or predicted.shape[0] != truth.shape[0]:
        return None
    error = np.linalg.norm(predicted[:, chosen, :] - truth[:, chosen, :], axis=2)
    if not np.isfinite(error).any():
        return None
    return float(np.nanmean(error))


def _tokens(positions: np.ndarray, velocities: np.ndarray, same_team: np.ndarray, ball: np.ndarray) -> np.ndarray:
    offset = positions - ball
    distance = np.linalg.norm(offset, axis=1)
    return np.column_stack(
        [
            positions[:, 0] / SCALE_M,
            positions[:, 1] / SCALE_M,
            velocities[:, 0] / 10.0,
            velocities[:, 1] / 10.0,
            same_team,
            np.full(len(positions), ball[0] / SCALE_M),
            np.full(len(positions), ball[1] / SCALE_M),
            distance / SCALE_M,
        ]
    ).astype(np.float32)


def _same_team(segment: dict) -> np.ndarray:
    roster = segment["roster"]
    distances = [np.linalg.norm(player["xy"] - segment["origin"]) for player in roster]
    possessor = roster[int(np.argmin(distances))]["team"]
    return np.asarray([1.0 if player["team"] == possessor else 0.0 for player in roster], dtype=np.float32)


def rollout(model: nn.Module, segment: dict, ball_path: np.ndarray, device: torch.device) -> np.ndarray:
    """Closed loop from the kick. The ball path is given; players are not."""
    roster = segment["roster"]
    positions = np.stack([player["xy"] for player in roster]).astype(np.float64)
    velocities = np.stack([player["velocity"] for player in roster]).astype(np.float64)
    same = _same_team(segment)
    path = [positions.copy()]
    model.eval()
    for step in range(1, len(ball_path)):
        tokens = torch.tensor(_tokens(positions, velocities, same, ball_path[step - 1]), device=device)[None]
        mask = torch.zeros(1, len(roster), dtype=torch.bool, device=device)
        with torch.no_grad():
            delta = model(tokens, mask)[0].cpu().numpy()
        positions = positions + delta
        velocities = delta / segment["dt"]
        path.append(positions.copy())
    return np.stack(path)


def score_players(model: nn.Module | None, segments: list[dict], use_drag_ball: bool, device: torch.device) -> dict:
    at_one = []
    at_end = []
    index_1s = int(round(1.0 / DT))
    for segment in segments:
        ball = drag_ball(segment) if use_drag_ball else segment["future"]
        truth = segment["players"]
        if model is None:
            predicted = np.stack([constant_player(player, len(ball)) for player in segment["roster"]], axis=1)
        else:
            predicted = rollout(model, segment, ball, device)
        chosen = nearest_to_path(np.stack([player["xy"] for player in segment["roster"]]), ball, 5)
        if truth.shape[0] > index_1s and predicted.shape[0] > index_1s:
            error = displacement_error(predicted[1 : index_1s + 1], truth[1 : index_1s + 1], chosen)
            if error is not None:
                at_one.append(error)
        error = displacement_error(predicted[-1:], truth[-1:], chosen)
        if error is not None:
            at_end.append(error)
    return {
        "n_1s": len(at_one),
        "ade_1s": float(np.mean(at_one)) if at_one else None,
        "n_end": len(at_end),
        "end_error": float(np.mean(at_end)) if at_end else None,
    }


def train_player_motion(
    train_segments: list[dict],
    val_segments: list[dict],
    epochs: int = 12,
    batch_size: int = 64,
    patience: int = 4,
    seed: int = 0,
) -> tuple[PlayerMotion, list[dict]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlayerMotion().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_score = float("inf")
    waited = 0
    history = []
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(train_segments))
        total = 0.0
        seen = 0
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for index in order:
            example = _example(train_segments[int(index)], int(generator.integers(0, 2**31)))
            if example is None:
                continue
            batch.append(example)
            if len(batch) >= batch_size:
                total += _step(model, optimizer, batch, device)
                seen += len(batch)
                batch = []
        if batch:
            total += _step(model, optimizer, batch, device)
            seen += len(batch)
        scores = score_players(model, val_segments[:300], use_drag_ball=True, device=device)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_ade_1s": scores["ade_1s"]})
        score = scores["ade_1s"] if scores["ade_1s"] is not None else float("inf")
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


def _example(segment: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    players = segment["players"]
    steps = len(players) - 1
    if steps < 1:
        return None
    step = int(np.random.default_rng(seed).integers(0, steps))
    current = players[step]
    target = players[step + 1]
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        return None
    if step == 0:
        velocity = np.stack([player["velocity"] for player in segment["roster"]])
    else:
        velocity = (current - players[step - 1]) / segment["dt"]
    ball = segment["future"][step]
    tokens = _tokens(current, velocity, _same_team(segment), ball)
    chosen = set(nearest_to_path(players[0], segment["future"], 5).tolist())
    weight = np.asarray([3.0 if index in chosen else 1.0 for index in range(len(segment["roster"]))], dtype=np.float32)
    return tokens, (target - current).astype(np.float32), weight


def _step(model: PlayerMotion, optimizer: torch.optim.Optimizer, batch: list, device: torch.device) -> float:
    width = max(item[0].shape[0] for item in batch)
    tokens = np.zeros((len(batch), width, 8), dtype=np.float32)
    target = np.zeros((len(batch), width, 2), dtype=np.float32)
    weight = np.zeros((len(batch), width), dtype=np.float32)
    mask = np.ones((len(batch), width), dtype=bool)
    for row, (token_row, target_row, weight_row) in enumerate(batch):
        count = len(token_row)
        tokens[row, :count] = token_row
        target[row, :count] = target_row
        weight[row, :count] = weight_row
        mask[row, :count] = False
    predicted = model(torch.tensor(tokens, device=device), torch.tensor(mask, device=device))
    truth = torch.tensor(target, device=device)
    weights = torch.tensor(weight, device=device)
    squared = ((predicted - truth) ** 2).sum(dim=-1)
    loss = (squared * weights).sum() / weights.sum().clamp(min=1.0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()) * len(batch)

