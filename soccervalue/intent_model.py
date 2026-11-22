"""A learned aimed receiver, compared against the geometric rule.

The geometric rule in `soccervalue.intent` reads the aim off the kick and takes
the teammate closest to that ray. It agrees with the annotated recipient on 76%
of successful passes. This module asks whether a trained model can do better
from the same information.

Supervision. On a successful pass the annotated recipient is the teammate who
received the ball, so it is a usable label for the aim. On an unsuccessful pass
the annotated recipient is the opponent who intercepted, so there is no label
for the aim and those passes cannot train the model. The model therefore learns
from successful passes only. That is a filtered training set and the filter
keeps the easy cases, so the round reports the retention rate and never claims
the model was trained on the same population it is scored on.

The model scores every candidate teammate and takes the best one. The softmax
runs across the candidates of one pass, so the loss asks for the right teammate
of that pass, not for "some teammate somewhere".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from soccervalue.intent import FEATURE_NAMES


class LinearIntent(nn.Module):
    """One weight per feature. The learned version of the geometric rule."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features).squeeze(-1)


class MlpIntent(nn.Module):
    """A small shared scorer over the candidate table."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features).squeeze(-1)


def label_rows(row: dict) -> tuple[np.ndarray, list[str], int | None]:
    """The candidate table of one pass, with the index of the labelled one."""
    candidates = row.get("intent_candidates") or []
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES))), [], None
    features = np.asarray([item["features"] for item in candidates], dtype=np.float64)
    players = [item["player"] for item in candidates]
    recipient = row.get("recipient")
    index = players.index(recipient) if recipient in players else None
    return features, players, index


def build_dataset(rows: list[dict], require_label: bool = True):
    """Passes turned into candidate tables with the annotated receiver as label."""
    dataset = []
    skipped = 0
    for row in rows:
        features, players, index = label_rows(row)
        if len(features) == 0:
            skipped += 1
            continue
        if require_label and index is None:
            skipped += 1
            continue
        dataset.append(
            {
                "match_id": row["match_id"],
                "event_id": row.get("event_id"),
                "success": row["success"],
                "features": features,
                "players": players,
                "xy": [item.get("xy") for item in (row.get("intent_candidates") or [])],
                "label": index,
                "rule_pick": row.get("intent_player"),
                "recipient": row.get("recipient"),
            }
        )
    return dataset, skipped


def fit_intent_model(
    dataset: list[dict],
    hidden: int = 32,
    steps: int = 600,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 0,
    model: str = "mlp",
):
    """Fit the scorer. The softmax runs over the candidates of each pass.

    The loss is a listwise cross entropy: every pass contributes the negative
    log probability it gives to the annotated recipient, with the normaliser
    taken over that pass's own candidates. The model is trained in full batch,
    so the number of steps is fixed and no held-out match is ever looked at.
    """
    if not dataset:
        raise ValueError("no training passes")
    torch.manual_seed(seed)
    width = len(FEATURE_NAMES)
    net = LinearIntent(width) if model == "linear" else MlpIntent(width, hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    stacked = np.concatenate([item["features"] for item in dataset], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    group = np.concatenate(
        [np.full(len(item["players"]), index, dtype=np.int64) for index, item in enumerate(dataset)]
    )
    offset = 0
    label_rows = []
    for item in dataset:
        label_rows.append(offset + int(item["label"]))
        offset += len(item["players"])

    features = torch.tensor((stacked - mean) / std, dtype=torch.float32)
    groups = torch.tensor(group, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    n_groups = len(dataset)

    net.train()
    for _step in range(steps):
        scores = net(features)
        highest = torch.full((n_groups,), float("-inf")).scatter_reduce_(
            0, groups, scores, reduce="amax", include_self=True
        )
        total = torch.zeros(n_groups).scatter_add_(0, groups, torch.exp(scores - highest[groups]))
        log_norm = highest + torch.log(total)
        loss = (log_norm[groups][labels] - scores[labels]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    return net, mean, std


def predict_pick(net, item: dict, mean: np.ndarray, std: np.ndarray) -> str | None:
    """The candidate the model chooses for one pass."""
    features = torch.tensor((item["features"] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        scores = net(features)
    return item["players"][int(torch.argmax(scores))]


def score_top1(dataset: list[dict], picks: dict) -> dict:
    """Top-1 accuracy of one picker, and the cases it gets wrong."""
    hits = 0
    misses = []
    for item in dataset:
        chosen = picks.get(item["event_id"])
        truth = item["players"][item["label"]] if item["label"] is not None else None
        if chosen == truth:
            hits += 1
        else:
            misses.append(
                {
                    "match_id": item["match_id"],
                    "event_id": item["event_id"],
                    "chosen": chosen,
                    "truth": truth,
                    "rule_pick": item["rule_pick"],
                }
            )
    return {
        "n": len(dataset),
        "top1": (hits / len(dataset)) if dataset else None,
        "misses": misses,
    }


def rule_picks(dataset: list[dict]) -> dict:
    """What the geometric rule chooses for each pass."""
    return {item["event_id"]: item["rule_pick"] for item in dataset}
