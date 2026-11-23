"""What the receiver has that the other candidates do not.

The position surface values a zone and could not pick a player. This round
looks for the player level difference that separates the player who received
the ball from the teammates he was competing with, and it adds the opponent's
defensive lines as a feature.

The rules below are written down before the run.

  feature  A feature qualifies when the receiver holds the better value in at
           least 55% of the comparisons on the held-out match and the direction
           agrees in at least 6 of the 7 matches. The better direction of each
           feature is fixed in the table below, from the football reading of
           what should make a teammate the better target.
  model    GO only if the trained ranker puts the receiver first at least one
           point more often than the geometric ray rule does on the held-out
           match, and it does not lose on more than one of the 7 matches.
  report   The comparisons are also reported on the passes where the ray rule
           gets it wrong, because that is where the difference has to show.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from soccervalue.align import choose_transform, orient_samples
from soccervalue.events import load_open_flat_passes
from soccervalue.intent_model import build_dataset, fit_intent_model, predict_pick
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.player_value import (
    FEATURE_NAMES,
    build_candidates,
    read_goalkeepers,
)
from soccervalue.tracking import _read_trajectories, attach_tracking

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "player_value"

RULES = {
    "feature": "A feature qualifies when the receiver holds the better value in at least 55% of the comparisons on the held-out match and the direction agrees in at least 6 of the 7 matches.",
    "model": "GO only if the trained ranker puts the receiver first at least one point more often than the geometric ray rule does on the held-out match, and it does not lose on more than one of the 7 matches.",
    "report": "The comparisons are also reported on the passes where the ray rule gets it wrong.",
}

# The features that describe the opponent's defensive lines. They are trained
# as a group and then taken out again, so the round can say whether they earn
# their place.
LINE_FEATURES = (
    "distance_to_last_line_m",
    "lines_in_front",
    "lines_crossed_by_the_pass",
    "gap_to_next_line_m",
    "offside_line_distance_m",
)

# Which direction of each feature should favour the receiver.
BETTER_HIGHER = {
    "along_m": True,
    "perp_m": False,
    "distance_to_ball_m": False,
    "nearest_defender_to_candidate_m": True,
    "lane_clearance_m": True,
    "defenders_within_3m_of_lane": False,
    "candidate_speed_mps": True,
    "velocity_along_ray_mps": True,
    "distance_to_opponent_goal_m": False,
    "goal_angle_rad": True,
    "lateral_from_centre_m": False,
    "distance_to_last_line_m": True,
    "lines_in_front": True,
    "lines_crossed_by_the_pass": True,
    "gap_to_next_line_m": True,
    "offside_line_distance_m": False,
}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def comparisons(candidates: list[dict], truth: str) -> list[tuple[int, float]]:
    """The feature difference between the receiver and each other candidate."""
    receiver = next((item for item in candidates if item["player"] == truth), None)
    if receiver is None:
        return []
    out = []
    for item in candidates:
        if item["player"] == truth:
            continue
        for index, name in enumerate(FEATURE_NAMES):
            out.append((index, receiver["features"][index] - item["features"][index]))
    return out


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    keepers = {}
    grouped = {split: [] for split in ("train", "val", "test")}
    for match_id in MATCH_COMPETITION:
        keepers[match_id] = read_goalkeepers(Path(f"data/raw/{match_id}_meta.xml"))
        trajectories = _read_trajectories(Path(f"data/raw/{match_id}_tracking.xml"))
        passes = load_open_flat_passes(Path(f"data/raw/{match_id}_events.xml"), match_id)
        attach_tracking(
            Path(f"data/raw/{match_id}_tracking.xml"),
            passes,
            (105.0, 68.0),
            trajectories=trajectories,
        )
        grouped[split_of(match_id)].extend(passes)
        print(f"{match_id}: {len(passes)} passes, keepers {keepers[match_id]}", flush=True)

    transform = choose_transform(grouped["train"], 105.0, 68.0)
    rows = {
        split: orient_samples(items, transform, 105.0, 68.0, end_mode="intent")
        for split, items in grouped.items()
    }
    print(f"transform {transform}", flush=True)

    # 1. what separates the receiver from the other candidates
    results = {"rules": RULES, "feature_names": list(FEATURE_NAMES), "per_split": {}}
    datasets = {}
    for split, items in rows.items():
        wins = {name: 0 for name in FEATURE_NAMES}
        losses = {name: 0 for name in FEATURE_NAMES}
        counts = {name: 0 for name in FEATURE_NAMES}
        dataset = []
        for row in items:
            truth = row.get("recipient")
            if not truth or not row.get("success"):
                continue
            candidates = build_candidates(row, keepers[row["match_id"]], (105.0, 68.0))
            if len(candidates) < 2:
                continue
            player_ids = [item["player"] for item in candidates]
            if truth not in player_ids:
                continue
            ray_pick = min(candidates, key=lambda item: item["features"][1])["player"]
            diffs = comparisons(candidates, truth)
            if not diffs:
                continue
            for index, difference in diffs:
                name = FEATURE_NAMES[index]
                sign = 1 if difference > 0 else (-1 if difference < 0 else 0)
                counts[name] += 1
                wins[name] += int(sign == (1 if BETTER_HIGHER[name] else -1))
                losses[name] += int(sign == (-1 if BETTER_HIGHER[name] else 1))
            entry = {
                "match_id": row["match_id"],
                "event_id": row.get("event_id"),
                "features": np.asarray([item["features"] for item in candidates]),
                "players": player_ids,
                "label": player_ids.index(truth),
                "ray_pick": ray_pick,
                "ray_correct": ray_pick == truth,
            }
            dataset.append(entry)
        datasets[split] = dataset
        results["per_split"][split] = {
            "n_passes": len(dataset),
            "receiver_better_rate": {
                name: (wins[name] / counts[name]) if counts[name] else None for name in FEATURE_NAMES
            },
            "receiver_worse_rate": {
                name: (losses[name] / counts[name]) if counts[name] else None for name in FEATURE_NAMES
            },
            "ray_rule_accuracy": (
                sum(1 for item in dataset if item["ray_correct"]) / len(dataset) if dataset else None
            ),
        }
        print(split, json.dumps(results["per_split"][split]["receiver_better_rate"], indent=2), flush=True)

    # the same, restricted to the passes where the ray rule is wrong
    results["where_the_ray_rule_fails"] = {}
    for split, dataset in datasets.items():
        failing = [item for item in dataset if not item["ray_correct"]]
        wins = {name: 0 for name in FEATURE_NAMES}
        total = 0
        for item in failing:
            receiver = item["features"][item["label"]]
            for other in range(len(item["features"])):
                if other == item["label"]:
                    continue
                total += 1
                for index, name in enumerate(FEATURE_NAMES):
                    difference = receiver[index] - item["features"][other][index]
                    sign = 1 if difference > 0 else (-1 if difference < 0 else 0)
                    wins[name] += int(sign == (1 if BETTER_HIGHER[name] else -1))
        results["where_the_ray_rule_fails"][split] = {
            "n_passes": len(failing),
            "receiver_better_rate": {name: (wins[name] / total) if total else None for name in FEATURE_NAMES},
        }
        print(f"where the ray rule fails ({split}):", json.dumps(results["where_the_ray_rule_fails"][split], indent=2), flush=True)

    # 2. a ranker built from the player level features
    train = datasets["train"]
    net, mean, std = fit_intent_model(train, model="linear")
    print("linear ranker trained", flush=True)
    results["linear_weights"] = {
        name: float(value) for name, value in zip(FEATURE_NAMES, net.score.weight.detach().cpu().numpy().ravel())
    }

    def score_with(model, mean_values, std_values, dataset):
        hits = 0
        for item in dataset:
            probe = {
                "features": item["features"],
                "players": item["players"],
                "label": item["label"],
            }
            if predict_pick(model, probe, mean_values, std_values) == item["players"][item["label"]]:
                hits += 1
        return hits / len(dataset) if dataset else None

    accuracy = {}
    for split, dataset in datasets.items():
        if not dataset:
            continue
        hits = score_with(net, mean, std, dataset)
        ray = sum(1 for item in dataset if item["ray_correct"]) / len(dataset)
        accuracy[split] = {
            "n": len(dataset),
            "ranker_top1": hits,
            "ray_rule_top1": ray,
            "delta": hits - ray,
        }
    results["accuracy"] = accuracy
    print(json.dumps(accuracy, indent=2), flush=True)

    # 3. the same comparison with one match held out at a time
    every = [item for dataset in datasets.values() for item in dataset]

    def drop_columns(dataset, names):
        keep = [index for index, name in enumerate(FEATURE_NAMES) if name not in names]
        out = []
        for item in dataset:
            copy = dict(item)
            copy["features"] = np.asarray(
                [[row[index] for index in keep] for row in item["features"]]
            )
            out.append(copy)
        return out

    def leave_one_out(dataset, columns=None):
        per_match = {}
        for hold in sorted({item["match_id"] for item in dataset}):
            fold_train = [item for item in dataset if item["match_id"] != hold]
            fold_test = [item for item in dataset if item["match_id"] == hold]
            if not fold_train or not fold_test:
                continue
            if columns is not None:
                fold_train = drop_columns(fold_train, columns)
                fold_test = drop_columns(fold_test, columns)
            fold_net, fold_mean, fold_std = fit_intent_model(fold_train, model="linear")
            hits = score_with(fold_net, fold_mean, fold_std, fold_test)
            ray = sum(1 for item in fold_test if item["ray_correct"]) / len(fold_test)
            per_match[hold] = {
                "n": len(fold_test),
                "ranker_top1": hits,
                "ray_rule_top1": ray,
                "delta": hits - ray,
            }
        return per_match

    per_match = leave_one_out(every)
    results["leave_one_match_out"] = per_match
    results["matches_where_the_ranker_loses"] = [
        match for match, entry in per_match.items() if entry["delta"] < 0
    ]
    print(json.dumps(per_match, indent=2), flush=True)

    # 4. the same ranker without the defensive line features
    without_lines = leave_one_out(every, LINE_FEATURES)
    results["leave_one_match_out_without_the_line_features"] = without_lines
    results["matches_where_the_ranker_without_lines_loses"] = [
        match for match, entry in without_lines.items() if entry["delta"] < 0
    ]
    results["line_feature_contribution"] = {
        match: per_match[match]["ranker_top1"] - without_lines[match]["ranker_top1"]
        for match in per_match
    }
    print(json.dumps(without_lines, indent=2), flush=True)

    _write(OUT / "results.json", results)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
