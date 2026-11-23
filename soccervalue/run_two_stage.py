"""Does the zone value belong in the choice of a candidate?

The position surface values a zone. The player level ranker, which reads the
availability and the openness of the teammates, decides who inside a zone the
passer would pick. The proposal is to let the zone value choose the zone and
the ranker choose the player inside it.

This round builds both and compares them on the held-out match, against the
single term baselines.

The rules below are written down before the run.

  combination  The combination must put the receiver first at least one point
          more often than the player level ranker alone does on the held-out
          match, and it must not lose on more than one of the 7 matches.
  lex    The lexicographic reading, the zone value decides the zone and the
          ranker decides the player inside it, is bounded by how often the
          zone value alone already picks the right zone. That bound is
          reported before the combination is scored.
  add    The weighted reading, the zone value added to the ranker's score, has
          its weight chosen on the validation match and is then read on the
          test match.
  report  Every method is compared per match, with the zone value alone given
          the most favourable tie break so the baseline is not understated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import choose_transform, orient_samples
from soccervalue.events import load_open_flat_passes, load_timeline
from soccervalue.intent_model import fit_intent_model
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.player_value import build_candidates, read_goalkeepers
from soccervalue.run_state_value import _prepare
from soccervalue.surface import ZoneSurface, zone_name, zone_of
from soccervalue.tracking import _read_trajectories, attach_tracking

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "two_stage"
PITCH = (105.0, 68.0)

RULES = {
    "combination": "The combination must put the receiver first at least one point more often than the player level ranker alone does on the held-out match, and it must not lose on more than one of the 7 matches.",
    "lex": "The lexicographic reading is bounded by how often the zone value alone already picks the right zone. That bound is reported before the combination is scored.",
    "add": "The weighted reading has its weight chosen on the validation match and is then read on the test match.",
    "report": "Every method is compared per match, with the zone value alone given the most favourable tie break.",
}

METHODS = ("zone_only", "ranker_only", "lexicographic", "weighted", "ray_rule")
# The weights tried for the weighted reading, on the ranker's logit scale.
WEIGHTS = np.arange(0.0, 205.0, 5.0)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scores_for(net, matrix, mean, std) -> np.ndarray:
    """The ranker's score for every candidate of one pass."""
    features = torch.tensor((matrix - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return net(features).cpu().numpy()


def prepare(item: dict, net, mean, std, surface: ZoneSurface) -> list[dict]:
    """The candidate table with the zone value and the ranker score on it."""
    scores = scores_for(net, item["features"], mean, std)
    out = []
    for index, candidate in enumerate(item["candidates"]):
        point = candidate.get("xy")
        copy = dict(candidate)
        copy["ranker_score"] = float(scores[index])
        copy["zone_value"] = surface.value(point[0], point[1], 1, PITCH) if point else 0.0
        copy["zone"] = zone_name(zone_of(point[0], point[1], PITCH)) if point else ""
        copy["perp"] = candidate["features"][1]
        out.append(copy)
    return out


def pick(prepared: list[dict], method: str, weight: float = 0.0) -> str:
    """The candidate a method chooses."""
    if not prepared:
        return ""
    if method == "zone_only":
        # the most favourable tie break for the baseline: the one nearest the ray
        return max(prepared, key=lambda item: (item["zone_value"], -item["perp"]))["player"]
    if method == "ranker_only":
        return max(prepared, key=lambda item: item["ranker_score"])["player"]
    if method == "lexicographic":
        return max(prepared, key=lambda item: (item["zone_value"], item["ranker_score"]))["player"]
    if method == "weighted":
        return max(prepared, key=lambda item: item["ranker_score"] + weight * item["zone_value"])["player"]
    if method == "ray_rule":
        return min(prepared, key=lambda item: item["perp"])["player"]
    raise ValueError(f"unknown method {method}")


def score_dataset(dataset: list[dict], net, mean, std, surface: ZoneSurface, weight: float) -> dict:
    """Every method on one set of passes."""
    hits = {method: 0 for method in METHODS}
    zone_right = 0
    for item in dataset:
        prepared = prepare(item, net, mean, std, surface)
        item["prepared"] = prepared
        best_zone = max(candidate["zone_value"] for candidate in prepared)
        zone_right += int(
            any(
                candidate["zone_value"] >= best_zone - 1e-12 and candidate["player"] == item["truth"]
                for candidate in prepared
            )
        )
        for method in METHODS:
            hits[method] += int(pick(prepared, method, weight) == item["truth"])
    total = len(dataset)
    return {
        "n": total,
        "accuracy": {method: (hits[method] / total if total else None) for method in METHODS},
        "zone_hit_rate": (zone_right / total if total else None),
        "zones": zone_shape(dataset, net, mean, std, surface),
    }


def zone_shape(dataset, net, mean, std, surface) -> dict:
    """How much the zone value can say about a candidate set.

    If the candidates of a pass sit in one zone, the zone value is the same for
    all of them and cannot say anything about which one to pick.
    """
    distinct = []
    single = 0
    spread = []
    gap_to_best = []
    for item in dataset:
        prepared = prepare(item, net, mean, std, surface)
        zones = {candidate["zone"] for candidate in prepared}
        distinct.append(len(zones))
        single += int(len(zones) == 1)
        values = [candidate["zone_value"] for candidate in prepared]
        spread.append(max(values) - min(values))
        best = max(values)
        truth_value = next(
            candidate["zone_value"] for candidate in prepared if candidate["player"] == item["truth"]
        )
        gap_to_best.append(best - truth_value)
    if not dataset:
        return {}
    return {
        "mean_distinct_zones": float(np.mean(distinct)),
        "share_with_one_zone": single / len(dataset),
        "mean_zone_value_spread": float(np.mean(spread)),
        "mean_gap_from_the_best_zone": float(np.mean(gap_to_best)),
    }


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    keepers = {}
    pitches = {}
    grouped = {split: [] for split in ("train", "val", "test")}
    timelines = {}
    trajectories = {}
    for match_id in MATCH_COMPETITION:
        keepers[match_id] = read_goalkeepers(Path(f"data/raw/{match_id}_meta.xml"))
        timelines[match_id] = load_timeline(Path(f"data/raw/{match_id}_events.xml"), match_id)
        trajectories[match_id] = _read_trajectories(Path(f"data/raw/{match_id}_tracking.xml"))
        pitches[match_id] = PITCH
        passes = load_open_flat_passes(Path(f"data/raw/{match_id}_events.xml"), match_id)
        attach_tracking(
            Path(f"data/raw/{match_id}_tracking.xml"),
            passes,
            PITCH,
            trajectories=trajectories[match_id],
        )
        grouped[split_of(match_id)].extend(passes)
        print(f"{match_id}: {len(passes)} passes", flush=True)

    transform = choose_transform(grouped["train"], PITCH[0], PITCH[1])
    rows = {
        split: orient_samples(items, transform, PITCH[0], PITCH[1], end_mode="intent")
        for split, items in grouped.items()
    }
    state_rows = _prepare(10.0, timelines, trajectories, pitches, transform, PITCH)
    surface = ZoneSurface().fit(state_rows["train"], PITCH)
    print(f"transform {transform}, surface on {len(state_rows['train'])} states", flush=True)

    datasets = {}
    for split, items in rows.items():
        dataset = []
        for row in items:
            truth = row.get("recipient")
            if not truth or not row.get("success"):
                continue
            candidates = build_candidates(row, keepers[row["match_id"]], PITCH)
            if len(candidates) < 2:
                continue
            players = [item["player"] for item in candidates]
            if truth not in players:
                continue
            dataset.append(
                {
                    "match_id": row["match_id"],
                    "event_id": row.get("event_id"),
                    "truth": truth,
                    "candidates": candidates,
                    "features": np.asarray([item["features"] for item in candidates]),
                    "players": players,
                    "label": players.index(truth),
                }
            )
        for item in dataset:
            missing = {"features", "players", "label", "match_id"} - set(item)
            if missing:
                raise KeyError(f"dataset item is missing {sorted(missing)}")
        datasets[split] = dataset
        print(f"{split}: {len(dataset)} passes with a receiver", flush=True)

    net, mean, std = fit_intent_model(datasets["train"], model="linear")

    # the weight for the weighted reading, chosen on the validation match
    best = (0.0, -1)
    for weight in WEIGHTS:
        score = score_dataset(datasets["val"], net, mean, std, surface, float(weight))["accuracy"]["weighted"]
        hits = score * len(datasets["val"]) if datasets["val"] else 0
        if hits > best[1]:
            best = (float(weight), hits)
    weight = best[0]
    results = {
        "rules": RULES,
        "methods": list(METHODS),
        "weight_chosen_on_validation": weight,
        "validation_hits_at_that_weight": best[1],
    }

    headline = {}
    for split, dataset in datasets.items():
        headline[split] = score_dataset(dataset, net, mean, std, surface, weight)
    results["headline"] = headline
    print(json.dumps(headline, indent=2), flush=True)

    # the same comparison with one match held out at a time
    every = [item for dataset in datasets.values() for item in dataset]
    per_match = {}
    for hold in sorted({item["match_id"] for item in every}):
        fold_train = [item for item in every if item["match_id"] != hold]
        fold_test = [item for item in every if item["match_id"] == hold]
        if not fold_train or not fold_test:
            continue
        fold_net, fold_mean, fold_std = fit_intent_model(fold_train, model="linear")
        scored = score_dataset(fold_test, fold_net, fold_mean, fold_std, surface, weight)
        per_match[hold] = scored["accuracy"]
        per_match[hold]["n"] = scored["n"]
        per_match[hold]["zone_hit_rate"] = scored["zone_hit_rate"]
    results["leave_one_match_out"] = per_match
    print(json.dumps(per_match, indent=2), flush=True)

    _write(OUT / "results.json", results)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
