"""Score unexecuted candidate passes with the position surface.

The surface in `soccervalue.surface` values a zone, not a player. The pass
success model in `soccervalue.run_pass_success` values a pass. This round puts
the two on the same candidate set and asks whether they agree.

For every recorded pass the candidates are the teammates that stand ahead of
the ball. Each candidate gets three scores: the value the surface gives it, the
probability the success model gives it, and the combination of the two. The
held-out match answers which of them puts the player who actually received the
ball first.

The rules below are written down before the run.

  choice   On the held-out match, the combined score must put the player who
           received the ball first at least as often as the geometric ray rule
           does. That rule is the baseline: it agrees with the annotated
           recipient on 76.0% of successful passes.
  zone     The surface ranks zones, not players, so it must also be reported on
           its own terms: how often the receiver sits in the best valued zone
           of the candidate set, and how many candidates share that zone.
  agreement    The overlap between the surface's first choice and the success
           model's first choice is reported. The two measure different things,
           the value of the target and the ease of reaching it, so a low
           overlap is not a defect, but a high one would mean the surface adds
           nothing the success model does not already have.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from soccervalue.align import choose_transform, orient_samples
from soccervalue.events import load_open_flat_passes, load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.run_pass_success import fit_linear
from soccervalue.run_state_value import _prepare
from soccervalue.surface import ZoneSurface, zone_name, zone_of
from soccervalue.tracking import _read_trajectories, attach_tracking

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidate_pass"

RULES = {
    "choice": "On the held-out match, the combined score must put the player who received the ball first at least as often as the geometric ray rule does.",
    "zone": "The surface ranks zones, not players, so the zone-level hit rate and the number of candidates sharing the top zone are reported.",
    "agreement": "The overlap between the surface's first choice and the success model's first choice is reported.",
}

# The geometric rule is the baseline this round has to beat.
BASELINE = 0.760


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def aim_direction(row: dict) -> tuple[float, float] | None:
    """The direction the ball left on, taken from the ball to the recorded target.

    The target is the teammate nearest that ray, so the line to it stands in for
    the ray itself.
    """
    ball = row.get("ball")
    end = row.get("end")
    if ball is None or end is None:
        return None
    dx = end[0] - ball[0]
    dy = end[1] - ball[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return (dx / norm, dy / norm)


def candidate_scores(row: dict, surface: ZoneSurface, success_model, pitch: tuple[float, float]) -> list[dict]:
    """Score every candidate teammate of one pass."""
    out = []
    ball = row["ball"]
    direction = aim_direction(row)
    now = surface.value(ball[0], ball[1], 1, pitch)
    for candidate in row.get("intent_candidates") or []:
        point = candidate.get("xy")
        if point is None:
            continue
        probe = dict(row)
        probe["end"] = (float(point[0]), float(point[1]))
        probability = float(success_model.predict([probe])[0])
        zone = zone_of(point[0], point[1], pitch)
        value = surface.value(point[0], point[1], 1, pitch)
        value_against = surface.value(point[0], point[1], 0, pitch)
        out.append(
            {
                "player": candidate["player"],
                "zone": zone_name(zone),
                "probability": probability,
                "value": value,
                "value_against": value_against,
                "delta_value": value - now,
                "combined": probability * value + (1.0 - probability) * value_against - now,
                "distance_to_ball": math.hypot(point[0] - ball[0], point[1] - ball[1]),
                "forwardness": forwardness(candidate.get("xy"), ball, direction),
            }
        )
    return out


def forwardness(point, ball, direction) -> float | None:
    """How far along the aim the candidate stands."""
    if point is None or direction is None:
        return None
    return (point[0] - ball[0]) * direction[0] + (point[1] - ball[1]) * direction[1]


def rank_hit(scored: list[dict], truth: str, key: str) -> tuple[bool, int]:
    """Whether the truth is first under a score, and how many share the top."""
    if not scored:
        return (False, 0)
    best = max(item[key] for item in scored)
    tied = [item for item in scored if abs(item[key] - best) < 1e-12]
    return (tied[0]["player"] == truth, len(tied))


def zone_hit(scored: list[dict], truth_zone: str) -> bool:
    """Whether the truth sits in the best valued zone of the candidate set."""
    if not scored:
        return False
    best = max(item["value"] for item in scored)
    zones = {item["zone"] for item in scored if abs(item["value"] - best) < 1e-12}
    return truth_zone in zones


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    pitches = {}
    trajectories = {}
    timelines = {}
    for match_id in MATCH_COMPETITION:
        pitches[match_id] = pitch_size(Path(f"data/raw/{match_id}_meta.xml"))
        trajectories[match_id] = _read_trajectories(Path(f"data/raw/{match_id}_tracking.xml"))
        timelines[match_id] = load_timeline(Path(f"data/raw/{match_id}_events.xml"), match_id)
    pitch = next(iter(pitches.values()))

    grouped = {split: [] for split in ("train", "val", "test")}
    for match_id in MATCH_COMPETITION:
        passes = load_open_flat_passes(Path(f"data/raw/{match_id}_events.xml"), match_id)
        attach_tracking(
            Path(f"data/raw/{match_id}_tracking.xml"),
            passes,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        grouped[split_of(match_id)].extend(passes)
        print(f"{match_id}: {len(passes)} passes", flush=True)

    transform = choose_transform(grouped["train"], pitch[0], pitch[1])
    rows = {
        split: orient_samples(items, transform, pitch[0], pitch[1], end_mode="intent")
        for split, items in grouped.items()
    }
    print(f"transform {transform}", flush=True)

    # the pass success model, fitted on the intent endpoint
    _geometry, success_model = fit_linear(rows["train"])
    # the surface, fitted on the state samples and their window xG
    state_rows = _prepare(10.0, timelines, trajectories, pitches, transform, pitch)
    surface = ZoneSurface().fit(state_rows["train"], pitch)
    print(f"surface fitted on {len(state_rows['train'])} states", flush=True)

    results = {"rules": RULES, "baseline_ray_agreement": BASELINE, "by_split": {}}

    for split in ("train", "val", "test"):
        hits = {"probability": 0, "value": 0, "combined": 0, "zone": 0}
        ties = []
        agreements = []
        correlations = []
        n_truth = 0
        candidate_counts = []
        surface_is_depth = 0
        probability_is_safety = 0
        for row in rows[split]:
            truth = row.get("recipient")
            if not truth or not row.get("success"):
                continue
            scored = candidate_scores(row, surface, success_model, pitch)
            if len(scored) < 2:
                continue
            truth_zone = None
            for item in scored:
                if item["player"] == truth:
                    truth_zone = item["zone"]
            if truth_zone is None:
                continue
            n_truth += 1
            candidate_counts.append(len(scored))
            for key in ("probability", "value", "combined"):
                hit, tied = rank_hit(scored, truth, key)
                hits[key] += int(hit)
                if key == "combined":
                    ties.append(tied)
            hits["zone"] += int(zone_hit(scored, truth_zone))
            best_probability = max(item["probability"] for item in scored)
            best_value = max(item["value"] for item in scored)
            first_probability = [item for item in scored if abs(item["probability"] - best_probability) < 1e-12][0]
            first_value = [item for item in scored if abs(item["value"] - best_value) < 1e-12][0]
            agreements.append(int(first_probability["player"] == first_value["player"]))
            values = [item["value"] for item in scored]
            probabilities = [item["probability"] for item in scored]
            if len(set(values)) > 1 and len(set(probabilities)) > 1:
                correlations.append(float(np.corrcoef(values, probabilities)[0, 1]))
            forward = [item["forwardness"] for item in scored if item["forwardness"] is not None]
            if len(forward) == len(scored) and forward:
                most_forward = max(scored, key=lambda item: item["forwardness"])
                if most_forward["player"] == first_value["player"]:
                    surface_is_depth += 1
                closest = min(scored, key=lambda item: item["distance_to_ball"])
                if closest["player"] == first_probability["player"]:
                    probability_is_safety += 1
        entry = {
            "n_with_a_receiver": n_truth,
            "first_choice_rate": {key: (hits[key] / n_truth if n_truth else None) for key in hits},
            "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else None,
            "mean_candidates_sharing_the_top": float(np.mean(ties)) if ties else None,
            "same_first_choice_rate": float(np.mean(agreements)) if agreements else None,
            "mean_correlation_between_value_and_probability": float(np.mean(correlations)) if correlations else None,
            "surface_first_choice_is_the_most_forward_candidate": (surface_is_depth / n_truth) if n_truth else None,
            "probability_first_choice_is_the_closest_candidate": (probability_is_safety / n_truth) if n_truth else None,
        }
        results["by_split"][split] = entry
        print(split, json.dumps(entry, indent=2, ensure_ascii=False), flush=True)

    _write(OUT / "results.json", results)
    print(json.dumps(results["by_split"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
