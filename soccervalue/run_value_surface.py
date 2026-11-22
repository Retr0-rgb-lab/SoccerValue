"""A position value surface, and whether it beats the shot-based value.

The shot label is sparse: a shot happens in about one state in fifty, and the
side that rarely shoots has too few events for the relation to be identified.
The xG window label is graded, and a surface that only says where the ball is
uses every state. This round asks whether that buys anything.

The rules below are written down before the run.

  identifiability  The opponent-side relation must have the same sign in all
          three splits. The binary shot label failed this: the measured
          conditional rates were 0.79, 0.43 and 1.84 times on train, validation
          and test.
  shape    The surface must be monotone in depth on the held-out match, and the
          centre band must beat the wide band inside the attacking third.
  delta    The surface's value change must separate a successful pass from a
          failed one in at least 6 of the 7 matches, and it must not be worse
          than the shot-based value model on the same passes.
  candidate    The surface must be evaluable at zones where no pass was ever
          recorded. That is what lets it value a pass that was not played.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from soccervalue.align import choose_transform
from soccervalue.ball_flight import fit_drag
from soccervalue.events import load_open_flat_passes, load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.pass_value import add_values, collect_pass_states
from soccervalue.run_pass_success import ensure_file
from soccervalue.run_state_value import _prepare
from soccervalue.state_frames import attach_state_frame
from soccervalue.state_value import build_state_samples
from soccervalue.surface import (
    DEPTH_EDGES,
    DEPTH_NAMES,
    LATERAL_EDGE,
    LATERAL_NAMES,
    ZONE_COUNT,
    ZoneSurface,
    zone_name,
    zone_of,
)
from soccervalue.touch_head import fit_touch
from soccervalue.tracking import _read_trajectories, attach_tracking
from soccervalue.value_model import SimpleStateValue
from soccervalue.world_segments import build_segments, in_play_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "value_surface"

RULES = {
    "identifiability": "The opponent-side relation must have the same sign in all three splits.",
    "shape": "The surface must be monotone in depth on the held-out match, and the centre band must beat the wide band inside the attacking third.",
    "delta": "The surface's value change must separate a successful pass from a failed one in at least 6 of the 7 matches, and it must not be worse than the shot-based value model on the same passes.",
    "candidate": "The surface must be evaluable at zones where no pass was ever recorded.",
}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def conditional_rates(rows: list[dict]) -> dict:
    """The window xG of each side, split by who holds the ball.

    This is measured before any model is fitted. It is the question the binary
    shot label could not answer.
    """
    out = {}
    for label, key_for, key_against in (
        ("sum", "xg_for_sum", "xg_against_sum"),
        ("product", "xg_for_product", "xg_against_product"),
    ):
        entry = {}
        for side, key in (("for", key_for), ("against", key_against)):
            for possession in (1, 0):
                chosen = [row[key] for row in rows if row["possession_by_t"] == possession]
                shots = sum(1 for row in rows if row["possession_by_t"] == possession and row[key] > 0)
                entry[f"{side}|possession_{possession}"] = {
                    "n": len(chosen),
                    "mean_window_xg": float(np.mean(chosen)) if chosen else None,
                    "windows_with_a_shot": shots,
                }
        # The evaluated team should produce more when it holds the ball, and the
        # opponent should produce more when it does not. Both effects must point
        # the same way in every split.
        for_mean = entry["for|possession_1"]["mean_window_xg"]
        for_mean_without = entry["for|possession_0"]["mean_window_xg"]
        against_mean = entry["against|possession_1"]["mean_window_xg"]
        against_mean_without = entry["against|possession_0"]["mean_window_xg"]
        entry["for_effect"] = (for_mean - for_mean_without) if (for_mean is not None and for_mean_without is not None) else None
        entry["for_ratio"] = (for_mean / for_mean_without) if (for_mean and for_mean_without) else None
        entry["against_effect"] = (
            (against_mean_without - against_mean) if (against_mean is not None and against_mean_without is not None) else None
        )
        entry["against_ratio"] = (against_mean / against_mean_without) if (against_mean and against_mean_without) else None
        out[label] = entry
    return out


def sign_agreement(rates: dict, label_form: str, key: str) -> dict:
    """The effect of holding the ball, per split, and whether they agree."""
    values = {split: rates[split][label_form].get(key) for split in rates}
    kept = {split: value for split, value in values.items() if value is not None}
    if not kept:
        return {"per_split": values, "agreement": "no data"}
    positive = sum(1 for value in kept.values() if value > 0)
    agreement = (
        f"same direction in {positive}/{len(kept)} splits"
        if positive == len(kept)
        else f"directions disagree ({positive}/{len(kept)} positive)"
    )
    return {"per_split": values, "agreement": agreement, "all_positive": positive == len(kept)}


def separation(rows: list[dict], key: str) -> dict:
    good = [row[key] for row in rows if row["success"] == 1]
    bad = [row[key] for row in rows if row["success"] == 0]
    return {
        "n_success": len(good),
        "n_failure": len(bad),
        "mean_success": float(np.mean(good)) if good else None,
        "mean_failure": float(np.mean(bad)) if bad else None,
        "gap": (float(np.mean(good)) - float(np.mean(bad))) if good and bad else None,
    }


def per_match_separation(rows: list[dict], key: str) -> dict:
    out = {}
    for match_id in sorted({row["match_id"] for row in rows}):
        out[match_id] = separation([row for row in rows if row["match_id"] == match_id], key)
    return out


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    timelines = {}
    trajectories = {}
    pitches = {}
    for match_id in MATCH_COMPETITION:
        print(f"loading {match_id}", flush=True)
        timelines[match_id] = load_timeline(ensure_file(match_id, "events"), match_id)
        pitches[match_id] = pitch_size(ensure_file(match_id, "meta"))
        trajectories[match_id] = _read_trajectories(ensure_file(match_id, "tracking"))
    pitch = next(iter(pitches.values()))

    raw_for_shift = []
    for match_id, events in timelines.items():
        samples = build_state_samples(events, window_s=10.0)
        attach_state_frame(
            ensure_file(match_id, "tracking"),
            samples,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        if split_of(match_id) != "train":
            continue
        for sample in samples:
            if sample.get("track_ball") is None:
                continue
            raw_for_shift.append(
                {"start_x": sample["x"], "start_y": sample["y"], "track_ball": (*sample["track_ball"], None)}
            )
    transform = choose_transform(raw_for_shift, pitch[0], pitch[1])
    grouped = _prepare(10.0, timelines, trajectories, pitches, transform, pitch)
    print(f"transform {transform}", flush=True)
    for split, rows in grouped.items():
        print(
            f"  {split}: {len(rows)} states, windows with a shot "
            f"{sum(1 for row in rows if row['y_for'] or row['y_against'])}, "
            f"total window xG {sum(row['xg_for_sum'] for row in rows):.2f}",
            flush=True,
        )

    # 1. can the dense label be identified at all
    rates = {split: conditional_rates(rows) for split, rows in grouped.items()}
    results = {
        "rules": RULES,
        "conditional_rates": rates,
        "identifiability": {
            "for_side_sum": sign_agreement(rates, "sum", "for_effect"),
            "against_side_sum": sign_agreement(rates, "sum", "against_effect"),
            "for_side_product": sign_agreement(rates, "product", "for_effect"),
            "against_side_product": sign_agreement(rates, "product", "against_effect"),
        },
    }
    print(json.dumps(results["identifiability"], indent=2, ensure_ascii=False), flush=True)

    # 2. the surface, fitted on the training matches only
    surface = ZoneSurface().fit(grouped["train"], pitch)
    _write(OUT / "surface.json", {zone_name(index): surface.values.get((index, 1)) for index in range(ZONE_COUNT)})
    test_shape = surface.shape(pitch)
    results["surface_shape_on_the_fitted_surface"] = test_shape
    # the same shape read from the held-out match, with the train-fitted zones
    held_out = {}
    for index in range(ZONE_COUNT):
        chosen = [
            row for row in grouped["test"]
            if zone_of(row["ball"][0], row["ball"][1], pitch) == index
        ]
        held_out[zone_name(index)] = {
            "states": len(chosen),
            "mean_window_xg": float(
                np.mean([row["xg_for_sum"] - row["xg_against_sum"] for row in chosen])
            )
            if chosen
            else None,
        }
    results["shape_measured_on_the_held_out_match"] = held_out
    print(json.dumps(test_shape, indent=2, ensure_ascii=False), flush=True)

    # 3. the world model, for the pass states
    segments_by_match = {}
    for match_id in MATCH_COMPETITION:
        window = in_play_window(timelines[match_id])
        if window is None:
            continue
        ball, players = trajectories[match_id]
        segments_by_match[match_id] = build_segments(ball, players, pitch, window[0], window[1])
    train_segments = [
        segment
        for match_id, items in segments_by_match.items()
        if split_of(match_id) == "train"
        for segment in items
    ]
    drag = fit_drag(train_segments)
    labelled = [segment for segment in train_segments if segment.get("toucher") is not None]
    touch_model, touch_mean, touch_std = fit_touch(labelled)

    passes = []
    for match_id in MATCH_COMPETITION:
        found = load_open_flat_passes(ensure_file(match_id, "events"), match_id)
        attach_tracking(
            ensure_file(match_id, "tracking"),
            found,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        passes.extend(found)
    states = collect_pass_states(
        passes, segments_by_match, touch_model, touch_mean, touch_std, drag, pitch, transform
    )
    print(f"passes {len(passes)}, linked {len(states)}", flush=True)

    # 4. the surface's value change against the shot-based value model
    value_model = SimpleStateValue()
    value_model.fit(grouped["train"])
    add_values(states, value_model)
    for state in states:
        state["surface_value_before"] = surface.value(state["ball_before"][0], state["ball_before"][1], 1, pitch)
        state["surface_value_after"] = surface.value(
            state["ball_after_real"][0], state["ball_after_real"][1], state["possession_after_real"], pitch
        )
        state["surface_change"] = state["surface_value_after"] - state["surface_value_before"]
        state["model_change"] = state["value_change_real"]

    by_split = {
        split: [state for state in states if split_of(state["match_id"]) == split]
        for split in ("train", "val", "test")
    }
    results["pass_value"] = {
        "surface_change": {split: separation(rows, "surface_change") for split, rows in by_split.items()},
        "model_change": {split: separation(rows, "model_change") for split, rows in by_split.items()},
        "surface_change_per_match": per_match_separation(states, "surface_change"),
        "model_change_per_match": per_match_separation(states, "model_change"),
    }
    for name in ("surface_change", "model_change"):
        per_match = results["pass_value"][f"{name}_per_match"]
        positive = sum(1 for item in per_match.values() if item["gap"] is not None and item["gap"] > 0)
        results["pass_value"][f"{name}_matches_positive"] = f"{positive}/{len(per_match)}"

    _write(OUT / "states.json", states)
    _write(OUT / "state_rows.json", {split: rows for split, rows in grouped.items()})

    # 5. the payoff: zones where no pass was recorded
    recorded = {
        (round(state["ball_after_real"][0], 1), round(state["ball_after_real"][1], 1))
        for state in states
    }
    candidates = []
    for index in range(ZONE_COUNT):
        for possession in (1, 0):
            candidates.append({"zone": zone_name(index), "possession": possession})
    results["candidate_interface"] = {
        "zones": ZONE_COUNT,
        "recorded_pass_endpoints": len(recorded),
        "evaluable_candidate_cells": len(candidates),
        "example": [
            {
                "zone": item["zone"],
                "possession": item["possession"],
                "value": surface.value(
                    *surface_zone_centre(item["zone"], pitch),
                    item["possession"],
                    pitch,
                ),
            }
            for item in candidates[:8]
        ],
    }

    _write(OUT / "results.json", results)
    print(json.dumps(results["pass_value"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


def surface_zone_centre(name: str, pitch: tuple[float, float]) -> tuple[float, float]:
    """A representative point inside a zone, for reading the surface."""
    depth, lateral = name.rsplit("_", 1)
    index = DEPTH_NAMES.index(depth)
    pitch_x, pitch_y = pitch
    edges = (0.0,) + DEPTH_EDGES + (pitch_x,)
    x = (edges[index] + edges[index + 1]) / 2.0
    if lateral == LATERAL_NAMES[1]:
        y = pitch_y / 2.0
    else:
        y = pitch_y / 2.0 + LATERAL_EDGE + (pitch_y / 2.0 - LATERAL_EDGE) / 2.0
    return (x, y)


if __name__ == "__main__":
    sys.exit(main())
