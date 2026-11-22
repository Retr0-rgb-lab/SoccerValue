"""Put the world model and the value model together.

The world model knows how the ball moves and who touches it next. The value
model knows what a state is worth. This round asks whether the successor
state the world model produces has the value the recorded successor state has,
and whether the value model separates the states that follow a successful pass
from the states that follow a failed one.

The rules below are written down before the run.

  chain   The value of the simulated successor state must be closer to the value
          of the recorded successor state than the value of the pre-pass state
          is, on the test match. The sign of the value change must agree
          between the two in at least 6 of the 7 matches.
  value   The value after a successful pass must be higher than the value after
          a failed pass, on the test match and in at least 6 of the 7 matches.
  report  How much of that separation survives with the possession flag held
          fixed is reported either way. The world model does not see the label.
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
from soccervalue.pass_value import add_values, collect_pass_states, value_row
from soccervalue.run_pass_success import ensure_file
from soccervalue.state_frames import attach_state_frame, orient_state
from soccervalue.state_value import build_state_samples
from soccervalue.touch_head import fit_touch
from soccervalue.tracking import _read_trajectories, attach_tracking
from soccervalue.value_model import SimpleStateValue
from soccervalue.world_segments import build_segments, in_play_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pass_value"

RULES = {
    "chain": "The value of the simulated successor state must be closer to the value of the recorded successor state than the value of the pre-pass state is, on the test match. The sign of the value change must agree between the two in at least 6 of the 7 matches.",
    "value": "The value after a successful pass must be higher than the value after a failed pass, on the test match and in at least 6 of the 7 matches.",
    "report": "How much of that separation survives with the possession flag held fixed is reported either way.",
}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def value_of(model: SimpleStateValue, triples: list[tuple]) -> np.ndarray:
    """The two heads for each state: the team's shot chance and the other's.

    A triple is (ball position, possession flag, pitch). The value is the
    difference between the two columns.
    """
    rows = [value_row(ball, possession, pitch) for ball, possession, pitch in triples]
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return model.predict(rows)


def separation(rows: list[dict]) -> dict:
    """The value after a successful pass against the value after a failed one."""
    good = [row["value_after_real"] for row in rows if row["success"] == 1]
    bad = [row["value_after_real"] for row in rows if row["success"] == 0]
    return {
        "n_success": len(good),
        "n_failure": len(bad),
        "mean_after_success": float(np.mean(good)) if good else None,
        "mean_after_failure": float(np.mean(bad)) if bad else None,
        "gap": (float(np.mean(good)) - float(np.mean(bad))) if good and bad else None,
    }


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
    if any(size != pitch for size in pitches.values()):
        raise RuntimeError(f"pitch sizes differ: {pitches}")

    # the value model, trained on real states from the training matches
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
    print(f"transform {transform}", flush=True)

    value_rows = {"train": [], "val": [], "test": []}
    for match_id, events in timelines.items():
        samples = build_state_samples(events, window_s=10.0)
        attach_state_frame(
            ensure_file(match_id, "tracking"),
            samples,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        for sample in samples:
            row = orient_state(sample, transform, pitch[0], pitch[1])
            if row is not None:
                value_rows[split_of(match_id)].append(row)
    value_model = SimpleStateValue()
    value_model.fit(value_rows["train"])
    value_feature_names = [
        "ball_x_over_pitch_x",
        "ball_y_over_pitch_y",
        "ball_x_over_pitch_x_again",
        "distance_to_opponent_goal_over_pitch_x",
        "possession_by_t",
    ]
    value_coefficients = {
        "for": {
            name: float(value)
            for name, value in zip(value_feature_names, value_model.for_model.coef_[0])
        },
        "against": {
            name: float(value)
            for name, value in zip(value_feature_names, value_model.against_model.coef_[0])
        },
        "for_intercept": float(value_model.for_model.intercept_[0]),
        "against_intercept": float(value_model.against_model.intercept_[0]),
    }
    print(
        f"value model on {len(value_rows['train'])} states, "
        f"for positives {sum(row['y_for'] for row in value_rows['train'])}",
        flush=True,
    )

    # the world model: ground-ball segments, a drag coefficient and a touch ranker
    segments_by_match = {}
    segments = {"train": [], "val": [], "test": []}
    for match_id in MATCH_COMPETITION:
        window = in_play_window(timelines[match_id])
        if window is None:
            continue
        ball, players = trajectories[match_id]
        built = build_segments(ball, players, pitch, window[0], window[1])
        segments_by_match[match_id] = built
        segments[split_of(match_id)].extend(built)
    drag = fit_drag(segments["train"])
    labelled = [segment for segment in segments["train"] if segment.get("toucher") is not None]
    touch_model, touch_mean, touch_std = fit_touch(labelled)
    print(f"drag {drag}, touch model on {len(labelled)} segments", flush=True)

    # the passes, with their intent
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
    print(f"passes {len(passes)}", flush=True)

    states = collect_pass_states(
        passes, segments_by_match, touch_model, touch_mean, touch_std, drag, pitch, transform
    )
    print(f"passes with a linked successor state: {len(states)}", flush=True)

    add_values(states, value_model)

    by_split = {split: [state for state in states if split_of(state["match_id"]) == split] for split in segments}
    results = {
        "rules": RULES,
        "drag": drag,
        "value_model_coefficients": value_coefficients,
        "counts": {
            "passes": len(passes),
            "linked": len(states),
            "by_split": {split: len(rows) for split, rows in by_split.items()},
            "touch_top1_on_linked": float(
                np.mean([state["toucher_correct"] for state in states if state["toucher_correct"] is not None])
            ),
        },
    }

    # the chain: does the simulated successor carry the right value
    chain = {}
    for split, rows in by_split.items():
        if not rows:
            continue
        sim = np.asarray([row["value_after_sim"] for row in rows])
        real = np.asarray([row["value_after_real"] for row in rows])
        before = np.asarray([row["value_before"] for row in rows])
        chain[split] = {
            "n": len(rows),
            "mae_simulated_vs_recorded": float(np.mean(np.abs(sim - real))),
            "mae_prepass_vs_recorded": float(np.mean(np.abs(before - real))),
            "mean_value_recorded": float(np.mean(real)),
            "mean_value_simulated": float(np.mean(sim)),
            "mean_value_prepass": float(np.mean(before)),
        }
    results["chain"] = chain

    # the value: does it separate the successors of a good pass from a bad one
    results["value_separation"] = {split: separation(rows) for split, rows in by_split.items()}
    per_match = {}
    for match_id in sorted({state["match_id"] for state in states}):
        rows = [state for state in states if state["match_id"] == match_id]
        per_match[match_id] = separation(rows)
    results["value_separation_per_match"] = per_match
    positive = sum(1 for entry in per_match.values() if entry["gap"] is not None and entry["gap"] > 0)
    results["matches_with_a_positive_gap"] = f"{positive}/{len(per_match)}"

    # what the separation is made of
    parts = {}
    for split, rows in by_split.items():
        if not rows:
            continue
        good = [row for row in rows if row["success"] == 1]
        bad = [row for row in rows if row["success"] == 0]
        parts[split] = {
            "mean_value_change_real_success": float(np.mean([row["value_change_real"] for row in good])) if good else None,
            "mean_value_change_real_failure": float(np.mean([row["value_change_real"] for row in bad])) if bad else None,
            "mean_value_change_sim_success": float(np.mean([row["value_change_sim"] for row in good])) if good else None,
            "mean_value_change_sim_failure": float(np.mean([row["value_change_sim"] for row in bad])) if bad else None,
            "mean_position_only_gap": (
                float(np.mean([row["value_after_real_same_team"] for row in good]))
                - float(np.mean([row["value_after_real_same_team"] for row in bad]))
            )
            if good and bad
            else None,
            "mean_possession_effect_on_value_failure": float(
                np.mean([row["possession_effect_on_value"] for row in bad])
            )
            if bad
            else None,
            "mean_p_for_change_success": float(
                np.mean([row["p_for_after_real"] - row["p_for_before"] for row in good])
            )
            if good
            else None,
            "mean_p_against_change_success": float(
                np.mean([row["p_against_after_real"] - row["p_against_before"] for row in good])
            )
            if good
            else None,
            "mean_p_for_change_failure": float(
                np.mean([row["p_for_after_real"] - row["p_for_before"] for row in bad])
            )
            if bad
            else None,
            "mean_p_against_change_failure": float(
                np.mean([row["p_against_after_real"] - row["p_against_before"] for row in bad])
            )
            if bad
            else None,
            "mean_ball_position_error_m": float(np.mean([row["ball_position_error_m"] for row in rows])),
            "sign_agreement_of_the_value_change": float(
                np.mean([np.sign(row["value_change_real"]) == np.sign(row["value_change_sim"]) for row in rows])
            ),
        }
    results["decomposition"] = parts

    _write(OUT / "states.json", states)
    _write(OUT / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
