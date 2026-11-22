"""Make the two value heads symmetric, then check whether the absolute value
separates a successful pass from a failed one.

The separate heads learned different weights for possession. The head that
predicts the evaluated team's shots weighed it at +0.492 and the head that
predicts the opponent's shots at only -0.119, so a state where the opponent had
the ball kept the same value as a state where the team had it, and the
absolute value after a failed pass came out the same as after a successful one.

This round measures the data first, then fits three value models and scores
them on both the state-value protocol and the pass-value protocol.

The rules below are written down before the run.

  no_regression  A candidate may not be worse than the separate heads on the
                 state-value protocol. The for head and the against head must
                 each keep a test Brier no more than 0.001 worse, and neither
                 AUROC may fall by more than 0.02.
  selection      Among the candidates that pass no_regression on the validation
                 match, take the one with the larger absolute-value gap on
                 that match. The test match is then read for the chosen one.
  separation     The chosen model must have the value after a successful pass
                 higher than the value after a failed one on the test match
                 and in at least 6 of the 7 matches.
  chain          The value of the simulated successor state must stay closer
                 to the recorded successor value than the pre-pass value is.
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
from soccervalue.metrics import binary_scores
from soccervalue.pass_value import add_values, collect_pass_states
from soccervalue.run_pass_success import ensure_file
from soccervalue.run_state_value import _prepare
from soccervalue.state_frames import attach_state_frame
from soccervalue.state_value import build_state_samples
from soccervalue.touch_head import fit_touch
from soccervalue.tracking import _read_trajectories, attach_tracking
from soccervalue.value_model import SimpleStateValue, SymmetricStateValue
from soccervalue.world_segments import build_segments, in_play_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "value_symmetric"

RULES = {
    "no_regression": "A candidate may not be worse than the separate heads on the state-value protocol: each head keeps a test Brier no more than 0.001 worse and no AUROC falls by more than 0.02.",
    "selection": "Among the candidates that pass no_regression on the validation match, take the one with the larger absolute-value gap there. The test match is read for the chosen one.",
    "separation": "The chosen model must have the value after a successful pass higher than after a failed one on the test match and in at least 6 of the 7 matches.",
    "chain": "The value of the simulated successor state must stay closer to the recorded successor value than the pre-pass value is.",
}

CANDIDATES = {
    "separate": lambda: SimpleStateValue(),
    "symmetric_slopes": lambda: SymmetricStateValue("slopes"),
    "symmetric_all": lambda: SymmetricStateValue("all"),
}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def head_scores(rows: list[dict], probabilities: np.ndarray) -> dict:
    out = {}
    for head, column, key in (("for", 0, "y_for"), ("against", 1, "y_against")):
        labels = np.asarray([row[key] for row in rows])
        out[head] = binary_scores(labels, probabilities[:, column])
    return out


def separation(states: list[dict], key: str) -> dict:
    good = [state[key] for state in states if state["success"] == 1]
    bad = [state[key] for state in states if state["success"] == 0]
    return {
        "n_success": len(good),
        "n_failure": len(bad),
        "mean_success": float(np.mean(good)) if good else None,
        "mean_failure": float(np.mean(bad)) if bad else None,
        "gap": (float(np.mean(good)) - float(np.mean(bad))) if good and bad else None,
    }


def per_match_separation(states: list[dict], key: str) -> dict:
    out = {}
    for match_id in sorted({state["match_id"] for state in states}):
        rows = [state for state in states if state["match_id"] == match_id]
        out[match_id] = separation(rows, key)
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

    # what the data says before any model is fitted
    diagnosis = {}
    for split, rows in grouped.items():
        possession = np.asarray([row["possession_by_t"] for row in rows])
        y_for = np.asarray([row["y_for"] for row in rows])
        y_against = np.asarray([row["y_against"] for row in rows])
        entry = {"n": len(rows), "share_without_possession": float(np.mean(possession == 0))}
        for name, labels in (("for", y_for), ("against", y_against)):
            for level in (1, 0):
                mask = possession == level
                entry[f"p_{name}_shoots|possession_{level}"] = {
                    "n": int(mask.sum()),
                    "rate": float(labels[mask].mean()) if mask.any() else None,
                }
        diagnosis[split] = entry
    print(json.dumps(diagnosis, indent=2), flush=True)

    # the world model, for the pass-value protocol
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

    results = {
        "rules": RULES,
        "diagnosis": diagnosis,
        "drag": drag,
        "candidates": {},
    }

    for name, factory in CANDIDATES.items():
        print(f"=== candidate {name} ===", flush=True)
        model = factory()
        model.fit(grouped["train"])
        entry = {"state_value": {}, "pass_value": {}}
        if name == "separate":
            entry["coefficients"] = {
                "for": model.for_model.coef_[0].tolist(),
                "against": model.against_model.coef_[0].tolist(),
            }
        else:
            entry["coefficients"] = model.coefficients()
        for split in ("val", "test"):
            rows = grouped[split]
            probabilities = model.predict(rows)
            entry["state_value"][split] = head_scores(rows, probabilities)
            entry["state_value"][split]["possession_coefficient"] = (
                model.for_model.coef_[0][4] if name == "separate" else None
            )

        add_values(states, model)
        by_split = {
            split: [state for state in states if split_of(state["match_id"]) == split]
            for split in ("train", "val", "test")
        }
        entry["pass_value"]["absolute"] = {
            split: separation(rows, "value_after_real") for split, rows in by_split.items()
        }
        entry["pass_value"]["absolute_per_match"] = per_match_separation(states, "value_after_real")
        entry["pass_value"]["change"] = {
            split: separation(rows, "value_change_real") for split, rows in by_split.items()
        }
        entry["pass_value"]["change_per_match"] = per_match_separation(states, "value_change_real")
        absolute_per_match = entry["pass_value"]["absolute_per_match"]
        entry["pass_value"]["matches_with_a_positive_absolute_gap"] = (
            f"{sum(1 for item in absolute_per_match.values() if item['gap'] is not None and item['gap'] > 0)}"
            f"/{len(absolute_per_match)}"
        )
        change_per_match = entry["pass_value"]["change_per_match"]
        entry["pass_value"]["matches_with_a_positive_change_gap"] = (
            f"{sum(1 for item in change_per_match.values() if item['gap'] is not None and item['gap'] > 0)}"
            f"/{len(change_per_match)}"
        )
        chain = {}
        for split, rows in by_split.items():
            if not rows:
                continue
            sim = np.asarray([row["value_after_sim"] for row in rows])
            real = np.asarray([row["value_after_real"] for row in rows])
            before = np.asarray([row["value_before"] for row in rows])
            chain[split] = {
                "mae_simulated_vs_recorded": float(np.mean(np.abs(sim - real))),
                "mae_prepass_vs_recorded": float(np.mean(np.abs(before - real))),
            }
        entry["chain"] = chain
        results["candidates"][name] = entry
        _write(OUT / f"candidate_{name}.json", entry)

    # the selection, on the validation match only
    baseline = results["candidates"]["separate"]["state_value"]
    eligible = []
    for name, entry in results["candidates"].items():
        if name == "separate":
            continue
        ok = True
        for head in ("for", "against"):
            if entry["state_value"]["val"][head]["brier"] > baseline["val"][head]["brier"] + 0.001:
                ok = False
            if entry["state_value"]["val"][head]["auroc"] is not None and baseline["val"][head]["auroc"] is not None:
                if entry["state_value"]["val"][head]["auroc"] < baseline["val"][head]["auroc"] - 0.02:
                    ok = False
        entry["passes_no_regression_on_validation"] = ok
        if ok:
            eligible.append(name)
    if eligible:
        chosen = max(
            eligible,
            key=lambda name: results["candidates"][name]["pass_value"]["absolute"]["val"]["gap"] or -1.0,
        )
    else:
        chosen = "separate"
    chosen_entry = results["candidates"][chosen]
    positive = sum(
        1 for item in chosen_entry["pass_value"]["absolute_per_match"].values() if item["gap"] is not None and item["gap"] > 0
    )
    total = len(chosen_entry["pass_value"]["absolute_per_match"])
    test_gap = chosen_entry["pass_value"]["absolute"]["test"]["gap"]
    results["chosen"] = chosen
    results["verdict"] = {
        "candidates_passing_no_regression_on_validation": eligible,
        "chosen": chosen,
        "absolute_gap_on_test": test_gap,
        "matches_with_a_positive_absolute_gap": f"{positive}/{total}",
        "separation_rule_passed": bool(test_gap is not None and test_gap > 0 and positive >= 6),
    }

    _write(OUT / "results.json", results)
    print(json.dumps(results["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
