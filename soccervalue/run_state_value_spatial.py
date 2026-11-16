"""Does a four-feature spatial logistic beat ball location plus possession?

The comparison is only the 10-second own-shot head. The raster is not retrained.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from soccervalue.align import choose_transform
from soccervalue.events import load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, split_of
from soccervalue.metrics import binary_scores
from soccervalue.run_pass_success import ensure_file
from soccervalue.run_state_value import OUT, _prepare
from soccervalue.state_frames import attach_state_frame
from soccervalue.state_value import build_state_samples
from soccervalue.tracking import _read_trajectories
from soccervalue.value_model import SimpleStateValue, spatial_features

LOCAL_OUT = OUT.parent / "state_value_spatial"
BRIER_SLACK = 0.002


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _own_shot(model: SimpleStateValue, rows: list[dict]) -> dict:
    labels = np.asarray([row["y_for"] for row in rows])
    probabilities = model.predict(rows)[:, 0]
    return binary_scores(labels, probabilities)


def _leave_one_match(rows: list[dict]) -> list[dict]:
    report = []
    for hold in sorted({row["match_id"] for row in rows}):
        train = [row for row in rows if row["match_id"] != hold]
        test = [row for row in rows if row["match_id"] == hold]
        if len({row["y_for"] for row in train}) < 2 or len({row["y_for"] for row in test}) < 2:
            report.append({"match_id": hold, "n": len(test), "comparable": False})
            continue
        simple = SimpleStateValue()
        spatial = SimpleStateValue(spatial_features)
        simple.fit(train)
        spatial.fit(train)
        simple_scores = _own_shot(simple, test)
        spatial_scores = _own_shot(spatial, test)
        report.append(
            {
                "match_id": hold,
                "n": len(test),
                "comparable": True,
                "simple": simple_scores,
                "spatial": spatial_scores,
                "spatial_auroc_higher": spatial_scores["auroc"] > simple_scores["auroc"],
            }
        )
    return report


def _decision(val: dict, folds: list[dict]) -> dict:
    comparable = [row for row in folds if row["comparable"]]
    wins = sum(row["spatial_auroc_higher"] for row in comparable)
    val_auroc = val["spatial"]["auroc"] > val["simple"]["auroc"]
    val_brier = val["spatial"]["brier"] <= val["simple"]["brier"] + BRIER_SLACK
    go = bool(val_auroc and val_brier and wins >= 5)
    return {
        "go": go,
        "val_auroc_higher": val_auroc,
        "val_brier_within_slack": val_brier,
        "leave_one_match_auroc_wins": f"{wins}/{len(comparable)}",
        "rule": "GO only if validation own-shot AUROC rises, validation Brier is not worse by more than 0.002, and at least 5 leave-one-match folds also rise.",
    }


def main() -> int:
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
        if split_of(match_id) != "train":
            continue
        samples = build_state_samples(events, window_s=10.0)
        attach_state_frame(
            ensure_file(match_id, "tracking"),
            samples,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        for sample in samples:
            if sample.get("track_ball") is None:
                continue
            raw_for_shift.append(
                {"start_x": sample["x"], "start_y": sample["y"], "track_ball": (*sample["track_ball"], None)}
            )
    transform = choose_transform(raw_for_shift, pitch[0], pitch[1])
    grouped = _prepare(10.0, timelines, trajectories, pitches, transform, pitch)
    simple = SimpleStateValue()
    spatial = SimpleStateValue(spatial_features)
    simple.fit(grouped["train"])
    spatial.fit(grouped["train"])
    heldout = {}
    for split in ("val", "test"):
        heldout[split] = {"simple": _own_shot(simple, grouped[split]), "spatial": _own_shot(spatial, grouped[split])}
        print(split, json.dumps(heldout[split]), flush=True)
    every = [row for split in grouped for row in grouped[split]]
    folds = _leave_one_match(every)
    payload = {
        "question": "Do four meter features improve 10-second own-shot ranking beyond ball location and possession?",
        "features_added": [
            "nearest opponent to the ball",
            "ball distance to the opponent goal line",
            "count of T players ahead of the ball",
            "opponents within 5 m of the ball",
        ],
        "val": heldout["val"],
        "test": heldout["test"],
        "leave_one_match": folds,
        "decision": _decision(heldout["val"], folds),
    }
    _write(LOCAL_OUT / "metrics.json", payload)
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"wrote {LOCAL_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
