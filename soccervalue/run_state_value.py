"""Train state-value heads on real matches. This does not fit a world model.

The 10-second window is the main result. Five and twenty seconds are refit
separately and are not added into one loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from soccervalue.align import choose_transform
from soccervalue.events import load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, TEST_MATCHES, TRAIN_MATCHES, VAL_MATCHES, split_of
from soccervalue.metrics import binary_scores, calibration_table
from soccervalue.run_pass_success import ensure_file
from soccervalue.state_frames import attach_state_frame, orient_state
from soccervalue.state_value import build_state_samples
from soccervalue.tracking import _read_trajectories
from soccervalue.value_model import (
    SimpleStateValue,
    constant_rates,
    predict_state_value,
    train_state_value,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "state_value"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _head_scores(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    scores = binary_scores(y_true, probabilities)
    if y_true.size and len(np.unique(y_true)) == 2:
        scores["log_loss"] = float(log_loss(y_true, probabilities, labels=[0, 1]))
    else:
        scores["log_loss"] = None
    return scores


def _prepare(window_s: float, timelines: dict, trajectories: dict, pitches: dict, transform: str, pitch: tuple[float, float]):
    grouped = {split: [] for split in ("train", "val", "test")}
    for match_id, events in timelines.items():
        samples = build_state_samples(events, window_s=window_s)
        attach_state_frame(
            ensure_file(match_id, "tracking"),
            samples,
            pitches[match_id],
            trajectories=trajectories[match_id],
        )
        for sample in samples:
            row = orient_state(sample, transform, pitch[0], pitch[1])
            if row is None:
                continue
            row["time"] = sample["time"].isoformat()
            grouped[split_of(match_id)].append(row)
    return grouped


def _fit_and_score(grouped: dict, write_calibration: bool = False) -> tuple[dict, dict, object]:
    simple = SimpleStateValue()
    simple.fit(grouped["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, history = train_state_value(grouped["train"], grouped["val"])
    rates = constant_rates(grouped["train"])
    metrics = {"train_history": history}
    predictions = {}
    for split in ("val", "test"):
        rows = grouped[split]
        constant = np.column_stack([np.full(len(rows), rates[0]), np.full(len(rows), rates[1])])
        simple_p = simple.predict(rows)
        net_p = predict_state_value(net, rows, device)
        predictions[split] = {"constant": constant, "simple": simple_p, "spatial": net_p}
        metrics[split] = {}
        for head, column in (("for", 0), ("against", 1)):
            labels = np.asarray([row["y_for"] if head == "for" else row["y_against"] for row in rows])
            metrics[split][head] = {
                "constant": _head_scores(labels, constant[:, column]),
                "simple": _head_scores(labels, simple_p[:, column]),
                "spatial": _head_scores(labels, net_p[:, column]),
            }
            if write_calibration:
                _write(
                    OUT / f"calibration_{split}_{head}.json",
                    {
                        "simple": calibration_table(labels, simple_p[:, column]),
                        "spatial": calibration_table(labels, net_p[:, column]),
                    },
                )
    return metrics, predictions, net


def _failures(rows: list[dict], predictions: dict, limit: int = 8) -> dict:
    names = ("constant", "simple", "spatial")

    def pack(index: int) -> dict:
        row = rows[index]
        return {
            "match_id": row["match_id"],
            "event_id": row["event_id"],
            "y_for": row["y_for"],
            "y_against": row["y_against"],
            "probabilities": {
                name: {"for": float(predictions[name][index, 0]), "against": float(predictions[name][index, 1])}
                for name in names
            },
            "shots_in_window": row.get("shots_in_window") or [],
        }

    report = {}
    for head, column, label_key in (("for", 0, "y_for"), ("against", 1, "y_against")):
        positives = [index for index, row in enumerate(rows) if row[label_key] == 1]
        negatives = [index for index, row in enumerate(rows) if row[label_key] == 0 and predictions["spatial"][index, column] > 0.5]
        positives.sort(key=lambda index: predictions["spatial"][index, column])
        negatives.sort(key=lambda index: predictions["spatial"][index, column], reverse=True)
        report[head] = {
            "low_probability_positives": [pack(index) for index in positives[:limit]],
            "confident_negatives": [pack(index) for index in negatives[:limit]],
        }
    return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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

    counts = {"shots": {}, "goals": {}, "positives": {}, "note": None}
    for match_id, events in timelines.items():
        counts["shots"][match_id] = sum(item["is_shot"] for item in events)
        counts["goals"][match_id] = sum(item["is_goal"] for item in events)
    grouped = _prepare(10.0, timelines, trajectories, pitches, transform, pitch)
    for split, rows in grouped.items():
        counts["positives"][split] = {
            "n": len(rows),
            "y_for": int(sum(row["y_for"] for row in rows)),
            "y_against": int(sum(row["y_against"] for row in rows)),
        }
    if any(item["y_for"] == 0 or item["y_against"] == 0 for item in counts["positives"].values()):
        counts["note"] = "这是管线原型"
    _write(OUT / "label_counts.json", counts)
    for split, rows in grouped.items():
        print(
            f"10s {split}: {len(rows)} states, for {counts['positives'][split]['y_for']}, against {counts['positives'][split]['y_against']}",
            flush=True,
        )

    metrics, predictions, net = _fit_and_score(grouped, write_calibration=True)
    _write(OUT / "metrics_w10.json", metrics)
    torch.save(net.state_dict(), OUT / "state_value_net.pt")
    _write(OUT / "failures_test.json", _failures(grouped["test"], predictions["test"]))

    for window in (5.0, 20.0):
        print(f"refitting {window:g}s", flush=True)
        window_rows = _prepare(window, timelines, trajectories, pitches, transform, pitch)
        window_metrics, _predictions, _net = _fit_and_score(window_rows)
        _write(OUT / f"metrics_w{int(window)}.json", window_metrics)
    print(f"wrote {OUT}", flush=True)
    print(f"matches train={TRAIN_MATCHES} val={VAL_MATCHES} test={TEST_MATCHES}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
