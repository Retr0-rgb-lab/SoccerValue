"""Ball flight and next-touch on tracking segments. Value is not trained."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from soccervalue.ball_flight import fit_drag, summarise
from soccervalue.events import load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, split_of
from soccervalue.run_pass_success import ensure_file
from soccervalue.touch_head import fit_touch, score_touch
from soccervalue.tracking import _read_trajectories
from soccervalue.world_segments import build_segments, in_play_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "world_ball"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _better(summary: dict) -> bool:
    one = summary["drag_ade_1s"] is not None and summary["constant_velocity_ade_1s"] is not None
    end = summary["drag_at_end"] is not None and summary["constant_velocity_at_end"] is not None
    return bool(one and end and summary["drag_ade_1s"] < summary["constant_velocity_ade_1s"] and summary["drag_at_end"] < summary["constant_velocity_at_end"])


def main() -> int:
    grouped = {split: [] for split in ("train", "val", "test")}
    for match_id in MATCH_COMPETITION:
        print(f"segments {match_id}", flush=True)
        events = load_timeline(ensure_file(match_id, "events"), match_id)
        window = in_play_window(events)
        if window is None:
            continue
        pitch = pitch_size(ensure_file(match_id, "meta"))
        ball, players = _read_trajectories(ensure_file(match_id, "tracking"))
        grouped[split_of(match_id)].extend(build_segments(ball, players, pitch, window[0], window[1]))
        print(f"  {match_id}: running total {sum(len(rows) for rows in grouped.values())}", flush=True)
    drag = fit_drag(grouped["train"])
    ball_metrics = {split: summarise(rows, drag) for split, rows in grouped.items()}
    ball_go = _better(ball_metrics["val"])
    labelled = [segment for segment in grouped["train"] if segment["toucher"] is not None]
    model, mean, std = fit_touch(labelled)
    chosen_ball = str(drag) if ball_go else "constant"
    touch_metrics = {
        split: {
            "real_ball": score_touch(rows, model, mean, std, "real"),
            "model_ball": score_touch(rows, model, mean, std, chosen_ball if chosen_ball != "constant" else "constant"),
        }
        for split, rows in grouped.items()
        if split != "train"
    }
    real_val = touch_metrics["val"]["real_ball"]
    touch_go = (
        real_val["model_top1"] is not None
        and real_val["baseline_top1"] is not None
        and real_val["model_top1"] > real_val["baseline_top1"]
    )
    payload = {
        "drag": drag,
        "ball": ball_metrics,
        "ball_go": ball_go,
        "touch": touch_metrics,
        "touch_go_on_real_ball": touch_go,
        "rule": "Ball passes only if validation error at 1s and at the segment end both beat constant velocity. Touch passes only if top-1 on the real ball path beats earliest intercept.",
    }
    _write(OUT / "metrics.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
