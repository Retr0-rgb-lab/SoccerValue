"""Train one player rollout and compare it with constant velocity.

The ball path is the frozen drag model from the previous round. One predicted
trajectory is scored. Twenty samples are not drawn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from soccervalue.events import load_timeline, pitch_size
from soccervalue.matches import MATCH_COMPETITION, split_of
from soccervalue.player_motion import score_players, train_player_motion
from soccervalue.run_pass_success import ensure_file
from soccervalue.tracking import _read_trajectories
from soccervalue.world_segments import build_segments, in_play_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "player_motion"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training player motion on {device}", flush=True)
    model, history = train_player_motion(grouped["train"], grouped["val"])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "player_motion.pt")
    metrics = {"train_history": history, "drag_ball": {}, "real_ball": {}}
    for name, use_drag in (("drag_ball", True), ("real_ball", False)):
        for split in ("val", "test"):
            metrics[name][split] = {
                "constant_velocity": score_players(None, grouped[split], use_drag, device),
                "player_motion": score_players(model, grouped[split], use_drag, device),
            }
            print(name, split, json.dumps(metrics[name][split]), flush=True)
    val = metrics["drag_ball"]["val"]
    go = (
        val["player_motion"]["ade_1s"] is not None
        and val["constant_velocity"]["ade_1s"] is not None
        and val["player_motion"]["ade_1s"] < val["constant_velocity"]["ade_1s"]
        and val["player_motion"]["end_error"] < val["constant_velocity"]["end_error"]
    )
    metrics["go"] = go
    metrics["rule"] = "Pass only if one drag-ball rollout has lower validation error than constant velocity at 1s and at the segment end."
    _write(OUT / "metrics.json", metrics)
    print(json.dumps({"go": go}, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
