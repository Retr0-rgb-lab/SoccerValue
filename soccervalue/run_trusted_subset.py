"""Refit geometry, spatial features, and the same raster on the checked subset.

The raster keeps the architecture in soccervalue.model. This round does not add layers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import alignment_report, choose_transform, orient_samples
from soccervalue.matches import MATCH_COMPETITION, split_of
from soccervalue.model import predict_destination, train_soccermap
from soccervalue.run_pass_success import (
    OUT,
    fit_linear,
    leave_one_match_out,
    load_match,
    score_split,
)
from soccervalue.trusted import CHECKED_EVENTS, is_trusted, kick_frame_is_close

ROOT = Path(__file__).resolve().parents[1]
LOCAL = OUT.parent / "trusted_subset"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample(rows: list[dict], per_cell: int = 2, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["success"], row.get("end_source"), kick_frame_is_close(row))
        cells.setdefault(key, []).append(row)
    chosen = []
    for key in sorted(cells, key=lambda item: (str(item[0]), str(item[1]), str(item[2]))):
        pool = cells[key]
        take = min(per_cell, len(pool))
        indexes = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[int(index)] for index in indexes)
    return [_public(row) for row in chosen]


def _public(row: dict) -> dict:
    return {
        "match_id": row["match_id"],
        "event_id": row.get("event_id"),
        "success": row["success"],
        "end_source": row.get("end_source"),
        "next_event_relation": row.get("next_event_relation"),
        "next_event_tag": row.get("next_event_tag"),
        "end_gap_s": row.get("end_gap_s"),
        "frame_signed_dt": row.get("frame_signed_dt"),
        "ball_event_m": row.get("ball_event_m"),
        "kick_frame_is_close": kick_frame_is_close(row),
        "trusted": is_trusted(row),
        "start": [round(float(value), 2) for value in row["start"]],
        "end": [round(float(value), 2) for value in row["end"]],
    }


def _counts(rows: list[dict]) -> dict:
    trusted = [row for row in rows if is_trusted(row)]
    failures = [row for row in trusted if row["success"] == 0]
    return {
        "n": len(rows),
        "n_trusted": len(trusted),
        "n_trusted_failures": len(failures),
        "trusted_success_rate": float(np.mean([row["success"] for row in trusted])) if trusted else None,
    }


def main() -> int:
    grouped_raw = {split: [] for split in ("train", "val", "test")}
    pitches = {}
    for match_id in MATCH_COMPETITION:
        passes, pitch = load_match(match_id)
        pitches[match_id] = pitch
        grouped_raw[split_of(match_id)].extend(passes)
    pitch_x, pitch_y = next(iter(pitches.values()))
    transform = choose_transform(grouped_raw["train"], pitch_x, pitch_y)
    rows = {split: orient_samples(grouped_raw[split], transform, pitch_x, pitch_y) for split in grouped_raw}
    trusted = {split: [row for row in items if is_trusted(row)] for split, items in rows.items()}
    for split, items in trusted.items():
        failures = sum(row["success"] == 0 for row in items)
        print(f"{split}: {len(items)} trusted of {len(rows[split])}, failures {failures}", flush=True)
    if len({row["success"] for row in trusted["train"]}) < 2:
        print("trusted training rows do not contain both outcomes", flush=True)
        return 2

    every = [row for split in rows for row in rows[split]]
    _write(LOCAL / "sample.json", _sample(every))
    _write(
        LOCAL / "checked_events.json",
        {
            "rule": "Keep a pass only when its own event writes the end, the tracking frame is within 1 s, and the ball is within 3 m of that kick.",
            "events_opened_in_the_xml": list(CHECKED_EVENTS),
            "counts": {split: _counts(rows[split]) for split in rows},
            "alignment": {split: alignment_report(grouped_raw[split], transform, pitch_x, pitch_y) for split in grouped_raw},
        },
    )

    geometry, spatial = fit_linear(trusted["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training the same raster on the trusted rows, {device}", flush=True)
    raster, history = train_soccermap(trusted["train"], trusted["val"])
    _write(LOCAL / "train_history.json", history)
    torch.save(raster.state_dict(), LOCAL / "position_raster.pt")
    train_rate = float(np.mean([row["success"] for row in trusted["train"]]))
    metrics = {"note": "Same raster architecture as the first round. Validation Brier on trusted rows chose the epoch."}
    for split in ("val", "test"):
        if len(trusted[split]) < 5:
            metrics[split] = {"n": len(trusted[split]), "skipped": True}
            continue
        raster_p = predict_destination(raster, trusted[split], device)
        metrics[split] = score_split(trusted[split], geometry, spatial, train_rate, raster_p)
    lomo_rows = [row for split in trusted for row in trusted[split]]
    lomo = leave_one_match_out(lomo_rows, min_test=8)
    _write(LOCAL / "metrics.json", metrics)
    _write(LOCAL / "leave_one_match_out.json", lomo)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {LOCAL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
