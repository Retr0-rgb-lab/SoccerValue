"""Train the ground-pass success baseline and the SoccerMap success map.

Data are the seven public IDSSE matches (DFL, CC-BY 4.0). Cite Bassek et al.,
Scientific Data 2025, https://doi.org/10.1038/s41597-025-04505-y
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import (
    ALIGN_MEDIAN_LIMIT_M,
    alignment_report,
    choose_transform,
    orient_samples,
)
from soccervalue.baseline import BallLocationBaseline
from soccervalue.events import load_open_flat_passes, pitch_size
from soccervalue.matches import MATCH_COMPETITION, TEST_MATCHES, TRAIN_MATCHES, VAL_MATCHES, assert_split_disjoint, split_of
from soccervalue.metrics import binary_scores, calibration_table
from soccervalue.model import predict_destination, train_soccermap
from soccervalue.tracking import attach_tracking

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "pass_success"
HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"


def filename(match_id: str, kind: str) -> str:
    competition = MATCH_COMPETITION[match_id]
    if kind == "events":
        return f"DFL_03_02_events_raw_{competition}_DFL-MAT-{match_id}.xml"
    if kind == "meta":
        return f"DFL_02_01_matchinformation_{competition}_DFL-MAT-{match_id}.xml"
    if kind == "tracking":
        return f"DFL_04_03_positions_raw_observed_{competition}_DFL-MAT-{match_id}.xml"
    raise KeyError(kind)


def file_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 400))
        tail = handle.read().decode("utf-8", errors="ignore")
    return "</PutDataRequest>" in tail


def ensure_file(match_id: str, kind: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{match_id}_{kind}.xml"
    url = f"{HF}/{filename(match_id, kind)}"
    for attempt in range(1, 31):
        if file_is_complete(path):
            return path
        print(f"downloading {path.name} attempt {attempt}", flush=True)
        subprocess.run(
            ["curl.exe", "-L", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(path), url],
            check=False,
        )
        if file_is_complete(path):
            return path
        time.sleep(min(attempt, 10))
    raise RuntimeError(f"incomplete download: {path}")


def load_match(match_id: str) -> tuple[list[dict], tuple[float, float]]:
    events_path = ensure_file(match_id, "events")
    meta_path = ensure_file(match_id, "meta")
    tracking_path = ensure_file(match_id, "tracking")
    pitch = pitch_size(meta_path)
    passes = load_open_flat_passes(events_path, match_id)
    print(f"{match_id}: {len(passes)} open flat passes, attaching tracking", flush=True)
    attach_tracking(tracking_path, passes, pitch)
    return passes, pitch


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = {split: [] for split in ("train", "val", "test")}
    pitches = {}
    for match_id in MATCH_COMPETITION:
        passes, pitch = load_match(match_id)
        pitches[match_id] = pitch
        grouped[split_of(match_id)].extend(passes)

    pitch_x, pitch_y = next(iter(pitches.values()))
    if any(size != (pitch_x, pitch_y) for size in pitches.values()):
        raise RuntimeError(f"pitch sizes differ: {pitches}")

    transform = choose_transform(grouped["train"], pitch_x, pitch_y)
    report = {
        split: alignment_report(rows, transform, pitch_x, pitch_y) for split, rows in grouped.items()
    }
    report["transform_chosen_on_train"] = transform
    report["median_limit_m"] = ALIGN_MEDIAN_LIMIT_M
    (OUT / "alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["train"]["usable"]:
        print("alignment on the training matches is outside the limit; model fitting stopped", flush=True)
        return 2

    rows = {
        split: orient_samples(grouped[split], transform, pitch_x, pitch_y) for split in grouped
    }
    for split, items in rows.items():
        print(f"{split}: {len(items)} aligned passes, success rate {np.mean([r['success'] for r in items]):.3f}", flush=True)

    baseline = BallLocationBaseline()
    baseline.fit(rows["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training SoccerMap success map on {device}", flush=True)
    model, history = train_soccermap(rows["train"], rows["val"])
    (OUT / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics = {}
    for split in ("val", "test"):
        y_true = np.asarray([row["success"] for row in rows[split]])
        baseline_p = baseline.predict(rows[split])
        map_p = predict_destination(model, rows[split], device)
        train_rate = float(np.mean([row["success"] for row in rows["train"]]))
        metrics[split] = {
            "constant_train_rate": binary_scores(y_true, np.full(y_true.shape, train_rate)),
            "ball_location": binary_scores(y_true, baseline_p),
            "soccermap": binary_scores(y_true, map_p),
        }
        if split == "test":
            (OUT / "calibration_test_soccermap.json").write_text(
                json.dumps(calibration_table(y_true, map_p), indent=2), encoding="utf-8"
            )
            (OUT / "calibration_test_baseline.json").write_text(
                json.dumps(calibration_table(y_true, baseline_p), indent=2), encoding="utf-8"
            )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    print(f"held-out matches: val={VAL_MATCHES[0]} test={TEST_MATCHES[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
