"""Finish the ground-pass success comparison.

Model A is a logistic regression on the ball and the recorded pass segment.
Model B is the same regression plus hand-built defender and teammate distances.
Model C is the fully convolutional success map on the player raster.
The label is the recorded Sportec outcome at that point, not an intended target.

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
from soccervalue.baseline import LogisticSuccess
from soccervalue.events import LABEL_DEFINITION, label_audit, load_open_flat_passes, pitch_size
from soccervalue.features import GEOMETRY_NAMES, SPATIAL_NAMES, describe_space, geometry_matrix, spatial_matrix
from soccervalue.matches import MATCH_COMPETITION, TEST_MATCHES, VAL_MATCHES, assert_split_disjoint, split_of
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


def fit_linear(rows: list[dict]) -> tuple[LogisticSuccess, LogisticSuccess]:
    geometry = LogisticSuccess(geometry_matrix, GEOMETRY_NAMES)
    spatial = LogisticSuccess(spatial_matrix, SPATIAL_NAMES)
    geometry.fit(rows)
    spatial.fit(rows)
    return geometry, spatial


def score_split(rows: list[dict], geometry: LogisticSuccess, spatial: LogisticSuccess, train_rate: float, raster_p: np.ndarray | None) -> dict:
    y_true = np.asarray([row["success"] for row in rows])
    scored = {
        "constant_train_rate": binary_scores(y_true, np.full(y_true.shape, train_rate)),
        "geometry": binary_scores(y_true, geometry.predict(rows)),
        "geometry_spatial": binary_scores(y_true, spatial.predict(rows)),
    }
    if raster_p is not None:
        scored["position_raster"] = binary_scores(y_true, raster_p)
    return scored


def leave_one_match_out(rows: list[dict], min_test: int = 20) -> list[dict]:
    """Refit A and B on every match except one. The raster is not refit here."""
    matches = sorted({row["match_id"] for row in rows})
    report = []
    for hold in matches:
        train = [row for row in rows if row["match_id"] != hold]
        test = [row for row in rows if row["match_id"] == hold]
        if len(test) < min_test or len({row["success"] for row in train}) < 2:
            continue
        train_rate = float(np.mean([row["success"] for row in train]))
        geometry, spatial = fit_linear(train)
        scored = score_split(test, geometry, spatial, train_rate, None)
        report.append({"match_id": hold, "n": len(test), **scored})
    return report


def failure_report(rows: list[dict], probabilities: dict[str, np.ndarray], limit: int = 12) -> dict:
    missed = _ranked(rows, probabilities, want_success=0, highest=True, limit=limit)
    surprised = _ranked(rows, probabilities, want_success=1, highest=False, limit=limit)
    y_true = np.asarray([row["success"] for row in rows])
    summary = {}
    for name, prob in probabilities.items():
        failures = prob[y_true == 0]
        summary[name] = {
            "n_failures": int(failures.size),
            "mean_predicted_on_failures": float(failures.mean()) if failures.size else None,
            "failures_predicted_above_0_9": int(np.sum(failures > 0.9)) if failures.size else 0,
        }
    return {
        "note": "High probability on a recorded failure means the model still called that pass safe. These examples are not intended-target errors.",
        "summary": summary,
        "confident_failures": missed,
        "low_probability_successes": surprised,
    }


def _ranked(rows, probabilities, want_success: int, highest: bool, limit: int) -> list[dict]:
    order = sorted(
        range(len(rows)),
        key=lambda index: probabilities["position_raster"][index],
        reverse=highest,
    )
    chosen = []
    for index in order:
        row = rows[index]
        if row["success"] != want_success:
            continue
        space = describe_space(row)
        chosen.append(
            {
                "match_id": row["match_id"],
                "event_id": row["event_id"],
                "success": row["success"],
                "end_source": row.get("end_source"),
                "next_event_relation": row.get("next_event_relation"),
                "next_event_tag": row.get("next_event_tag"),
                "frame_signed_dt": row.get("frame_signed_dt"),
                "geometry": float(probabilities["geometry"][index]),
                "geometry_spatial": float(probabilities["geometry_spatial"][index]),
                "position_raster": float(probabilities["position_raster"][index]),
                **space,
            }
        )
        if len(chosen) >= limit:
            break
    return chosen


def prediction_rows(rows: list[dict], probabilities: dict[str, np.ndarray]) -> list[dict]:
    exported = []
    for index, row in enumerate(rows):
        exported.append(
            {
                "match_id": row["match_id"],
                "event_id": row["event_id"],
                "success": row["success"],
                "end_source": row.get("end_source"),
                "next_event_relation": row.get("next_event_relation"),
                "geometry": float(probabilities["geometry"][index]),
                "geometry_spatial": float(probabilities["geometry_spatial"][index]),
                "position_raster": float(probabilities["position_raster"][index]),
            }
        )
    return exported


def reading(test: dict, heldout_raster: dict, lomo: list[dict]) -> dict:
    spatial_wins = sum(row["geometry_spatial"]["brier"] < row["geometry"]["brier"] for row in lomo)
    raster_beats_spatial = {
        split: heldout_raster[split]["position_raster"]["brier"] < heldout_raster[split]["geometry_spatial"]["brier"]
        for split in heldout_raster
    }
    raster_beats_geometry = {
        split: heldout_raster[split]["position_raster"]["brier"] < heldout_raster[split]["geometry"]["brier"]
        for split in heldout_raster
    }
    return {
        "question": "After the recorded destination is fixed, does player positioning improve the official completion label beyond the pass geometry?",
        "label": LABEL_DEFINITION,
        "test_brier": {
            "geometry": test["geometry"]["brier"],
            "geometry_spatial": test["geometry_spatial"]["brier"],
            "position_raster": test["position_raster"]["brier"],
        },
        "handcrafted_space_lower_brier_than_geometry_on_n_matches": f"{spatial_wins}/{len(lomo)}",
        "raster_lower_brier_than_geometry": raster_beats_geometry,
        "raster_lower_brier_than_handcrafted_space": raster_beats_spatial,
        "what_this_does_not_say": "A lower Brier is a gain on the recorded outcome. It is not a license to recommend passes that were never attempted. If the raster loses to the logistic models, that does not show that player positions are useless.",
    }


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = {split: [] for split in ("train", "val", "test")}
    pitches = {}
    raw_passes = []
    for match_id in MATCH_COMPETITION:
        passes, pitch = load_match(match_id)
        pitches[match_id] = pitch
        raw_passes.extend(passes)
        grouped[split_of(match_id)].extend(passes)

    pitch_x, pitch_y = next(iter(pitches.values()))
    if any(size != (pitch_x, pitch_y) for size in pitches.values()):
        raise RuntimeError(f"pitch sizes differ: {pitches}")

    _write(OUT / "label_audit.json", label_audit(raw_passes))
    transform = choose_transform(grouped["train"], pitch_x, pitch_y)
    report = {split: alignment_report(rows, transform, pitch_x, pitch_y) for split, rows in grouped.items()}
    report["transform_chosen_on_train"] = transform
    report["median_limit_m"] = ALIGN_MEDIAN_LIMIT_M
    report["frame_time_meaning"] = (
        "frame_signed_dt is the chosen tracking frame minus the event timestamp, in seconds. "
        "Positive means the frame is after the annotated event. The frame is chosen because the "
        "ball and passer match the kick location, not because it is guaranteed to precede the kick."
    )
    _write(OUT / "alignment.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["train"]["usable"]:
        print("alignment on the training matches is outside the limit; model fitting stopped", flush=True)
        return 2

    rows = {split: orient_samples(grouped[split], transform, pitch_x, pitch_y) for split in grouped}
    for split, items in rows.items():
        print(f"{split}: {len(items)} aligned passes, success rate {np.mean([r['success'] for r in items]):.3f}", flush=True)

    geometry, spatial = fit_linear(rows["train"])
    _write(
        OUT / "linear_coefficients.json",
        {"geometry": geometry.coefficients(), "geometry_spatial": spatial.coefficients()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training position raster on {device}", flush=True)
    raster, history = train_soccermap(rows["train"], rows["val"])
    _write(OUT / "train_history.json", history)
    torch.save(raster.state_dict(), OUT / "position_raster.pt")

    train_rate = float(np.mean([row["success"] for row in rows["train"]]))
    metrics = {"note": "Validation Brier chose the raster epoch. The test match did not."}
    probabilities = {}
    for split in ("val", "test"):
        raster_p = predict_destination(raster, rows[split], device)
        metrics[split] = score_split(rows[split], geometry, spatial, train_rate, raster_p)
        probabilities[split] = {
            "geometry": geometry.predict(rows[split]),
            "geometry_spatial": spatial.predict(rows[split]),
            "position_raster": raster_p,
        }
        for source in ("event", "next_event"):
            chosen = [index for index, row in enumerate(rows[split]) if row.get("end_source") == source]
            if not chosen:
                continue
            subset = [rows[split][index] for index in chosen]
            subset_p = raster_p[np.asarray(chosen)]
            metrics[f"{split}_end_from_{source}"] = score_split(subset, geometry, spatial, train_rate, subset_p)
        if split == "test":
            y_true = np.asarray([row["success"] for row in rows[split]])
            _write(OUT / "calibration_test_position_raster.json", calibration_table(y_true, raster_p))
            _write(OUT / "calibration_test_geometry_spatial.json", calibration_table(y_true, probabilities[split]["geometry_spatial"]))

    every = [row for split in ("train", "val", "test") for row in rows[split]]
    lomo = leave_one_match_out(every)
    _write(OUT / "leave_one_match_out.json", lomo)
    _write(OUT / "metrics.json", metrics)
    _write(OUT / "predictions_val.json", prediction_rows(rows["val"], probabilities["val"]))
    _write(OUT / "predictions_test.json", prediction_rows(rows["test"], probabilities["test"]))
    _write(OUT / "failures_test.json", failure_report(rows["test"], probabilities["test"]))
    heldout = {split: metrics[split] for split in ("val", "test")}
    _write(OUT / "reading.json", reading(metrics["test"], heldout, lomo))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    print(f"held-out matches: val={VAL_MATCHES[0]} test={TEST_MATCHES[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
