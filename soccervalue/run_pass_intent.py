"""Rebuild the pass-success endpoint from the aimed receiver, not from the result.

The first round scored every pass at the point the event file gives. For 76.5%
of the passes that point came from the next event, and on an unsuccessful pass
the next event belongs to the opponent. The scored cell therefore carried the
outcome: the point sat where the ball was won, not where the pass was aimed.

This round scores the same passes at the teammate the passer aimed at, read
from the kick. Two arms are fitted on the same passes, so the comparison is
about the endpoint and nothing else.

The rules below are written down before the run.

  leak    The scored endpoint must stop sitting on the result. Measured as the
          distance from the scored point to where the ball actually ended up,
          the annotated recipient. The recorded endpoint sits on that point for
          both outcomes; the intent endpoint must not.
  spatial GO only if the hand-built spatial model has a lower Brier than the
          geometry model on validation and on test, and at least 5 of the 7
          leave-one-match-out folds improve. The first round reached 6 of 7 on
          an endpoint that carried the outcome.
  raster  The raster is reported, not gated. No extra layers in either
          direction.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import alignment_report, choose_transform, orient_samples
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.metrics import binary_scores
from soccervalue.model import predict_destination, train_soccermap
from soccervalue.run_pass_success import (
    fit_linear,
    leave_one_match_out,
    load_match,
    score_split,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pass_intent"

RULES = {
    "leak": "The scored endpoint must stop sitting on the result. Measured as the distance from the scored point to where the ball actually ended up.",
    "spatial": "GO only if the spatial model has a lower Brier than geometry on validation and test, and at least 5 of 7 leave-one-match-out folds improve.",
    "raster": "Reported, not gated. No extra layers either way.",
}

ARMS = ("recorded", "intent")
# How close a scored point has to be to the annotated recipient to count as
# sitting on the result.
ON_RESULT_M = 3.0


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _filter(rows: list[dict], keep: set) -> list[dict]:
    return [row for row in rows if row.get("event_id") in keep]


def endpoint_vs_result(rows: list[dict]) -> dict:
    """How far the scored point sits from where the ball actually ended up.

    The annotated recipient is the teammate who received a successful pass and
    the opponent who intercepted a failed one. A scored point close to that
    player is a point on the result.
    """
    out = {}
    for success in (1, 0):
        distances = [
            math.hypot(row["end"][0] - row["annotated_xy"][0], row["end"][1] - row["annotated_xy"][1])
            for row in rows
            if row["success"] == success and row.get("annotated_xy") is not None
        ]
        if not distances:
            out["success" if success else "failure"] = {"n": 0}
            continue
        array = np.asarray(distances)
        out["success" if success else "failure"] = {
            "n": len(array),
            "median_m": float(np.median(array)),
            "within_3m_fraction": float(np.mean(array <= ON_RESULT_M)),
        }
    return out


def opponent_near_endpoint(rows: list[dict], limit_m: float = 2.0) -> dict:
    """How often a defender stands on the scored cell, and what it is worth."""
    near = failures_near = far = failures_far = 0
    for row in rows:
        if row["defense"]:
            nearest = min(math.hypot(x - row["end"][0], y - row["end"][1]) for x, y in row["defense"])
        else:
            nearest = 1e9
        if nearest <= limit_m:
            near += 1
            failures_near += 1 - int(row["success"])
        else:
            far += 1
            failures_far += 1 - int(row["success"])
    return {
        "limit_m": limit_m,
        "opponent_within_limit": near,
        "failure_rate_with_opponent_near": (failures_near / near) if near else None,
        "no_opponent_within_limit": far,
        "failure_rate_without_opponent_near": (failures_far / far) if far else None,
    }


def intent_quality(rows: list[dict]) -> dict:
    """Agreement between the aimed teammate and the annotated recipient.

    On a successful pass the annotated recipient is the teammate who received
    the ball, so agreement here is the accuracy of the aim. On a failed pass
    the annotated recipient is the opponent who intercepted, so agreement there
    must stay near zero. That is the check that the target is the aim and not
    the result.
    """
    out = {}
    for success in (1, 0):
        chosen = [row for row in rows if row["success"] == success and row.get("recipient")]
        if not chosen:
            out["success" if success else "failure"] = {"n": 0}
            continue
        hits = sum(1 for row in chosen if row.get("intent_player") == row["recipient"])
        perps = np.asarray([row["intent_perp_m"] for row in chosen if row.get("intent_perp_m") is not None])
        alongs = np.asarray([row["intent_along_m"] for row in chosen if row.get("intent_along_m") is not None])
        out["success" if success else "failure"] = {
            "n": len(chosen),
            "agreement_with_annotated_recipient": hits / len(chosen),
            "target_perp_m": {
                "median": float(np.median(perps)) if len(perps) else None,
                "p25": float(np.percentile(perps, 25)) if len(perps) else None,
                "p75": float(np.percentile(perps, 75)) if len(perps) else None,
            },
            "target_along_m": {
                "median": float(np.median(alongs)) if len(alongs) else None,
                "p25": float(np.percentile(alongs, 25)) if len(alongs) else None,
                "p75": float(np.percentile(alongs, 75)) if len(alongs) else None,
            },
        }
    return out


def intent_status_counts(passes: list[dict]) -> dict:
    counts = Counter((pass_.get("intent") or {}).get("status", "missing") for pass_ in passes)
    return dict(sorted(counts.items()))


def per_match_agreement(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        bucket = out.setdefault(row["match_id"], {"n": 0, "hits": 0})
        if row["success"] == 1 and row.get("recipient"):
            bucket["n"] += 1
            bucket["hits"] += int(row.get("intent_player") == row["recipient"])
    return {
        match: {"n": values["n"], "agreement": (values["hits"] / values["n"]) if values["n"] else None}
        for match, values in sorted(out.items())
    }


def by_recorded_source(rows: list[dict], y_true: np.ndarray, probabilities: dict) -> dict:
    """Score the split by where the old endpoint came from.

    The first round found the raster at chance on the passes whose own event
    wrote the end, and strong on the passes whose end was borrowed from the
    next event. The same split is reported here for both arms.
    """
    out = {}
    for source in sorted({str(row.get("recorded_end_source")) for row in rows}):
        chosen = [index for index, row in enumerate(rows) if str(row.get("recorded_end_source")) == source]
        if not chosen:
            continue
        subset_y = y_true[np.asarray(chosen)]
        entry = {"n": len(chosen)}
        for name, prob in probabilities.items():
            entry[name] = binary_scores(subset_y, prob[np.asarray(chosen)])
        out[source] = entry
    return out


def fit_arm(rows: dict, label: str) -> dict:
    """Fit the three models on one arm and score validation and test."""
    geometry, spatial = fit_linear(rows["train"])
    train_rate = float(np.mean([row["success"] for row in rows["train"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{label}] training the position raster on {device}", flush=True)
    raster, history = train_soccermap(rows["train"], rows["val"])
    torch.save(raster.state_dict(), OUT / f"position_raster_{label}.pt")
    _write(OUT / f"train_history_{label}.json", history)
    metrics = {"note": "Validation Brier chose the raster epoch."}
    for split in ("val", "test"):
        raster_p = predict_destination(raster, rows[split], device)
        metrics[split] = score_split(rows[split], geometry, spatial, train_rate, raster_p)
        y_true = np.asarray([row["success"] for row in rows[split]])
        probabilities = {
            "geometry": geometry.predict(rows[split]),
            "geometry_spatial": spatial.predict(rows[split]),
            "position_raster": raster_p,
        }
        metrics[f"{split}_by_recorded_source"] = by_recorded_source(rows[split], y_true, probabilities)
        _write(
            OUT / f"predictions_{label}_{split}.json",
            [
                {
                    "match_id": row["match_id"],
                    "event_id": row.get("event_id"),
                    "success": row["success"],
                    "recorded_end_source": row.get("recorded_end_source"),
                    "intent_player": row.get("intent_player"),
                    "recipient": row.get("recipient"),
                    "intent_perp_m": row.get("intent_perp_m"),
                    "intent_along_m": row.get("intent_along_m"),
                    "geometry": float(probabilities["geometry"][index]),
                    "geometry_spatial": float(probabilities["geometry_spatial"][index]),
                    "position_raster": float(probabilities["position_raster"][index]),
                }
                for index, row in enumerate(rows[split])
            ],
        )
    lomo = leave_one_match_out([row for split in rows for row in rows[split]])
    _write(OUT / f"leave_one_match_out_{label}.json", lomo)
    return {
        "metrics": metrics,
        "lomo": lomo,
        "spatial_wins": sum(row["geometry_spatial"]["brier"] < row["geometry"]["brier"] for row in lomo),
        "n_folds": len(lomo),
    }


def main() -> int:
    assert_split_disjoint()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

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
    _write(
        OUT / "alignment.json",
        {
            "transform": transform,
            "by_split": {
                split: alignment_report(items, transform, pitch_x, pitch_y) for split, items in grouped.items()
            },
        },
    )

    arms = {
        mode: {
            split: orient_samples(grouped[split], transform, pitch_x, pitch_y, end_mode=mode)
            for split in grouped
        }
        for mode in ARMS
    }
    intent_ids = {row["event_id"] for rows in arms["intent"].values() for row in rows}
    arms["recorded"] = {split: _filter(rows, intent_ids) for split, rows in arms["recorded"].items()}

    for mode in ARMS:
        for split in ("train", "val", "test"):
            rate = np.mean([row["success"] for row in arms[mode][split]]) if arms[mode][split] else None
            print(f"[{mode}] {split}: {len(arms[mode][split])} passes, success rate {rate}", flush=True)

    _write(
        OUT / "intent_coverage.json",
        {
            "passes_parsed": sum(len(items) for items in grouped.values()),
            "passes_with_intent": len(intent_ids),
            "status_counts": intent_status_counts([p for items in grouped.values() for p in items]),
            "per_split": {
                split: {
                    "parsed": len(grouped[split]),
                    "with_intent": len(arms["intent"][split]),
                    "kept_rate": len(arms["intent"][split]) / max(len(grouped[split]), 1),
                }
                for split in grouped
            },
        },
    )

    results = {}
    for mode in ARMS:
        print(f"=== arm {mode} ===", flush=True)
        results[mode] = fit_arm(arms[mode], mode)
        everything = [row for split in arms[mode] for row in arms[mode][split]]
        results[mode]["endpoint_vs_result"] = {
            split: endpoint_vs_result(arms[mode][split]) for split in ("val", "test")
        }
        results[mode]["endpoint_vs_result"]["all"] = endpoint_vs_result(everything)
        results[mode]["opponent_near_endpoint"] = {
            split: opponent_near_endpoint(arms[mode][split]) for split in ("val", "test")
        }
        results[mode]["intent_quality"] = intent_quality(everything)
        results[mode]["per_match_agreement"] = per_match_agreement(
            [row for split in arms["intent"] for row in arms["intent"][split]]
        )
        _write(OUT / f"metrics_{mode}.json", results[mode])

    _write(OUT / "results.json", results)
    print(json.dumps(results, indent=2), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
