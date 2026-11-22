"""Rebuild the pass-success endpoint from the trained aimed receiver.

The geometric rule in `soccervalue.intent` gives the endpoint for the pass
success models. A trained aimed receiver is more accurate on successful passes,
but the diagnostic in `results/pass-intent-model/conclusion.json` shows where
that accuracy comes from: it prefers the more open teammate. On the failed
passes, where the endpoint has to be right and no label exists, that preference
is untested.

This round measures the effect downstream. The same passes are scored at three
endpoints: the rule, the linear model and the MLP. If the model's endpoint is
wrong on the failed passes, the label becomes "the ball did not reach the
teammate the model picked", and that teammate is the open one, so the spatial
features would start to point the wrong way. The sign of the spatial
coefficients is the test.

The rules below are written down before the run.

  leak    The spatial features must keep their sign. A defender near the
          endpoint must still mean failure. A sign flip on the model endpoint
          means the endpoint is wrong on the failed passes.
  spatial GO only if the spatial model still has a lower Brier than the
          geometry model on validation and on test, and at least 5 of the 7
          leave-one-match-out folds improve.
  endpoint The model endpoint replaces the rule only if the spatial gain
          survives and the signs hold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import choose_transform, orient_samples
from soccervalue.intent_model import build_dataset, fit_intent_model, predict_pick
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.model import predict_destination, train_soccermap
from soccervalue.run_pass_success import (
    fit_linear,
    leave_one_match_out,
    load_match,
    score_split,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pass_intent_endpoint"

RULES = {
    "leak": "The spatial features must keep their sign. A defender near the endpoint must still mean failure.",
    "spatial": "GO only if the spatial model still has a lower Brier than the geometry model on validation and test, and at least 5 of 7 leave-one-match-out folds improve.",
    "endpoint": "The model endpoint replaces the rule only if the spatial gain survives and the signs hold.",
}

ARMS = ("rule", "linear", "mlp")
# The spatial features whose sign says whether the endpoint is honest.
SPATIAL_SIGNS = (
    "nearest_defender_to_ball",
    "nearest_defender_to_end",
    "nearest_teammate_to_end",
    "defenders_within_3m_of_lane",
    "nearest_defender_to_lane",
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def intent_picks(train_rows: list[dict], target_rows: list[dict], variant: str) -> dict:
    """Which teammate the trained model picks, for every target row.

    The model is trained on the successful passes of the training rows only.
    The held-out match never contributes a label.
    """
    examples, _ = build_dataset(train_rows)
    if len(examples) < 50:
        return {}
    net, mean, std = fit_intent_model(examples, model=variant)
    picks = {}
    pool, _ = build_dataset(target_rows, require_label=False)
    for item in pool:
        picks[item["event_id"]] = predict_pick(net, item, mean, std)
    return picks


def apply_endpoint(rows: list[dict], picks: dict, source: str) -> list[dict]:
    """Put the picked teammate at the scored point."""
    out = []
    for row in rows:
        chosen = picks.get(row["event_id"])
        if chosen is None:
            continue
        point = None
        for candidate in row.get("intent_candidates") or []:
            if candidate.get("player") == chosen and candidate.get("xy") is not None:
                point = candidate["xy"]
                break
        if point is None:
            continue
        new = dict(row)
        new["end"] = (float(point[0]), float(point[1]))
        new["end_source"] = source
        out.append(new)
    return out


def rule_picks(rows: list[dict]) -> dict:
    return {row["event_id"]: row.get("intent_player") for row in rows}


def fold_endpoint_rows(rows: list[dict], variant: str) -> dict:
    """Endpoint rows per held-out match, with the intent model refit per fold."""
    matches = sorted({row["match_id"] for row in rows})
    out = {}
    for hold in matches:
        train = [row for row in rows if row["match_id"] != hold]
        test = [row for row in rows if row["match_id"] == hold]
        if len(train) < 50 or not test:
            continue
        if variant == "rule":
            picks = rule_picks(train + test)
        else:
            picks = intent_picks(train, train + test, variant)
        out[hold] = apply_endpoint(train + test, picks, f"intent_{variant}")
    return out


def spatial_signs(coefficients: dict) -> dict:
    return {name: coefficients.get(name) for name in SPATIAL_SIGNS}


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
    transform = choose_transform(grouped["train"], pitch_x, pitch_y)

    rows = {
        split: orient_samples(grouped[split], transform, pitch_x, pitch_y, end_mode="intent")
        for split in grouped
    }
    everything = [row for split in rows for row in rows[split]]
    print(f"passes with an intent target: {len(everything)}", flush=True)

    results = {"rules": RULES, "arms": {}}
    for arm in ARMS:
        print(f"=== arm {arm} ===", flush=True)
        if arm == "rule":
            picks = rule_picks(everything)
        else:
            picks = intent_picks(rows["train"], everything, arm)
        arm_rows = apply_endpoint(everything, picks, f"intent_{arm}")
        by_split = {split: [row for row in arm_rows if split_of(row["match_id"]) == split] for split in rows}

        geometry, spatial = fit_linear(by_split["train"])
        coefficients = spatial.coefficients()
        train_rate = float(np.mean([row["success"] for row in by_split["train"]]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{arm}] training the position raster on {device}", flush=True)
        raster, history = train_soccermap(by_split["train"], by_split["val"])
        torch.save(raster.state_dict(), OUT / f"position_raster_{arm}.pt")
        _write(OUT / f"train_history_{arm}.json", history)

        metrics = {}
        for split in ("val", "test"):
            raster_p = predict_destination(raster, by_split[split], device)
            metrics[split] = score_split(by_split[split], geometry, spatial, train_rate, raster_p)

        # leave one match out, with the endpoint refit inside each fold
        folds = fold_endpoint_rows(everything, arm)
        lomo = []
        for hold in sorted(folds):
            fold_rows = folds[hold]
            fold_train = [row for row in fold_rows if row["match_id"] != hold]
            fold_test = [row for row in fold_rows if row["match_id"] == hold]
            if len(fold_test) < 20 or len({row["success"] for row in fold_train}) < 2:
                continue
            fold_rate = float(np.mean([row["success"] for row in fold_train]))
            fold_geometry, fold_spatial = fit_linear(fold_train)
            lomo.append(
                {
                    "match_id": hold,
                    "n": len(fold_test),
                    **score_split(fold_test, fold_geometry, fold_spatial, fold_rate, None),
                }
            )
        _write(OUT / f"leave_one_match_out_{arm}.json", lomo)

        spatial_wins = sum(row["geometry_spatial"]["brier"] < row["geometry"]["brier"] for row in lomo)
        results["arms"][arm] = {
            "metrics": metrics,
            "lomo": lomo,
            "spatial_wins": f"{spatial_wins}/{len(lomo)}",
            "spatial_signs": spatial_signs(coefficients),
            "spatial_coefficients": coefficients,
            "geometry_coefficients": geometry.coefficients(),
            "rows": {split: len(by_split[split]) for split in by_split},
            "endpoint_equals_the_rule": {
                split: float(
                    np.mean(
                        [
                            picks.get(row["event_id"]) == row.get("intent_player")
                            for row in by_split[split]
                        ]
                    )
                )
                for split in by_split
            },
        }
        _write(OUT / f"metrics_{arm}.json", results["arms"][arm])

    _write(OUT / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
