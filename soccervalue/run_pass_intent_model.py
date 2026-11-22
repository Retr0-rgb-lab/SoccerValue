"""Train an aimed receiver and compare it with the geometric rule.

The geometric rule reads the aim off the kick and takes the teammate closest to
that ray. It agrees with the annotated recipient on 76.0% of successful passes.
This round asks whether a trained model beats it from the same information.

Supervision comes from the annotated recipient on successful passes only. On an
unsuccessful pass the annotated recipient is the opponent who intercepted, so
there is no label for the aim. The training set is therefore filtered, and the
filter keeps the easy cases, so the retention rate is reported and the round
never claims the model was trained on the population it is scored on.

The rules below are written down before the run.

  model   GO only if the model beats the geometric rule on top-1 accuracy on the
          held-out test match, and on at least 5 of the 7 leave-one-match-out
          folds. A variant that does not clear the bar is not used as the
          estimator, whatever it looks like on the training split.
  report  The retention rate, the per-match table and the behaviour on failed
          passes are reported whatever the decision is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from soccervalue.align import choose_transform, orient_samples
from soccervalue.intent import FEATURE_NAMES
from soccervalue.intent_model import (
    build_dataset,
    fit_intent_model,
    label_rows,
    predict_pick,
    rule_picks,
    score_top1,
)
from soccervalue.matches import MATCH_COMPETITION, assert_split_disjoint, split_of
from soccervalue.run_pass_success import load_match

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pass_intent_model"

RULES = {
    "model": "GO only if the model beats the geometric rule on top-1 on the held-out test match and on at least 5 of the 7 leave-one-match-out folds.",
    "report": "The retention rate, the per-match table and the behaviour on failed passes are reported whatever the decision is.",
}

VARIANTS = ("linear", "mlp")


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _picks(net, mean, std, dataset) -> dict:
    return {item["event_id"]: predict_pick(net, item, mean, std) for item in dataset}


def leave_one_match_out(dataset: list[dict]) -> list[dict]:
    """Train on the other matches, score the held-out one."""
    matches = sorted({item["match_id"] for item in dataset})
    report = []
    for hold in matches:
        train = [item for item in dataset if item["match_id"] != hold]
        test = [item for item in dataset if item["match_id"] == hold]
        if len(train) < 50 or not test:
            continue
        entry = {"match_id": hold, "n_train": len(train), "n_test": len(test)}
        entry["rule"] = score_top1(test, rule_picks(test))["top1"]
        for variant in VARIANTS:
            net, mean, std = fit_intent_model(train, model=variant)
            entry[variant] = score_top1(test, _picks(net, mean, std, test))["top1"]
        report.append(entry)
    return report


def pick_report(net, mean, std, items: list[dict]) -> dict:
    """What the model changes relative to the geometric rule.

    On a failed pass there is no ground truth for the aim, so the only honest
    reports are how often the model keeps the rule's choice, how far each chosen
    teammate sits from the aim ray, and how much space each one has.
    """
    picks = _picks(net, mean, std, items)
    same = 0
    buckets: dict[str, list[tuple[float, float]]] = {"model": [], "rule": []}
    differing: dict[str, list[tuple[float, float]]] = {"model": [], "rule": []}
    for item in items:
        chosen = picks.get(item["event_id"])
        rule = item["rule_pick"]
        if chosen == rule:
            same += 1
        for player, row in zip(item["players"], item["features"]):
            if player == chosen:
                buckets["model"].append((float(row[1]) * 125.0, float(row[8]) * 125.0))
                if chosen != rule:
                    differing["model"].append((float(row[1]) * 125.0, float(row[8]) * 125.0))
            if player == rule:
                buckets["rule"].append((float(row[1]) * 125.0, float(row[8]) * 125.0))
                if chosen != rule:
                    differing["rule"].append((float(row[1]) * 125.0, float(row[8]) * 125.0))

    def summarise(rows: list[tuple[float, float]]) -> dict:
        if not rows:
            return {"n": 0}
        array = np.asarray(rows)
        return {
            "n": len(rows),
            "perp_to_ray_m": {"median": float(np.median(array[:, 0])), "p75": float(np.percentile(array[:, 0], 75))},
            "nearest_defender_m": {"median": float(np.median(array[:, 1])), "p75": float(np.percentile(array[:, 1], 75))},
        }

    return {
        "n": len(items),
        "same_pick_as_the_rule": (same / len(items)) if items else None,
        "picks_further_than_5m_off_the_ray": sum(1 for row in buckets["model"] if row[0] > 5.0),
        "model_pick": summarise(buckets["model"]),
        "rule_pick": summarise(buckets["rule"]),
        "where_they_differ": {
            "model_pick": summarise(differing["model"]),
            "rule_pick": summarise(differing["rule"]),
        },
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
    transform = choose_transform(grouped["train"], pitch_x, pitch_y)

    rows = {
        split: orient_samples(grouped[split], transform, pitch_x, pitch_y, end_mode="intent")
        for split in grouped
    }
    everything = [row for split in rows for row in rows[split]]
    dataset, skipped = build_dataset(everything)
    pool, _ = build_dataset(everything, require_label=False)
    successes = [item for item in dataset if item["success"] == 1]
    failures = [item for item in pool if item["success"] == 0]

    print(
        f"passes with an intent target: {len(everything)}; "
        f"with the recipient among the candidates: {len(dataset)}; skipped {skipped}",
        flush=True,
    )
    print(f"successful passes (usable as labels): {len(successes)}", flush=True)
    print(f"failed passes (no label, candidates only): {len(failures)}", flush=True)

    by_split = {split: [item for item in successes if split_of(item["match_id"]) == split] for split in rows}
    for split, items in by_split.items():
        print(f"[{split}] {len(items)} labelled passes", flush=True)

    results = {
        "rules": RULES,
        "features": list(FEATURE_NAMES),
        "coverage": {
            "passes_with_intent": len(everything),
            "with_recipient_in_candidates": len(dataset),
            "skipped_without_label": skipped,
            "successful_labelled": len(successes),
            "failed_labelled": len(failures),
            "training_retention_of_successful_passes": len(successes) / max(len(everything), 1),
        },    }

    # headline: train on the training split, score validation and test
    headline = {}
    for variant in VARIANTS:
        net, mean, std = fit_intent_model(by_split["train"], model=variant)
        entry = {}
        for split in ("val", "test"):
            items = by_split[split]
            rule = score_top1(items, rule_picks(items))
            model = score_top1(items, _picks(net, mean, std, items))
            entry[split] = {
                "n": len(items),
                "rule_top1": rule["top1"],
                "model_top1": model["top1"],
                "delta": (model["top1"] - rule["top1"]) if (model["top1"] is not None and rule["top1"] is not None) else None,
                "model_misses": model["misses"][:15],
            }
        headline[variant] = entry
    results["headline"] = headline

    lomo = leave_one_match_out(successes)
    results["leave_one_match_out"] = lomo
    results["per_match"] = {
        "rule": {
            entry["match_id"]: entry["rule"]
            for entry in lomo
        },
        **{variant: {entry["match_id"]: entry[variant] for entry in lomo} for variant in VARIANTS},
    }
    for variant in VARIANTS:
        wins = sum(1 for entry in lomo if entry[variant] is not None and entry["rule"] is not None and entry[variant] > entry["rule"])
        results.setdefault("fold_wins", {})[variant] = f"{wins}/{len(lomo)}"

    # What the model changes relative to the rule, on the passes with a label
    # and on the failed passes where no label exists.
    pick_diagnostics = {}
    for variant in VARIANTS:
        net, mean, std = fit_intent_model(by_split["train"], model=variant)
        pick_diagnostics[variant] = {
            "on_held_out_successful_passes": pick_report(net, mean, std, by_split["val"] + by_split["test"]),
            "on_failed_passes": pick_report(net, mean, std, failures),
        }
    results["pick_diagnostics"] = pick_diagnostics
    results["on_failed_passes"] = {
        variant: {
            "n": pick_diagnostics[variant]["on_failed_passes"]["n"],
            "same_pick_as_the_rule": pick_diagnostics[variant]["on_failed_passes"]["same_pick_as_the_rule"],
        }
        for variant in VARIANTS
    }

    # what the model learned about the rule it replaces
    net, mean, std = fit_intent_model(by_split["train"], model="linear")
    results["linear_weights"] = {
        name: float(value) for name, value in zip(FEATURE_NAMES, net.score.weight.detach().cpu().numpy().ravel())
    }

    _write(OUT / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
