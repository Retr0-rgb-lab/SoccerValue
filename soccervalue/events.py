"""Open-play flat passes from Sportec event XML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lxml import etree

SET_PIECES = {"KickOff", "ThrowIn", "CornerKick", "FreeKick", "GoalKick", "Penalty"}
SUCCESS_VALUES = {"successfullycompleted", "successful"}
FAIL_VALUES = {"unsuccessful"}


def pitch_size(meta_path: Path) -> tuple[float, float]:
    root = etree.parse(str(meta_path)).getroot()
    env = root.find(".//Environment")
    if env is None:
        raise ValueError(f"no pitch size in {meta_path}")
    return float(env.get("PitchX")), float(env.get("PitchY"))


def _coords(elem: etree._Element, source: bool) -> tuple[float, float] | None:
    prefix = "X-Source-Position" if source else "X-Position"
    y_key = "Y-Source-Position" if source else "Y-Position"
    if elem.get(prefix) is None or elem.get(y_key) is None:
        return None
    return float(elem.get(prefix)), float(elem.get(y_key))


def load_open_flat_passes(events_path: Path, match_id: str) -> list[dict]:
    """Return open-play flat passes with a start and an end location.

    A pass whose own end coordinate repeats the start uses the next event
    location. Sportec often records the kick on this event and the arrival
    on the following one. Set pieces, high balls, and goalkeeper throw-outs
    are left out.
    """
    raw: list[dict] = []
    team_left_by_time: list[tuple[datetime, str]] = []
    for _event, elem in etree.iterparse(str(events_path), events=("end",), tag="Event"):
        when = datetime.fromisoformat(elem.get("EventTime"))
        kick = elem.find("./KickOff")
        if kick is not None and kick.get("TeamLeft"):
            team_left_by_time.append((when, kick.get("TeamLeft")))
        direct = list(elem)
        direct_tag = direct[0].tag if direct else ""
        play = elem.find("./Play")
        is_pass = play is not None and play.find("./Pass") is not None
        start = _coords(elem, source=True) or _coords(elem, source=False)
        end_on_event = _coords(elem, source=False)
        record = {
            "time": when,
            "event_id": elem.get("EventId"),
            "event_tag": direct_tag,
            "is_pass": is_pass and direct_tag not in SET_PIECES,
            "team_id": play.get("Team") if play is not None else None,
            "player_id": play.get("Player") if play is not None else None,
            "recipient": play.get("Recipient") if play is not None else None,
            "from_open_play": play.get("FromOpenPlay") if play is not None else None,
            "height": play.get("Height") if play is not None else None,
            "goalkeeper_action": play.get("GoalKeeperAction") if play is not None else None,
            "evaluation": play.get("Evaluation") if play is not None else None,
            "start": start,
            "end_on_event": end_on_event,
            "location": start,
        }
        raw.append(record)
        elem.clear()

    passes: list[dict] = []
    for index, record in enumerate(raw):
        if not record["is_pass"]:
            continue
        if record["from_open_play"] != "true":
            continue
        if record["height"] != "flat":
            continue
        if record["goalkeeper_action"]:
            continue
        evaluation = (record["evaluation"] or "").lower()
        if evaluation in SUCCESS_VALUES:
            success = 1
        elif evaluation in FAIL_VALUES:
            success = 0
        else:
            continue
        start = record["start"]
        end = record["end_on_event"]
        if start is None or end is None:
            continue
        end_source = "event"
        next_relation = "same_event"
        next_tag = None
        end_gap_s = 0.0
        if _distance(start, end) <= 1.0:
            nxt = _next_record(raw, index, record["time"])
            if nxt is None or nxt["location"] is None:
                continue
            end = nxt["location"]
            end_source = "next_event"
            next_relation = _next_relation(record, nxt)
            next_tag = nxt["event_tag"] or None
            end_gap_s = abs((nxt["time"] - record["time"]).total_seconds())
        if end is None or _distance(start, end) <= 1.0:
            continue
        team_left = _team_left_at(team_left_by_time, record["time"])
        if team_left is None:
            continue
        passes.append(
            {
                "match_id": match_id,
                "event_id": record["event_id"],
                "time": record["time"],
                "team_id": record["team_id"],
                "player_id": record["player_id"],
                "start_x": start[0],
                "start_y": start[1],
                "end_x": end[0],
                "end_y": end[1],
                "success": success,
                "team_left": team_left,
                "end_source": end_source,
                "next_event_relation": next_relation,
                "next_event_tag": next_tag,
                "end_gap_s": end_gap_s,
            }
        )
    return passes


LABEL_DEFINITION = (
    "The label is the Sportec Evaluation of an open-play flat pass. "
    "It is the recorded outcome, not a known intended target. "
    "When the event repeats the kick coordinate as its end, the point used "
    "as the destination is the next event within 8 seconds. On an unsuccessful "
    "pass that next event is often a tackle, clearance, or the opponent's first "
    "touch, so the coordinate can be where the ball was won rather than where "
    "it was aimed. A model that sees this point predicts the official outcome "
    "given the recorded point. It does not score an unexecuted aim."
)


def label_audit(passes: list[dict]) -> dict:
    """Count where the destination coordinate came from, by outcome."""
    rows = []
    for source in ("event", "next_event"):
        for success in (1, 0):
            chosen = [item for item in passes if item["end_source"] == source and item["success"] == success]
            relations: dict[str, int] = {}
            tags: dict[str, int] = {}
            for item in chosen:
                relation = item.get("next_event_relation") or "same_event"
                relations[relation] = relations.get(relation, 0) + 1
                tag = item.get("next_event_tag") or ""
                if tag:
                    tags[tag] = tags.get(tag, 0) + 1
            rows.append(
                {
                    "end_source": source,
                    "success": success,
                    "n": len(chosen),
                    "next_event_relation": relations,
                    "next_event_tag": tags,
                }
            )
    failures = [item for item in passes if item["success"] == 0]
    borrowed_failures = [item for item in failures if item["end_source"] == "next_event"]
    return {
        "label_definition": LABEL_DEFINITION,
        "n": len(passes),
        "n_success": sum(item["success"] == 1 for item in passes),
        "n_failure": len(failures),
        "fraction_end_from_next_event": _share(passes, "next_event"),
        "fraction_failures_end_from_next_event": _share(failures, "next_event"),
        "fraction_borrowed_failures_at_opponent": _share(borrowed_failures, "opponent", key="next_event_relation"),
        "by_source_and_outcome": rows,
    }


def _share(items: list[dict], value: str, key: str = "end_source") -> float | None:
    if not items:
        return None
    return sum(item.get(key) == value for item in items) / len(items)


def _next_record(raw: list[dict], index: int, when: datetime) -> dict | None:
    if index + 1 >= len(raw):
        return None
    nxt = raw[index + 1]
    if abs((nxt["time"] - when).total_seconds()) > 8:
        return None
    return nxt


def _next_relation(record: dict, nxt: dict) -> str:
    if nxt.get("player_id") and nxt["player_id"] == record.get("recipient"):
        return "recipient"
    if nxt.get("team_id") and record.get("team_id") and nxt["team_id"] == record["team_id"]:
        return "teammate"
    if nxt.get("team_id") and record.get("team_id") and nxt["team_id"] != record["team_id"]:
        return "opponent"
    return "unknown"


def _team_left_at(markers: list[tuple[datetime, str]], when: datetime) -> str | None:
    current = None
    for marked_at, team in markers:
        if marked_at <= when:
            current = team
        else:
            break
    return current


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
