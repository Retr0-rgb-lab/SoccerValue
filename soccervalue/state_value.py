"""State samples at completed actions, labeled by shots inside a fixed window.

Team T is fixed here. A later turnover does not move the positive head
onto whoever wins the ball. Samples whose window meets a whistle or the
next kickoff are dropped instead of being called shot-free.

The window carries two labels for the same shots. `y_for` and `y_against` say
that a shot happened. `xg_for` and `xg_against` say how much expected goal the
window produced. The second label is graded rather than binary, which is what
un-xPass uses and what the sparse shot label could not afford. The dataset
annotates every shot with its own xG, so the second label needs no extra model.
"""

from __future__ import annotations

from soccervalue.events import FAIL_VALUES, SUCCESS_VALUES


def window_xg(values: list[float]) -> tuple[float, float]:
    """The xG a window produced, as a sum and as the un-xPass probability.

    The sum is the expected goal added inside the window. The product form,
    `1 - prod(1 - xG)`, is the chance that one of those shots is converted, and
    it is the label the un-xPass value model uses.
    """
    if not values:
        return (0.0, 0.0)
    total = 1.0
    for value in values:
        total *= 1.0 - float(value)
    return (float(sum(values)), 1.0 - total)


def build_state_samples(events: list[dict], window_s: float = 10.0) -> list[dict]:
    """Sample action completions and label shots in (t, t + window].

    Team T is fixed here. A later turnover does not move the positive head
    onto whoever wins the ball. Samples whose window meets a whistle or the
    next kickoff are dropped instead of being called shot-free.
    """
    ordered = sorted(events, key=lambda item: item["time"])
    boundaries = [item for item in ordered if item.get("event_tag") in {"FinalWhistle", "KickOff"}]
    shots = [item for item in ordered if item.get("is_shot")]
    samples = []
    for event in ordered:
        base = _candidate(event)
        if base is None:
            continue
        boundary = next((item for item in boundaries if item["time"] > event["time"]), None)
        if boundary is None:
            continue
        gap = (boundary["time"] - event["time"]).total_seconds()
        if gap < window_s or event.get("game_section") != boundary.get("game_section"):
            continue
        y_for = 0
        y_against = 0
        xg_for = []
        xg_against = []
        seen = []
        for shot in shots:
            delay = (shot["time"] - event["time"]).total_seconds()
            if not 0 < delay <= window_s:
                continue
            if shot.get("team_id") == base["team_id"]:
                y_for = 1
                if shot.get("xg") is not None:
                    xg_for.append(float(shot["xg"]))
            elif shot.get("team_id"):
                y_against = 1
                if shot.get("xg") is not None:
                    xg_against.append(float(shot["xg"]))
            seen.append({"team_id": shot.get("team_id"), "offset_s": delay, "xg": shot.get("xg")})
        base["y_for"] = y_for
        base["y_against"] = y_against
        base["xg_for_sum"], base["xg_for_product"] = window_xg(xg_for)
        base["xg_against_sum"], base["xg_against_product"] = window_xg(xg_against)
        base["window_s"] = window_s
        base["shots_in_window"] = seen
        samples.append(base)
    return samples


def _candidate(event: dict) -> dict | None:
    if event.get("x") is None or event.get("y") is None or not event.get("team_id"):
        return None
    tag = event.get("event_tag")
    if tag == "Play":
        evaluation = (event.get("evaluation") or "").lower()
        if evaluation in SUCCESS_VALUES:
            possession = 1
        elif evaluation in FAIL_VALUES:
            possession = 0
        else:
            return None
    elif tag == "OtherBallAction":
        possession = 1
    elif tag == "TacklingGame" and event.get("possession_change"):
        possession = 1
    else:
        return None
    return {
        "time": event["time"],
        "event_id": event.get("event_id"),
        "match_id": event.get("match_id"),
        "team_id": event["team_id"],
        "team_left": event.get("team_left"),
        "x": event["x"],
        "y": event["y"],
        "player_id": event.get("player_id"),
        "possession_by_t": possession,
        "game_section": event.get("game_section"),
    }
