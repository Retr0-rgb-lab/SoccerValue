"""Which recorded passes are allowed into the checked subset.

An end point is kept only when the pass event itself wrote a destination.
Borrowing the next event confuses the aim with whoever touched the ball next.
On unsuccessful passes that next player is often an opponent.

A kick frame is kept only when it sits within one second of the event and the
tracked ball is within 3 m of the annotated kick. Three metres is the cut used
here; the public release itself reports about 2.6 m average ball error after
its own synchronisation.
"""

from __future__ import annotations

FRAME_ABS_LIMIT_S = 1.0
BALL_EVENT_LIMIT_M = 3.0

CHECKED_EVENTS = (
    {
        "match_id": "J03WMX",
        "event_id": "18226500000013",
        "finding": "Successful pass. The event itself writes an end that is not the receiver's next touch.",
    },
    {
        "match_id": "J03WMX",
        "event_id": "18226500000033",
        "finding": "Unsuccessful pass. The event writes its own end. The following tackle is at a different place, so the end was not taken from that tackle.",
    },
    {
        "match_id": "J03WMX",
        "event_id": "18226500000038",
        "finding": "Unsuccessful pass with no end of its own. The next play is by the named recipient, who is on the other team. That point is where the opponent played the ball, not a verified aim.",
    },
)


def end_is_on_event(row: dict) -> bool:
    return row.get("end_source") == "event"


def kick_frame_is_close(row: dict) -> bool:
    signed = row.get("frame_signed_dt")
    error = row.get("ball_event_m")
    if signed is None or error is None:
        return False
    return abs(float(signed)) <= FRAME_ABS_LIMIT_S and float(error) <= BALL_EVENT_LIMIT_M


def is_trusted(row: dict) -> bool:
    return end_is_on_event(row) and kick_frame_is_close(row)
