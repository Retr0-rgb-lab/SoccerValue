"""Fixed match split for the public IDSSE / DFL release.

The seven match ids are the ones kloppy exposes for this dataset. The
5/1/1 assignment is written down before any model is fit and is not
chosen from validation scores.
"""

MATCH_COMPETITION = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}

TRAIN_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY")
VAL_MATCHES = ("J03WQQ",)
TEST_MATCHES = ("J03WR9",)


def split_of(match_id: str) -> str:
    if match_id in TRAIN_MATCHES:
        return "train"
    if match_id in VAL_MATCHES:
        return "val"
    if match_id in TEST_MATCHES:
        return "test"
    raise KeyError(match_id)


def assert_split_disjoint() -> None:
    train, val, test = set(TRAIN_MATCHES), set(VAL_MATCHES), set(TEST_MATCHES)
    if train & val or train & test or val & test:
        raise RuntimeError("match split overlaps")
    if train | val | test != set(MATCH_COMPETITION):
        raise RuntimeError("match split does not cover the public release")
