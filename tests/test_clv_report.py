# Tests for clv_report and measurable CLV bookkeeping

from __future__ import annotations

import pandas as pd
import pytest

from nlvswe.betting.clv import NOT_MEASURABLE_NOTE, clv_report
from nlvswe.data.clean import flag_is_closing


KICKOFF = pd.Timestamp("2026-06-20T17:00:00", tz="UTC")
MATCH_ID = "m-nl-swe"
BOOK = "pinnacle"


def _odds_rows(
    snapshots: list[tuple[pd.Timestamp, float]],
    *,
    is_closing_override: bool = False,
) -> pd.DataFrame:
    rows = []
    for captured_at, price in snapshots:
        rows.append(
            {
                "match_id": MATCH_ID,
                "bookmaker": BOOK,
                "market": "1x2",
                "selection": "home",
                "decimal_odds": price,
                "captured_at": captured_at,
                "is_closing": is_closing_override,
                "corpus": "international",
            }
        )
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _bet(captured_at: pd.Timestamp, odds_taken: float = 2.10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": MATCH_ID,
                "model": "ensemble_weighted",
                "captured_at": captured_at,
                "kickoff_utc": KICKOFF,
                "bookmaker": BOOK,
                "market": "1x2",
                "selection": "home",
                "model_prob": 0.55,
                "book_prob_devig": 0.50,
                "edge": 0.05,
                "ev": 0.08,
                "odds_taken": odds_taken,
                "stake_method": "kelly",
                "stake": 25.0,
                "status": "bet",
                "rationale": "test",
            }
        ]
    )


def test_clv_report_measurable_with_price_move():
    taken_at = pd.Timestamp("2026-06-20T11:00:00", tz="UTC")
    close_at = pd.Timestamp("2026-06-20T16:00:00", tz="UTC")
    odds = _odds_rows([(taken_at, 2.10), (close_at, 2.00)])
    matches = pd.DataFrame({"match_id": [MATCH_ID], "date_utc": [KICKOFF]})
    odds = flag_is_closing(odds, matches)

    report = clv_report(_bet(taken_at, odds_taken=2.10), odds)
    row = report.iloc[0]
    assert bool(row["measurable"]) is True
    assert row["clv_odds_ratio"] == pytest.approx(0.05)
    assert row["clv_note"] == ""


def test_clv_report_not_measurable_single_snapshot():
    snap_at = pd.Timestamp("2026-06-20T16:00:00", tz="UTC")
    odds = _odds_rows([(snap_at, 2.00)])
    matches = pd.DataFrame({"match_id": [MATCH_ID], "date_utc": [KICKOFF]})
    odds = flag_is_closing(odds, matches)

    report = clv_report(_bet(snap_at, odds_taken=2.00), odds)
    row = report.iloc[0]
    assert bool(row["measurable"]) is False
    assert row["clv_note"] == NOT_MEASURABLE_NOTE
    assert pd.isna(row["clv_odds_ratio"])
    assert row["clv_odds_ratio"] != 0.0


def test_flag_is_closing_marks_only_last_pre_kickoff():
    odds = _odds_rows(
        [
            (pd.Timestamp("2026-06-20T11:00:00", tz="UTC"), 2.10),
            (pd.Timestamp("2026-06-20T16:00:00", tz="UTC"), 2.00),
            (pd.Timestamp("2026-06-20T18:00:00", tz="UTC"), 1.95),
        ]
    )
    matches = pd.DataFrame({"match_id": [MATCH_ID], "date_utc": [KICKOFF]})
    out = flag_is_closing(odds, matches)
    closing = out[out["is_closing"]]
    assert len(closing) == 1
    assert closing.iloc[0]["decimal_odds"] == 2.00
    assert closing.iloc[0]["captured_at"] == pd.Timestamp("2026-06-20T16:00:00", tz="UTC")
