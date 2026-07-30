# Ledger accumulation and track-record artifacts

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nlvswe.betting.clv import NOT_MEASURABLE_NOTE
from nlvswe.config import get_config, load_config
from nlvswe.io import read_table, write_table
from nlvswe.live.ledger import append_to_ledger, plot_track_record, refresh_track_record
from nlvswe.live.settle import settle_bet_slip
from nlvswe.schemas import validate_table

MATCH_A = "m_day1"
MATCH_B = "m_day2"
KICKOFF_A = pd.Timestamp("2026-06-20T17:00:00", tz="UTC")
KICKOFF_B = pd.Timestamp("2026-06-21T19:00:00", tz="UTC")


def _slip_row(match_id: str, kickoff: pd.Timestamp, captured: pd.Timestamp, *, stake: float = 20.0) -> dict:
    return {
        "match_id": match_id,
        "model": "poisson",
        "captured_at": captured,
        "kickoff_utc": kickoff,
        "bookmaker": "bet365",
        "market": "1x2",
        "selection": "home",
        "model_prob": 0.55,
        "book_prob_devig": 0.45,
        "edge": 0.10,
        "ev": 0.21,
        "odds_taken": 2.2,
        "stake_method": "kelly",
        "stake": stake,
        "status": "bet",
        "rationale": "value home",
    }


def _slip(match_id: str, kickoff: pd.Timestamp, captured: pd.Timestamp) -> pd.DataFrame:
    row = _slip_row(match_id, kickoff, captured)
    df = pd.DataFrame([row])
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    return validate_table(df, "bet_slip")


def _odds_single_snapshot(match_id: str, captured: pd.Timestamp) -> pd.DataFrame:
    rows = [
        {
            "match_id": match_id,
            "bookmaker": "bet365",
            "market": "1x2",
            "selection": sel,
            "decimal_odds": price,
            "captured_at": captured,
            "is_closing": True,
            "corpus": "international",
        }
        for sel, price in [("home", 2.2), ("draw", 3.4), ("away", 3.5)]
    ]
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _odds_two_snapshots(match_id: str, taken_at: pd.Timestamp, closing_at: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for captured, is_closing in ((taken_at, False), (closing_at, True)):
        for sel, price in [("home", 2.2 if not is_closing else 2.0), ("draw", 3.4), ("away", 3.5)]:
            rows.append(
                {
                    "match_id": match_id,
                    "bookmaker": "bet365",
                    "market": "1x2",
                    "selection": sel,
                    "decimal_odds": price,
                    "captured_at": captured,
                    "is_closing": is_closing,
                    "corpus": "international",
                }
            )
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def test_ledger_accumulates_two_matchdays(tmp_path: Path, monkeypatch):
    get_config.cache_clear()
    cfg = load_config()
    root = tmp_path / "proj"
    data = root / "data"
    live = data / "processed" / "live"
    live.mkdir(parents=True)
    figures = root / "reports" / "figures"
    figures.mkdir(parents=True)

    monkeypatch.setattr("nlvswe.io.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io._PROJECT_ROOT", root)
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", data)
    monkeypatch.setattr("nlvswe.io._FIGURES_ROOT", figures)
    monkeypatch.setattr("nlvswe.live.ledger.project_root", lambda: root)

    snap_a = pd.Timestamp("2026-06-20T16:00:00", tz="UTC")
    snap_b_taken = pd.Timestamp("2026-06-21T17:00:00", tz="UTC")
    snap_b_close = pd.Timestamp("2026-06-21T18:30:00", tz="UTC")

    day1 = settle_bet_slip(
        _slip(MATCH_A, KICKOFF_A, snap_a),
        home_goals=2,
        away_goals=0,
        odds=_odds_single_snapshot(MATCH_A, snap_a),
        devig_method=cfg.betting.devig_method,
    )
    day2 = settle_bet_slip(
        _slip(MATCH_B, KICKOFF_B, snap_b_taken),
        home_goals=1,
        away_goals=1,
        odds=_odds_two_snapshots(MATCH_B, snap_b_taken, snap_b_close),
        devig_method=cfg.betting.devig_method,
    )

    day1 = validate_table(day1, "live_results")
    day2 = validate_table(day2, "live_results")

    assert bool(day1.iloc[0]["measurable"]) is False
    assert day1.iloc[0]["clv_note"] == NOT_MEASURABLE_NOTE
    assert bool(day2.iloc[0]["measurable"]) is True
    assert day2.iloc[0]["clv_odds_ratio"] == pytest.approx(0.1)

    settled_at_a = pd.Timestamp("2026-06-20T22:00:00", tz="UTC")
    settled_at_b = pd.Timestamp("2026-06-21T22:00:00", tz="UTC")
    ledger = append_to_ledger(day1, cfg=cfg, settled_at=settled_at_a)
    ledger = append_to_ledger(day2, cfg=cfg, settled_at=settled_at_b)

    assert len(ledger) == 2
    assert list(ledger["bet_seq"]) == [1, 2]
    assert ledger.iloc[0]["match_id"] == MATCH_A
    assert ledger.iloc[1]["match_id"] == MATCH_B
    assert ledger.iloc[1]["cumulative_pnl"] == pytest.approx(float(ledger["pnl"].sum()))
    assert ledger.iloc[1]["bankroll"] == pytest.approx(cfg.betting.bankroll + float(ledger["pnl"].sum()))

    stored = read_table("ledger", "processed/live")
    assert len(stored) == 2

    refresh_track_record(ledger, cfg=cfg)
    assert (figures / "live_track_record.png").exists()
    assert (root / "reports" / "TRACK_RECORD.md").exists()


def test_plot_track_record_empty(tmp_path: Path, monkeypatch):
    figures = tmp_path / "figures"
    figures.mkdir()
    monkeypatch.setattr("nlvswe.io._FIGURES_ROOT", figures)
    plot_track_record(pd.DataFrame(), cfg=load_config())
    assert (figures / "live_track_record.png").exists()
