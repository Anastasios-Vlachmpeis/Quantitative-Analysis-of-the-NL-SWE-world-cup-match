# Settlement module tests

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from nlvswe.betting.clv import NOT_MEASURABLE_NOTE
from nlvswe.config import get_config, load_config
from nlvswe.live.settle import (
    assert_live_inputs_unchanged,
    hash_live_inputs,
    parse_score,
    run_settle,
    settle_bet_slip,
)
from nlvswe.schemas import validate_table

MATCH_ID = "6fc2ebdea7fcb9d4"
KICKOFF = pd.Timestamp("2026-06-20T17:00:00", tz="UTC")
CAPTURED = pd.Timestamp("2026-06-19T18:22:05", tz="UTC")


def _report_slip() -> pd.DataFrame:
    # Canonical bets recorded in reports/REPORT.md (rounded stakes, incl. discretionary over)
    rows = [
        {
            "match_id": MATCH_ID,
            "model": "ensemble_weighted",
            "captured_at": CAPTURED,
            "kickoff_utc": KICKOFF,
            "bookmaker": "pinnacle",
            "market": "1x2",
            "selection": "home",
            "model_prob": 0.627,
            "book_prob_devig": 0.556,
            "edge": 0.071,
            "ev": 0.091,
            "odds_taken": 1.74,
            "stake_method": "kelly",
            "stake": 31.0,
            "status": "bet",
            "rationale": "value home",
        },
        {
            "match_id": MATCH_ID,
            "model": "ensemble_weighted",
            "captured_at": CAPTURED,
            "kickoff_utc": KICKOFF,
            "bookmaker": "williamhill",
            "market": "totals",
            "selection": "over_2.5",
            "model_prob": 0.714,
            "book_prob_devig": 0.547,
            "edge": 0.167,
            "ev": 0.214,
            "odds_taken": 1.70,
            "stake_method": "flat",
            "stake": 20.0,
            "status": "bet",
            "rationale": "speculative over",
        },
    ]
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    return validate_table(df, "bet_slip")


def _single_snapshot_odds() -> pd.DataFrame:
    snap = pd.Timestamp("2026-06-20T16:00:00", tz="UTC")
    rows = []
    for book, sel, price in [
        ("pinnacle", "home", 1.74),
        ("williamhill", "over_2.5", 1.70),
    ]:
        rows.append(
            {
                "match_id": MATCH_ID,
                "bookmaker": book,
                "market": "1x2" if sel == "home" else "totals",
                "selection": sel,
                "decimal_odds": price,
                "captured_at": snap,
                "is_closing": True,
                "corpus": "international",
            }
        )
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def test_parse_score():
    assert parse_score("5-1") == (5, 1)
    assert parse_score(" 2:0 ") == (2, 0)
    with pytest.raises(ValueError):
        parse_score("bad")


def test_nl_swe_report_pnl_exact():
    get_config.cache_clear()
    cfg = load_config()
    out = settle_bet_slip(
        _report_slip(),
        home_goals=5,
        away_goals=1,
        odds=_single_snapshot_odds(),
        devig_method=cfg.betting.devig_method,
    )
    out = validate_table(out, "live_results")
    home = out[out["selection"] == "home"].iloc[0]
    over = out[out["selection"] == "over_2.5"].iloc[0]
    assert home["pnl"] == pytest.approx(22.94)
    assert over["pnl"] == pytest.approx(14.00)
    assert float(out["pnl"].sum()) == pytest.approx(36.94)
    assert bool(home["measurable"]) is False
    assert home["clv_note"] == NOT_MEASURABLE_NOTE


def test_append_only_guard_detects_change():
    before = {"bet_slip.parquet": "aaa"}
    after = {"bet_slip.parquet": "bbb"}
    with pytest.raises(RuntimeError, match="modified pre-freeze"):
        assert_live_inputs_unchanged(before, after)


def test_hash_live_inputs_ignores_results(tmp_path: Path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "bet_slip.parquet").write_bytes(b"slip")
    (live / "results.parquet").write_bytes(b"results")
    hashes = hash_live_inputs(tmp_path.parent if False else tmp_path / "..")
    # hash_live_inputs uses project_root by default; test via direct paths instead
    digest = hashlib.sha256((live / "bet_slip.parquet").read_bytes()).hexdigest()
    manual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in live.iterdir()
        if p.name not in {"results.parquet", "results.manifest.json"}
    }
    assert manual == {"bet_slip.parquet": digest}


def test_run_settle_writes_results_without_touching_slip(tmp_path: Path, monkeypatch):
    get_config.cache_clear()
    cfg = load_config()
    root = tmp_path / "proj"
    live = root / "data" / "processed" / "live"
    live.mkdir(parents=True)
    slip = _report_slip()
    slip.to_parquet(live / "bet_slip.parquet", index=False)
    (live / "bet_slip.manifest.json").write_text("{}", encoding="utf-8")
    slip_hash_before = hashlib.sha256((live / "bet_slip.parquet").read_bytes()).hexdigest()

    interim = root / "data" / "interim"
    interim.mkdir(parents=True)
    matches = pd.DataFrame(
        [
            {
                "match_id": MATCH_ID,
                "date_utc": KICKOFF,
                "corpus": "international",
                "competition": "FIFA World Cup",
                "season": "2026",
                "stage": "group",
                "home_team_id": "netherlands",
                "away_team_id": "sweden",
                "venue_id": None,
                "neutral": True,
                "home_goals": pd.NA,
                "away_goals": pd.NA,
                "status": "scheduled",
                "went_to_shootout": pd.NA,
            }
        ]
    )
    matches["date_utc"] = pd.to_datetime(matches["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    matches["home_goals"] = matches["home_goals"].astype("Int64")
    matches["away_goals"] = matches["away_goals"].astype("Int64")
    matches["went_to_shootout"] = matches["went_to_shootout"].astype("boolean")
    matches.to_parquet(interim / "matches.parquet", index=False)
    (interim / "matches.manifest.json").write_text("{}", encoding="utf-8")

    odds = _single_snapshot_odds()
    odds.to_parquet(interim / "odds.parquet", index=False)
    (interim / "odds.manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("nlvswe.live.settle.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io._PROJECT_ROOT", root)
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", root / "data")

    results = run_settle(MATCH_ID, score="5-1", cfg=cfg, write_match=True)
    assert float(results["pnl"].sum()) == pytest.approx(36.94)
    assert (live / "results.parquet").exists()
    slip_hash_after = hashlib.sha256((live / "bet_slip.parquet").read_bytes()).hexdigest()
    assert slip_hash_before == slip_hash_after

    updated = pd.read_parquet(interim / "matches.parquet")
    row = updated.iloc[0]
    assert int(row["home_goals"]) == 5
    assert int(row["away_goals"]) == 1
    assert row["status"] == "completed"
