"""Multi-fixture live pipeline tests (Plan 16 Part B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nlvswe.config import AppConfig, TargetMatch, get_config, load_config
from nlvswe.io import read_table, write_table
from nlvswe.live.fixtures import fixture_kickoff, load_live_fixtures, matchday_tag
from nlvswe.live.predict import run_live_predict
from nlvswe.live.settle import _load_bet_slip_for_match
from nlvswe.models.base import MatchPrediction
from nlvswe.schemas import validate_table


def _two_fixtures() -> list[TargetMatch]:
    return [
        TargetMatch(
            home="Netherlands",
            away="Sweden",
            competition="FIFA World Cup",
            season=2026,
            kickoff_utc="2026-06-20T17:00:00Z",
        ),
        TargetMatch(
            home="Germany",
            away="France",
            competition="FIFA World Cup",
            season=2026,
            kickoff_utc="2026-06-21T19:00:00Z",
        ),
    ]


def _multi_matches() -> pd.DataFrame:
    rows = [
        {
            "match_id": "m3",
            "date_utc": pd.Timestamp("2024-03-01", tz="UTC"),
            "corpus": "international",
            "competition": "Friendly",
            "season": "2024",
            "stage": None,
            "home_team_id": "netherlands",
            "away_team_id": "sweden",
            "venue_id": None,
            "neutral": False,
            "home_goals": 3,
            "away_goals": 0,
            "status": "completed",
            "went_to_shootout": False,
        },
        {
            "match_id": "m4",
            "date_utc": pd.Timestamp("2026-06-20T17:00:00", tz="UTC"),
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
        },
        {
            "match_id": "m5",
            "date_utc": pd.Timestamp("2026-06-21T19:00:00", tz="UTC"),
            "corpus": "international",
            "competition": "FIFA World Cup",
            "season": "2026",
            "stage": "group",
            "home_team_id": "germany",
            "away_team_id": "france",
            "venue_id": None,
            "neutral": True,
            "home_goals": pd.NA,
            "away_goals": pd.NA,
            "status": "scheduled",
            "went_to_shootout": pd.NA,
        },
    ]
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    df["went_to_shootout"] = df["went_to_shootout"].astype("boolean")
    return df


def _train_features() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append(
            {
                "match_id": f"t{i:02d}",
                "date_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i * 14),
                "corpus": "international",
                "competition": "Friendly",
                "result_1x2": "home",
            }
        )
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _mock_odds(match_ids: list[str]) -> pd.DataFrame:
    rows = []
    kickoffs = {
        "m4": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
        "m5": pd.Timestamp("2026-06-21T18:00:00", tz="UTC"),
    }
    for mid in match_ids:
        for sel, price in [("home", 2.2), ("draw", 3.4), ("away", 3.5)]:
            rows.append(
                {
                    "match_id": mid,
                    "bookmaker": "bet365",
                    "market": "1x2",
                    "selection": sel,
                    "decimal_odds": price,
                    "captured_at": kickoffs[mid],
                    "is_closing": False,
                    "corpus": "international",
                }
            )
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _empty_interim_tables() -> dict[str, pd.DataFrame]:
    return {
        "ratings": pd.DataFrame(columns=["team_id", "rating_date", "source", "corpus", "value", "rank"]),
        "conditions": pd.DataFrame(
            columns=["match_id", "kickoff_local", "temp_c", "humidity", "weather", "altitude_m"]
        ),
        "venues": pd.DataFrame(
            columns=["venue_id", "name", "city", "country_code", "lat", "lon", "altitude_m", "capacity"]
        ),
    }


class _StubPredictor:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, train: pd.DataFrame) -> None:
        self.train = train

    def predict(self, target_row: pd.Series):
        mat = np.ones((6, 6), dtype=float) / 36.0
        pred = MatchPrediction(
            match_id=str(target_row["match_id"]),
            scoreline=mat,
            probs_1x2={"home": 0.55, "draw": 0.25, "away": 0.20},
        )
        return pred, None


def test_load_live_fixtures_from_csv():
    get_config.cache_clear()
    cfg = load_config()
    fixtures = load_live_fixtures(cfg)
    assert len(fixtures) >= 1
    assert fixtures[0].home == "Netherlands"
    assert fixtures[0].away == "Sweden"


def test_load_live_fixtures_fallback_to_target_match():
    get_config.cache_clear()
    cfg = load_config()
    cfg = cfg.model_copy(update={"live_fixtures": "config/does_not_exist.csv", "target_matches": None})
    fixtures = load_live_fixtures(cfg)
    assert len(fixtures) == 1
    assert fixtures[0].home == cfg.target_match.home


def test_load_live_fixtures_uses_target_matches_list():
    get_config.cache_clear()
    cfg = load_config()
    pair = _two_fixtures()
    cfg = cfg.model_copy(update={"live_fixtures": None, "target_matches": pair})
    assert load_live_fixtures(cfg) == pair


def test_matchday_tag():
    kickoff = pd.Timestamp("2026-06-20T17:00:00Z")
    assert matchday_tag(kickoff) == "prematch-freeze-20260620"
    assert fixture_kickoff(_two_fixtures()[0]) == kickoff


def test_two_fixture_run_end_to_end(tmp_path: Path, monkeypatch):
    get_config.cache_clear()
    cfg: AppConfig = load_config()
    cfg = cfg.model_copy(update={"live_fixtures": None, "target_matches": _two_fixtures()})

    root = tmp_path / "proj"
    data = root / "data"
    interim = data / "interim"
    processed = data / "processed"
    reports = root / "reports"
    figures = reports / "figures"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    figures.mkdir(parents=True)

    monkeypatch.setattr("nlvswe.live.predict.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io._PROJECT_ROOT", root)
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", data)
    monkeypatch.setattr("nlvswe.io._FIGURES_ROOT", figures)
    monkeypatch.setattr("nlvswe.live.predict.LivePredictor", _StubPredictor)
    monkeypatch.setattr("nlvswe.live.predict.git_commit", lambda: "deadbeef")
    monkeypatch.setattr("nlvswe.live.predict.plot_live_figures", lambda *a, **k: None)
    monkeypatch.setattr("nlvswe.live.predict.plot_live_edges", lambda *a, **k: None)
    train = _train_features()
    monkeypatch.setattr("nlvswe.live.predict.load_model_features", lambda: train)

    empty = _empty_interim_tables()
    write_table(_multi_matches(), "matches", "interim", sort_by=["date_utc", "match_id"], plan="16")
    for name, frame in empty.items():
        write_table(frame, name, "interim", plan="16")
    write_table(_mock_odds(["m4", "m5"]), "odds", "interim", sort_by=["match_id"], plan="16")

    result = run_live_predict(cfg, model_name="poisson")

    assert result["n_fixtures"] == 2
    assert set(result["match_ids"]) == {"m4", "m5"}

    live = data / "processed" / "live"
    assert (live / "bet_slip_m4.parquet").exists()
    assert (live / "bet_slip_m5.parquet").exists()
    assert (live / "bet_slip.parquet").exists()

    combined = read_table("bet_slip", "processed/live")
    assert set(combined["match_id"].astype(str)) == {"m4", "m5"}

    pred_md = (reports / "PREDICTION.md").read_text(encoding="utf-8")
    assert "## Netherlands vs Sweden (`m4`)" in pred_md
    assert "## Germany vs France (`m5`)" in pred_md
    assert "**Fixtures priced:** 2" in pred_md


def test_settle_loads_per_fixture_slip(tmp_path: Path, monkeypatch):
    root = tmp_path / "proj"
    data = root / "data"
    live = data / "processed" / "live"
    live.mkdir(parents=True)

    monkeypatch.setattr("nlvswe.live.settle.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io.project_root", lambda: root)
    monkeypatch.setattr("nlvswe.io._PROJECT_ROOT", root)
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", data)

    slip_m4 = pd.DataFrame(
        [
            {
                "match_id": "m4",
                "model": "poisson",
                "captured_at": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
                "kickoff_utc": pd.Timestamp("2026-06-20T17:00:00", tz="UTC"),
                "bookmaker": "bet365",
                "market": "1x2",
                "selection": "home",
                "model_prob": 0.55,
                "book_prob_devig": 0.45,
                "edge": 0.10,
                "ev": 0.21,
                "odds_taken": 2.2,
                "stake_method": "kelly",
                "stake": 10.0,
                "status": "bet",
                "rationale": "value home",
            }
        ]
    )
    slip_m4["captured_at"] = pd.to_datetime(slip_m4["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    slip_m4["kickoff_utc"] = pd.to_datetime(slip_m4["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    slip_m4 = validate_table(slip_m4, "bet_slip")
    write_table(slip_m4, "bet_slip_m4", "processed/live", sort_by=["status", "ev"], plan="16", schema_name="bet_slip")

    loaded = _load_bet_slip_for_match("m4")
    assert len(loaded) == 1
    assert loaded.iloc[0]["match_id"] == "m4"
