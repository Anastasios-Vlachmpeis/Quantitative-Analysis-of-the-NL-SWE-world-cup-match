# Resolve which international fixtures the live pipeline should price

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlvswe.config import AppConfig, TargetMatch, project_root


def fixture_kickoff(fixture: TargetMatch) -> pd.Timestamp:
    if not fixture.kickoff_utc:
        raise ValueError(f"Fixture {fixture.home} vs {fixture.away} is missing kickoff_utc")
    ts = pd.Timestamp(fixture.kickoff_utc)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def matchday_tag(kickoff: pd.Timestamp) -> str:
    day = pd.Timestamp(kickoff).tz_convert("UTC").strftime("%Y%m%d")
    return f"prematch-freeze-{day}"


def load_live_fixtures(cfg: AppConfig) -> list[TargetMatch]:
    # CSV list when configured; otherwise the single target_match block still works
    if cfg.live_fixtures:
        path = Path(cfg.live_fixtures)
        if not path.is_absolute():
            path = project_root() / path
        if path.exists():
            return _fixtures_from_csv(path)

    if cfg.target_matches:
        return list(cfg.target_matches)

    return [cfg.target_match]


def _fixtures_from_csv(path: Path) -> list[TargetMatch]:
    df = pd.read_csv(path)
    required = {"home", "away", "competition", "season", "kickoff_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"live_fixtures CSV missing columns: {sorted(missing)}")
    fixtures: list[TargetMatch] = []
    for _, row in df.iterrows():
        season = row["season"]
        fixtures.append(
            TargetMatch(
                home=str(row["home"]),
                away=str(row["away"]),
                competition=str(row["competition"]),
                season=int(season),
                kickoff_utc=str(row["kickoff_utc"]),
            )
        )
    return fixtures


def resolve_fixture_match(matches: pd.DataFrame, fixture: TargetMatch) -> pd.Series:
    home = fixture.home.lower().replace(" ", "_")
    away = fixture.away.lower().replace(" ", "_")
    mask = (matches["home_team_id"] == home) & (matches["away_team_id"] == away)
    sub = matches[mask].sort_values("date_utc", kind="mergesort")
    if sub.empty:
        raise ValueError(f"Fixture not found in matches table: {fixture.home} vs {fixture.away}")
    scheduled = sub[sub["status"].astype(str) == "scheduled"]
    return scheduled.iloc[-1] if not scheduled.empty else sub.iloc[-1]
