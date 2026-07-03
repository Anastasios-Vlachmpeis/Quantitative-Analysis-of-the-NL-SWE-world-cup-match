"""Chronological Elo over international results (cleaning, featurizing, and the Elo model).

Defaults: 1500 for unseen teams, K=20, +100 home advantage in expectation,
ln-margin multiplier on updates. Each row is dated at kickoff so you can't
accidentally leak the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 100.0
GOAL_MARGIN_MULTIPLIER = 2.0 / 3.0

CORPUS_INTERNATIONAL = "international"


@dataclass(frozen=True)
class EloParams:
    initial: float = INITIAL_ELO
    k: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    goal_margin_multiplier: float = GOAL_MARGIN_MULTIPLIER


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(goal_diff: int, params: EloParams) -> float:
    if goal_diff <= 1:
        return 1.0
    return GOAL_MARGIN_MULTIPLIER * math.log(goal_diff + 1)


def apply_update(m: pd.Series, ratings: dict[str, float], params: EloParams) -> None:
    """Apply one completed match result to the in-memory rating state."""
    home_id = m["home_team_id"]
    away_id = m["away_team_id"]
    home_goals = int(m["home_goals"])
    away_goals = int(m["away_goals"])
    neutral = bool(m["neutral"])

    rh = ratings.setdefault(home_id, params.initial)
    ra = ratings.setdefault(away_id, params.initial)
    home_eff = rh + (0.0 if neutral else params.home_advantage)

    if home_goals > away_goals:
        score_home, score_away = 1.0, 0.0
    elif home_goals < away_goals:
        score_home, score_away = 0.0, 1.0
    else:
        score_home = score_away = 0.5

    exp_home = expected_score(home_eff, ra)
    exp_away = 1.0 - exp_home
    mult = margin_multiplier(abs(home_goals - away_goals), params)
    k = params.k * mult

    ratings[home_id] = rh + k * (score_home - exp_home)
    ratings[away_id] = ra + k * (score_away - exp_away)


def _completed_intl(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[(matches["status"] == "completed") & (matches["corpus"] == CORPUS_INTERNATIONAL)].copy()
    return m.sort_values(["date_utc", "match_id"], kind="mergesort")


def elo_pre_at_kickoff(
    match_row: pd.Series,
    history: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> tuple[float, float]:
    """Pre-kickoff Elo for home/away using only history with date_utc < kickoff."""
    params = params or EloParams()
    kickoff = pd.Timestamp(match_row["date_utc"]).tz_convert("UTC")
    prior = history[
        (history["status"] == "completed")
        & (history["corpus"] == CORPUS_INTERNATIONAL)
        & (history["date_utc"] < kickoff)
    ].sort_values(["date_utc", "match_id"], kind="mergesort")

    ratings: dict[str, float] = {}
    for _, m in prior.iterrows():
        apply_update(m, ratings, params)

    home_id = match_row["home_team_id"]
    away_id = match_row["away_team_id"]
    return (
        ratings.get(home_id, params.initial),
        ratings.get(away_id, params.initial),
    )


def compute_elo_pre_table(
    matches: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> pd.DataFrame:
    """Return match_id, home_elo_pre, away_elo_pre for every completed intl match."""
    params = params or EloParams()
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, m in _completed_intl(matches).iterrows():
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        rh = ratings.get(home_id, params.initial)
        ra = ratings.get(away_id, params.initial)
        rows.append(
            {
                "match_id": m["match_id"],
                "home_elo_pre": float(rh),
                "away_elo_pre": float(ra),
            }
        )
        apply_update(m, ratings, params)

    return pd.DataFrame(rows)


def roll_elo(
    matches: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> pd.DataFrame:
    """Return long-format post-match ratings (source='elo') from completed matches."""
    params = params or EloParams()
    required = {"date_utc", "home_team_id", "away_team_id", "home_goals", "away_goals", "neutral", "status"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns for Elo: {missing}")

    hist = matches[matches["status"] == "completed"].copy()
    hist = hist.sort_values(["date_utc", "match_id"], kind="mergesort")

    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, m in hist.iterrows():
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        apply_update(m, ratings, params)

        rating_date = pd.Timestamp(m["date_utc"])
        if rating_date.tzinfo is None:
            rating_date = rating_date.tz_localize("UTC")
        else:
            rating_date = rating_date.tz_convert("UTC")

        rows.append(
            {
                "team_id": home_id,
                "rating_date": rating_date,
                "source": "elo",
                "value": float(ratings[home_id]),
                "rank": None,
            }
        )
        rows.append(
            {
                "team_id": away_id,
                "rating_date": rating_date,
                "source": "elo",
                "value": float(ratings[away_id]),
                "rank": None,
            }
        )

    return pd.DataFrame(rows)
