# Match-importance bucketing and the v2 goal-rate covariate

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nlvswe.models._goals_common import fit_importance_adjustments, fit_poisson_strengths, predict_lambdas
from nlvswe.models.dixon_coles_v2 import DixonColesV2Model
from nlvswe.models.match_importance import (
    FRIENDLY,
    QUALIFIER,
    TOURNAMENT_GROUP,
    TOURNAMENT_KNOCKOUT,
    match_importance,
    match_importance_from_row,
)

CORPUS = "international"


@pytest.mark.parametrize(
    "competition,stage,is_knockout,expected",
    [
        ("International Friendly", None, False, FRIENDLY),
        ("FIFA World Cup qualification UEFA", "Group A", False, QUALIFIER),
        ("FIFA World Cup", "Group stage", False, TOURNAMENT_GROUP),
        ("UEFA Euro", "Group B", False, TOURNAMENT_GROUP),
        ("FIFA World Cup", "Quarter-final", True, TOURNAMENT_KNOCKOUT),
        ("Copa America", "Semi-final", False, TOURNAMENT_KNOCKOUT),
    ],
)
def test_match_importance_categories(competition, stage, is_knockout, expected):
    assert match_importance(competition=competition, stage=stage, is_knockout=is_knockout) == expected


def test_match_importance_from_row():
    row = pd.Series(
        {
            "competition": "FIFA World Cup",
            "stage": "Group C",
            "is_knockout": False,
        }
    )
    assert match_importance_from_row(row) == TOURNAMENT_GROUP


def _synthetic_by_importance(*, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    specs = [
        (FRIENDLY, 2.4, 40),
        (QUALIFIER, 3.2, 40),
        (TOURNAMENT_GROUP, 2.0, 40),
        (TOURNAMENT_KNOCKOUT, 1.8, 30),
    ]
    rows = []
    i = 0
    comp_map = {
        FRIENDLY: "International Friendly",
        QUALIFIER: "World Cup qualification UEFA",
        TOURNAMENT_GROUP: "FIFA World Cup",
        TOURNAMENT_KNOCKOUT: "FIFA World Cup",
    }
    stage_map = {
        FRIENDLY: None,
        QUALIFIER: "Group A",
        TOURNAMENT_GROUP: "Group B",
        TOURNAMENT_KNOCKOUT: "Quarter-final",
    }
    for cat, rate, n in specs:
        for _ in range(n):
            home, away = rng.choice(["alpha", "beta", "gamma"], size=2, replace=False)
            hg = int(rng.poisson(rate / 2))
            ag = int(rng.poisson(rate / 2))
            rows.append(
                {
                    "match_id": f"m{i:03d}",
                    "date_utc": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                    "corpus": CORPUS,
                    "competition": comp_map[cat],
                    "stage": stage_map[cat],
                    "home_team_id": home,
                    "away_team_id": away,
                    "neutral": True,
                    "is_knockout": cat == TOURNAMENT_KNOCKOUT,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result_1x2": "home" if hg > ag else ("away" if hg < ag else "draw"),
                }
            )
            i += 1
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return df


def test_importance_adjustments_lower_tournament_than_qualifier():
    train = _synthetic_by_importance(seed=7)
    strengths = fit_poisson_strengths(train, half_life_days=365)
    adj = fit_importance_adjustments(train, strengths, half_life_days=365, min_weight=5.0)
    assert adj[QUALIFIER] > adj[TOURNAMENT_GROUP]
    assert adj[TOURNAMENT_GROUP] < 0.0 or adj[TOURNAMENT_KNOCKOUT] < adj[QUALIFIER]


def test_dixon_coles_v2_lowers_tournament_lambdas_vs_v1_on_same_fixture():
    train = _synthetic_by_importance(seed=11)
    model = DixonColesV2Model(max_goals=5)
    model.fit(train)
    strengths = model._strengths
    base_lh, base_la = predict_lambdas(strengths, "alpha", "beta", neutral=True)
    shift = model._importance_adj.get(TOURNAMENT_GROUP, 0.0)
    adj_lh, adj_la = predict_lambdas(
        strengths, "alpha", "beta", neutral=True, importance_log_adj=shift
    )
    assert adj_lh + adj_la < base_lh + base_la
    assert math.exp(shift) < 1.0 or shift <= 0.0
