# Totals dispersion and isotonic recalibration tests

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlvswe.betting.induction import default_market_config, markets_from_matrix
from nlvswe.models.dixon_coles_v2 import DixonColesV2Model
from nlvswe.models.totals_calibration import (
    TotalsIsotonicCalibrator,
    assess_total_goals_dispersion,
    prob_over_from_scoreline,
)

CORPUS = "international"


def test_prob_over_from_scoreline():
    mat = np.zeros((4, 4))
    mat[1, 1] = 0.5
    mat[0, 1] = 0.5
    assert prob_over_from_scoreline(mat, 1.5) == pytest.approx(0.5)


def test_dispersion_flags_underdispersed_competitive_totals():
    rows = []
    for i in range(40):
        rows.append(
            {
                "match_id": f"t{i}",
                "competition": "FIFA World Cup",
                "stage": "Group A",
                "is_knockout": False,
                "home_goals": 2 if i % 2 == 0 else 3,
                "away_goals": 0,
                "result_1x2": "home",
            }
        )
    train = pd.DataFrame(rows)

    def expected_total(_row: pd.Series) -> float:
        return 4.5

    report = assess_total_goals_dispersion(train, expected_total, competitive_only=True)
    assert report["underdispersed"] is True
    assert float(report["dispersion_ratio"]) < 1.0


def test_isotonic_calibrator_pulls_overprob_down():
    raw = np.concatenate([np.full(20, 0.72), np.full(20, 0.82)])
    outcomes = np.concatenate([np.ones(20), np.zeros(20)])
    cal = TotalsIsotonicCalibrator(line=2.5)
    assert cal.fit(raw, outcomes)
    assert cal.calibrate(0.82) < 0.82


def test_markets_from_matrix_applies_totals_calibrator():
    mat = np.outer([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    raw_over = prob_over_from_scoreline(mat, 2.5)

    def shrink(p: float) -> float:
        return p * 0.7

    cfg = default_market_config()
    if 2.5 not in cfg.totals_lines:
        pytest.skip("totals_lines empty in config")

    rows = markets_from_matrix(mat, totals_calibrators={2.5: shrink})
    over = next(r for r in rows if r["market"] == "totals_2.5" and r["selection"] == "over")
    assert over["model_prob"] == pytest.approx(raw_over * 0.7, rel=1e-6)


def test_v2_fits_isotonic_when_competitive_totals_are_underdispersed():
    rng = np.random.default_rng(9)
    rows = []
    for i in range(50):
        total = 2 if i % 2 == 0 else 3
        split = rng.integers(0, total + 1)
        rows.append(
            {
                "match_id": f"c{i}",
                "date_utc": pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=i),
                "corpus": CORPUS,
                "competition": "FIFA World Cup",
                "stage": "Group B",
                "home_team_id": "alpha",
                "away_team_id": "beta",
                "neutral": True,
                "is_knockout": False,
                "home_goals": split,
                "away_goals": total - split,
                "result_1x2": "draw" if split * 2 == total else "home",
            }
        )
    for i in range(20):
        hg = int(rng.poisson(1.2))
        ag = int(rng.poisson(1.2))
        rows.append(
            {
                "match_id": f"f{i}",
                "date_utc": pd.Timestamp("2021-03-01", tz="UTC") + pd.Timedelta(days=i),
                "corpus": CORPUS,
                "competition": "International Friendly",
                "stage": None,
                "home_team_id": "alpha",
                "away_team_id": "gamma",
                "neutral": True,
                "is_knockout": False,
                "home_goals": hg,
                "away_goals": ag,
                "result_1x2": "home" if hg > ag else ("away" if hg < ag else "draw"),
            }
        )
    train = pd.DataFrame(rows)
    train["date_utc"] = pd.to_datetime(train["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    model = DixonColesV2Model(max_goals=5, totals_lines=(2.5,))
    model.fit(train)
    assert model.dispersion_report()["underdispersed"] is True
    assert 2.5 in model.totals_calibrators()
