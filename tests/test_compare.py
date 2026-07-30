"""Leaderboard and ensemble comparison tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlvswe.eval.compare import align_predictions, build_leaderboard, pairwise_vs_reference, rps_tie_clusters
from nlvswe.eval.scoring import ranked_probability_score
from nlvswe.models.base import validate_prediction
from nlvswe.models.ensemble import (
    ENSEMBLE_WEIGHTED,
    average_probs,
    build_walkforward_weighted_predictions,
    fit_inverse_rps_weights,
    fit_inverse_rps_weights_array,
    normalize_probs,
)


def _pred_row(match_id: str, model: str, p_home: float, p_draw: float, p_away: float, outcome: str, day: int):
    return {
        "match_id": match_id,
        "model": model,
        "date_utc": pd.Timestamp(f"2024-01-{day:02d}", tz="UTC"),
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "outcome": outcome,
        "has_scoreline": False,
    }


def test_leaderboard_manual_rps():
    rows = [
        _pred_row("m1", "a", 0.6, 0.2, 0.2, "home", 1),
        _pred_row("m2", "a", 0.2, 0.2, 0.6, "away", 2),
        _pred_row("m1", "b", 0.4, 0.3, 0.3, "home", 1),
        _pred_row("m2", "b", 0.4, 0.3, 0.3, "away", 2),
    ]
    preds = pd.DataFrame(rows)
    preds["date_utc"] = pd.to_datetime(preds["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    preds["outcome"] = preds["outcome"].astype("string")

    board = build_leaderboard(preds, bootstrap_samples=500, seed=0)
    rps_a = board[(board["model"] == "a") & (board["metric"] == "rps")]["mean"].iloc[0]
    manual = np.mean(
        [
            ranked_probability_score({"home": 0.6, "draw": 0.2, "away": 0.2}, "home"),
            ranked_probability_score({"home": 0.2, "draw": 0.2, "away": 0.6}, "away"),
        ]
    )
    assert rps_a == pytest.approx(manual)


def test_ensemble_average_normalized():
    probs = average_probs(
        [
            {"home": 0.5, "draw": 0.3, "away": 0.2},
            {"home": 0.4, "draw": 0.4, "away": 0.2},
        ]
    )
    assert sum(probs.values()) == pytest.approx(1.0)
    pred = __import__("nlvswe.models.base", fromlist=["MatchPrediction"]).MatchPrediction(
        match_id="x", scoreline=None, probs_1x2=probs
    )
    validate_prediction(pred)


def test_weighted_ensemble_uses_training_slice_only():
    members = ["m1", "m2"]
    rows = []
    for i in range(1, 8):
        outcome = "home" if i % 2 else "away"
        rows.append(_pred_row(f"s{i}", "m1", 0.7, 0.2, 0.1, outcome, i))
        rows.append(_pred_row(f"s{i}", "m2", 0.34, 0.33, 0.33, outcome, i))
    stacked = pd.DataFrame(rows)
    stacked["date_utc"] = pd.to_datetime(stacked["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    stacked["outcome"] = stacked["outcome"].astype("string")

    frames = {
        "m1": stacked[stacked["model"] == "m1"].drop(columns=["model"]),
        "m2": stacked[stacked["model"] == "m2"].drop(columns=["model"]),
    }

    class Spy:
        calls: list[int] = []

        @classmethod
        def fake_fit(cls, member_probs, outcome_idx, members_list):
            cls.calls.append(int(member_probs.shape[0]))
            return fit_inverse_rps_weights_array(member_probs, outcome_idx, members_list)

    spy = Spy
    original = fit_inverse_rps_weights_array
    import nlvswe.models.ensemble as ens

    ens.fit_inverse_rps_weights_array = spy.fake_fit  # type: ignore[method-assign]
    try:
        out = build_walkforward_weighted_predictions(frames, members, stacked, min_history_matches=3)
    finally:
        ens.fit_inverse_rps_weights_array = original

    assert not out.empty
    assert len(spy.calls) >= 1
    # First scored match is s4 (index 3); its weight fit must use 3 prior rows only.
    assert spy.calls[0] == 3


def test_rps_tie_clusters_overlap():
    board = pd.DataFrame(
        [
            {"model": "a", "mean": 0.18, "ci_low": 0.17, "ci_high": 0.19},
            {"model": "b", "mean": 0.181, "ci_low": 0.179, "ci_high": 0.183},
            {"model": "c", "mean": 0.30, "ci_low": 0.28, "ci_high": 0.32},
        ]
    )
    clusters = rps_tie_clusters(board)
    flat = {m for group in clusters for m in group}
    assert flat == {"a", "b", "c"}
    assert any(set(group) == {"a", "b"} for group in clusters)


def test_align_predictions_common_set():
    a = pd.DataFrame([_pred_row("m1", "a", 0.5, 0.3, 0.2, "home", 1)])
    b = pd.DataFrame([_pred_row("m1", "b", 0.4, 0.3, 0.3, "home", 1), _pred_row("m2", "b", 0.4, 0.3, 0.3, "draw", 2)])
    for df in (a, b):
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
        df["outcome"] = df["outcome"].astype("string")
    aligned = align_predictions({"a": a.drop(columns=["model"]), "b": b.drop(columns=["model"])}, ["m1"])
    assert len(aligned) == 2
    assert set(aligned["model"]) == {"a", "b"}
