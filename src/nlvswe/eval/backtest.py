"""Walk-forward backtest: each match is predicted using only prior history."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from nlvswe.models.base import BaseModel, MatchPrediction

CvWindow = Literal["expanding", "rolling"]


def prediction_to_row(pred: MatchPrediction, *, model: str, match: pd.Series) -> dict:
    row = {
        "match_id": pred.match_id,
        "model": model,
        "date_utc": pd.Timestamp(match["date_utc"]).tz_convert("UTC"),
        "p_home": pred.probs_1x2["home"],
        "p_draw": pred.probs_1x2["draw"],
        "p_away": pred.probs_1x2["away"],
        "outcome": match.get("result_1x2"),
        "has_scoreline": pred.scoreline is not None,
    }
    if pred.scoreline is not None:
        row["scoreline_flat"] = pred.scoreline.ravel().tolist()
    return row


def walk_forward_backtest(
    model: BaseModel,
    features: pd.DataFrame,
    *,
    min_history_matches: int = 100,
    cv_window: CvWindow = "expanding",
    rolling_matches: int | None = None,
    refit_every: int = 1,
    corpus: str = "international",
) -> pd.DataFrame:
    """Walk-forward predictions: train strictly on prior matches only."""
    subset = features[features["corpus"] == corpus].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    scored = subset[subset["result_1x2"].notna()].reset_index(drop=True)
    rows: list[dict] = []
    fitted = False
    matches_since_fit = refit_every

    for i in range(len(scored)):
        test = scored.iloc[i]
        train = scored.iloc[:i]
        if len(train) < min_history_matches:
            continue
        if cv_window == "rolling" and rolling_matches is not None:
            train = train.tail(rolling_matches)

        if not fitted or matches_since_fit >= refit_every:
            model.fit(train)
            fitted = True
            matches_since_fit = 0

        try:
            pred = model.predict(test)
        except ValueError:
            matches_since_fit += 1
            continue
        rows.append(prediction_to_row(pred, model=model.name, match=test))
        matches_since_fit += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    out["outcome"] = out["outcome"].astype("string")
    out["has_scoreline"] = out["has_scoreline"].astype("bool")
    return out.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(drop=True)
