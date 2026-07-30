# Fit live models once per run, then predict every fixture in the batch

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nlvswe.config import AppConfig
from nlvswe.config import project_root
from nlvswe.eval.compare import align_predictions, common_match_ids
from nlvswe.models.base import BaseModel, MatchPrediction, validate_prediction
from nlvswe.models.ensemble import (
    ENSEMBLE_WEIGHTED,
    ensemble_members_available,
    fit_inverse_rps_weights,
    weighted_probs,
)
from nlvswe.models.registry import create_model

PROCESSED = "processed"


def _load_member_prediction_frames() -> dict[str, pd.DataFrame]:
    from nlvswe.io import read_table

    root = project_root() / "data" / PROCESSED
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("predictions_*.parquet")):
        name = path.stem.replace("predictions_", "")
        if name.startswith("ensemble"):
            continue
        try:
            frames[name] = read_table(f"predictions_{name}", PROCESSED)
        except FileNotFoundError:
            pass
    return frames


@dataclass
class LivePredictor:
    model_name: str
    cfg: AppConfig
    odds: pd.DataFrame
    as_of: pd.Timestamp
    _members: dict[str, BaseModel] = field(default_factory=dict)
    _weights: dict[str, float] = field(default_factory=dict)
    _single: BaseModel | None = None
    _scoreline_source: str | None = None

    def fit(self, train: pd.DataFrame) -> None:
        if self.model_name == ENSEMBLE_WEIGHTED:
            member_frames = _load_member_prediction_frames()
            members = ensemble_members_available(member_frames)
            if len(members) < 2:
                raise ValueError("Need at least two member models for ensemble_weighted live prediction")
            match_ids = common_match_ids({m: member_frames[m] for m in members})
            stacked = align_predictions({m: member_frames[m] for m in members}, match_ids)
            hist = stacked[pd.to_datetime(stacked["date_utc"], utc=True) < self.as_of]
            self._weights = fit_inverse_rps_weights(hist, members)
            self._members = {}
            for name in members:
                model = create_model(name, self.cfg, self.odds)
                model.fit(train)
                self._members[name] = model
            self._scoreline_source = members[0]
            return

        self._single = create_model(self.model_name, self.cfg, self.odds)
        self._single.fit(train)

    def predict(self, target_row: pd.Series) -> tuple[MatchPrediction, BaseModel | None]:
        if self.model_name == ENSEMBLE_WEIGHTED:
            parts: list[dict[str, float]] = []
            scoreline = None
            for name, model in self._members.items():
                member_pred = model.predict(target_row)
                parts.append(member_pred.probs_1x2)
                if scoreline is None and member_pred.scoreline is not None:
                    scoreline = member_pred.scoreline
            probs = weighted_probs(parts, [self._weights[n] for n in self._members])
            pred = MatchPrediction(
                match_id=str(target_row["match_id"]),
                scoreline=scoreline,
                probs_1x2=probs,
            )
            validate_prediction(pred)
            fitted = self._members.get(self._scoreline_source or "")
            return pred, fitted

        assert self._single is not None, "call fit() before predict()"
        pred = self._single.predict(target_row)
        validate_prediction(pred)
        return pred, self._single
