# Closing-line value: did I beat the market's final price?

from __future__ import annotations

import numpy as np
import pandas as pd

from nlvswe.eval.market import devig

NOT_MEASURABLE_NOTE = "CLV not measurable: single snapshot"


def clv_odds_ratio(odds_taken: float, odds_closing: float) -> float:
    # Relative CLV from odds: taken/closing - 1 (positive = beat the close)
    if odds_closing <= 0:
        return float("nan")
    return float(odds_taken / odds_closing - 1.0)


def clv_prob_diff(
    model_prob: float,
    closing_odds: float,
    *,
    devig_method: str = "multiplicative",
    all_closing_odds: list[float] | None = None,
) -> float:
    # CLV as model prob minus de-vigged closing implied prob for the selection
    if all_closing_odds is not None and len(all_closing_odds) >= 2:
        probs = devig(np.asarray(all_closing_odds, dtype=float), method=devig_method)  # type: ignore[arg-type]
        idx = all_closing_odds.index(closing_odds) if closing_odds in all_closing_odds else 0
        closing_prob = float(probs[idx])
    else:
        closing_prob = 1.0 / closing_odds if closing_odds > 1.0 else float("nan")
    return float(model_prob - closing_prob)


def _ts_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def _quote_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row["match_id"]),
        str(row["bookmaker"]).lower(),
        str(row["market"]).lower(),
        str(row["selection"]).lower(),
    )


def _closing_lookup(odds: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.Series]:
    closing = odds[odds["is_closing"]].copy()
    out: dict[tuple[str, str, str, str], pd.Series] = {}
    for _, row in closing.iterrows():
        out[_quote_key(row)] = row
    return out


def _snapshot_count(odds: pd.DataFrame, key: tuple[str, str, str, str]) -> int:
    mid, bk, market, sel = key
    sub = odds[
        (odds["match_id"].astype(str) == mid)
        & (odds["bookmaker"].str.lower() == bk)
        & (odds["market"].astype(str).str.lower() == market)
        & (odds["selection"].astype(str).str.lower() == sel)
    ]
    if sub.empty:
        return 0
    return int(sub["captured_at"].nunique())


def _closing_1x2_odds(
    odds: pd.DataFrame,
    match_id: str,
    bookmaker: str,
) -> list[float] | None:
    sub = odds[
        (odds["is_closing"])
        & (odds["match_id"].astype(str) == str(match_id))
        & (odds["bookmaker"].str.lower() == str(bookmaker).lower())
        & (odds["market"].astype(str).str.lower().isin({"1x2", "h2h"}))
    ]
    if sub.empty:
        return None
    prices: list[float] = []
    for sel in ("home", "draw", "away"):
        row = sub[sub["selection"].astype(str).str.lower() == sel]
        if row.empty:
            return None
        prices.append(float(row.iloc[0]["decimal_odds"]))
    return prices


def clv_report(
    bets: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    devig_method: str = "multiplicative",
) -> pd.DataFrame:
    # Per-bet CLV with an explicit measurable flag
    #
    # Requires captured_at on the bet (which snapshot was taken) and multiple
    # pre-kickoff snapshots in odds so taken != closing timestamp
    if bets.empty:
        return bets.copy()

    closing_map = _closing_lookup(odds)
    rows: list[dict] = []

    for _, bet in bets.iterrows():
        row = bet.to_dict()
        key = _quote_key(bet)
        closing_row = closing_map.get(key)
        taken_at = _ts_utc(bet["captured_at"]) if pd.notna(bet.get("captured_at")) else None

        measurable = False
        note = NOT_MEASURABLE_NOTE
        odds_closing = float("nan")
        closing_at = pd.NaT
        clv_or = float("nan")
        clv_pd = float("nan")

        n_snaps = _snapshot_count(odds, key)
        if closing_row is None or taken_at is None:
            note = "CLV not measurable: missing closing line or captured_at"
        elif n_snaps < 2:
            note = NOT_MEASURABLE_NOTE
        else:
            closing_at = _ts_utc(closing_row["captured_at"])
            odds_closing = float(closing_row["decimal_odds"])
            if taken_at == closing_at:
                note = NOT_MEASURABLE_NOTE
            else:
                measurable = True
                note = ""
                odds_taken = float(bet["odds_taken"])
                clv_or = clv_odds_ratio(odds_taken, odds_closing)
                model_prob = float(bet["model_prob"])
                all_close = _closing_1x2_odds(odds, key[0], key[1])
                clv_pd = clv_prob_diff(
                    model_prob,
                    odds_closing,
                    devig_method=devig_method,
                    all_closing_odds=all_close,
                )

        row.update(
            {
                "odds_closing": odds_closing,
                "closing_captured_at": closing_at,
                "clv_odds_ratio": clv_or,
                "clv_prob_diff": clv_pd,
                "measurable": measurable,
                "clv_note": note,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if "closing_captured_at" in out.columns:
        out["closing_captured_at"] = pd.to_datetime(out["closing_captured_at"], utc=True)
    return out


def aggregate_clv(bets: pd.DataFrame) -> dict[str, float]:
    # Summary CLV statistics for a bet ledger (measurable rows only when flagged)
    if bets.empty:
        return {
            "mean_clv_odds_ratio": float("nan"),
            "median_clv_odds_ratio": float("nan"),
            "mean_clv_prob_diff": float("nan"),
            "pct_positive_clv": float("nan"),
            "n_with_clv": 0,
        }
    subset = bets[bets["measurable"]] if "measurable" in bets.columns else bets
    clv = subset["clv_odds_ratio"].dropna()
    cpd = subset["clv_prob_diff"].dropna()
    return {
        "mean_clv_odds_ratio": float(clv.mean()) if not clv.empty else float("nan"),
        "median_clv_odds_ratio": float(clv.median()) if not clv.empty else float("nan"),
        "mean_clv_prob_diff": float(cpd.mean()) if not cpd.empty else float("nan"),
        "pct_positive_clv": float((clv > 0).mean()) if not clv.empty else float("nan"),
        "n_with_clv": int(clv.shape[0]),
    }
