# Post-match settlement: score ingest, bet P&L, CLV on the frozen slip

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from nlvswe.betting.clv import clv_report
from nlvswe.betting.strategy import settle_bet
from nlvswe.config import AppConfig, get_config, load_config, project_root
from nlvswe.io import read_table, write_table
from nlvswe.logging import get_logger
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "16"
LIVE_SUBDIR = "processed/live"
INTERIM = "interim"
SCORE_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")

# Only results.parquet is written here; everything else must stay byte-identical
MUTABLE_LIVE_ARTIFACTS = frozenset({"results.parquet", "results.manifest.json"})


def parse_score(score: str) -> tuple[int, int]:
    m = SCORE_RE.match(str(score).strip())
    if not m:
        raise ValueError(f"Score must look like '5-1' or '2:0', got {score!r}")
    return int(m.group(1)), int(m.group(2))


def _result_1x2(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def live_input_paths(root: Path | None = None) -> list[Path]:
    base = (root or project_root()) / "data" / LIVE_SUBDIR
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_file())


def hash_live_inputs(root: Path | None = None) -> dict[str, str]:
    # fingerprint pre-freeze live artifacts so settlement cannot rewrite them
    out: dict[str, str] = {}
    for path in live_input_paths(root):
        if path.name in MUTABLE_LIVE_ARTIFACTS:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.name] = digest
    return out


def assert_live_inputs_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        changed = [name for name in before if before.get(name) != after.get(name)]
        added = [name for name in after if name not in before]
        removed = [name for name in before if name not in after]
        raise RuntimeError(
            "Settlement modified pre-freeze live inputs: "
            f"changed={changed} added={added} removed={removed}"
        )


def mark_match_completed(
    match_id: str,
    home_goals: int,
    away_goals: int,
    *,
    write: bool = True,
) -> pd.Series:
    matches = read_table("matches", INTERIM)
    mask = matches["match_id"].astype(str) == str(match_id)
    if not mask.any():
        raise KeyError(f"Match {match_id!r} not found in interim matches")

    idx = matches.index[mask][0]
    matches.loc[idx, "home_goals"] = int(home_goals)
    matches.loc[idx, "away_goals"] = int(away_goals)
    matches.loc[idx, "status"] = "completed"
    matches["home_goals"] = matches["home_goals"].astype("Int64")
    matches["away_goals"] = matches["away_goals"].astype("Int64")

    row = matches.loc[idx]
    if write:
        matches = validate_table(matches, "matches")
        write_table(matches, "matches", INTERIM, sort_by=["date_utc", "match_id"], plan=PLAN)
    return row


def load_score_from_canonical(match_id: str) -> tuple[int, int]:
    matches = read_table("matches", INTERIM)
    row = matches[matches["match_id"].astype(str) == str(match_id)]
    if row.empty:
        raise KeyError(f"Match {match_id!r} not found")
    rec = row.iloc[0]
    if str(rec["status"]) != "completed" or pd.isna(rec["home_goals"]) or pd.isna(rec["away_goals"]):
        raise ValueError(f"Match {match_id!r} has no completed score in canonical matches yet")
    return int(rec["home_goals"]), int(rec["away_goals"])


def settle_bet_slip(
    bet_slip: pd.DataFrame,
    *,
    home_goals: int,
    away_goals: int,
    odds: pd.DataFrame,
    devig_method: str = "multiplicative",
) -> pd.DataFrame:
    bets = bet_slip[bet_slip["status"].astype(str) == "bet"].copy()
    if bets.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, slip_row in bets.iterrows():
        market = str(slip_row["market"])
        selection = str(slip_row["selection"])
        stake = float(slip_row["stake"])
        odds_taken = float(slip_row["odds_taken"])
        bet_result, pnl = settle_bet(
            market=market,
            selection=selection,
            home_goals=home_goals,
            away_goals=away_goals,
            decimal_odds=odds_taken,
            stake=stake,
        )
        row = slip_row.to_dict()
        row.update(
            {
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "outcome_1x2": _result_1x2(home_goals, away_goals),
                "bet_result": bet_result,
                "pnl": float(pnl),
            }
        )
        rows.append(row)

    settled = pd.DataFrame(rows)
    clv = clv_report(settled, odds, devig_method=devig_method)
    merge_cols = [
        "odds_closing",
        "closing_captured_at",
        "clv_odds_ratio",
        "clv_prob_diff",
        "measurable",
        "clv_note",
    ]
    for col in merge_cols:
        settled[col] = clv[col].to_numpy()
    return _finalize_results(settled)


def _finalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["captured_at"] = pd.to_datetime(out["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    if "closing_captured_at" in out.columns:
        out["closing_captured_at"] = pd.to_datetime(out["closing_captured_at"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    out["measurable"] = out["measurable"].astype("bool")
    for col in ("home_goals", "away_goals"):
        out[col] = out[col].astype("Int64")
    return out.sort_values(["market", "selection", "bookmaker"], kind="mergesort").reset_index(drop=True)


def run_settle(
    match_id: str,
    *,
    score: str | None = None,
    refresh: bool = False,
    cfg: AppConfig | None = None,
    write_match: bool = True,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    before_hashes = hash_live_inputs()

    if refresh:
        if score is not None:
            raise ValueError("Pass either --score or --refresh, not both")
        home_goals, away_goals = load_score_from_canonical(match_id)
    else:
        if score is None:
            raise ValueError("Provide --score HOME-AWAY or use --refresh")
        home_goals, away_goals = parse_score(score)
        mark_match_completed(match_id, home_goals, away_goals, write=write_match)

    bet_slip = read_table("bet_slip", LIVE_SUBDIR)
    slip_rows = bet_slip[bet_slip["match_id"].astype(str) == str(match_id)]
    if slip_rows.empty:
        raise ValueError(f"No bet_slip rows for match_id={match_id!r}")

    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()

    results = settle_bet_slip(
        slip_rows,
        home_goals=home_goals,
        away_goals=away_goals,
        odds=odds,
        devig_method=cfg.betting.devig_method,
    )
    if results.empty:
        logger.warning("No bet rows to settle for %s", match_id)
        return results

    results = validate_table(results, "live_results")
    write_table(
        results,
        "results",
        LIVE_SUBDIR,
        sort_by=["market", "selection", "bookmaker"],
        plan=PLAN,
        schema_name="live_results",
    )

    after_hashes = hash_live_inputs()
    assert_live_inputs_unchanged(before_hashes, after_hashes)

    total_pnl = float(results["pnl"].sum())
    logger.info(
        "Settled %d bets for %s (%d-%d); total P&L=%.2f",
        len(results),
        match_id,
        home_goals,
        away_goals,
        total_pnl,
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Settle the frozen live bet slip after a result")
    parser.add_argument("--match", required=True, help="match_id to settle")
    parser.add_argument("--score", default=None, help="Final score as HOME-AWAY, e.g. 5-1")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Read the score from canonical matches (must already be completed)",
    )
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    run_settle(args.match, score=args.score, refresh=args.refresh, cfg=cfg)


if __name__ == "__main__":
    main()
