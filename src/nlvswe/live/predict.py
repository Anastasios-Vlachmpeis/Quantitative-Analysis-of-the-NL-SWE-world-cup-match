"""Pre-match pipeline: one or many fixtures, shared model fit, optional git freeze."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.betting.induction import induce_row_markets, simulate_scorelines
from nlvswe.betting.slip import build_live_bet_slip
from nlvswe.config import AppConfig, TargetMatch, get_config, load_config, project_root
from nlvswe.eval.backtest import prediction_to_row
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.live.fixtures import (
    fixture_kickoff,
    load_live_fixtures,
    matchday_tag,
    resolve_fixture_match,
)
from nlvswe.live.predictor import LivePredictor
from nlvswe.logging import get_logger
from nlvswe.models.base import MatchPrediction
from nlvswe.models.data import load_model_features
from nlvswe.models.ensemble import ENSEMBLE_WEIGHTED
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import git_commit, set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "16"
LIVE_SUBDIR = "processed/live"
INTERIM = "interim"
PROCESSED = "processed"
REPORTS = "reports"
CORPUS_INTERNATIONAL = "international"


@dataclass
class FixtureRunResult:
    fixture: TargetMatch
    match_id: str
    kickoff: pd.Timestamp
    features: pd.DataFrame
    pred: MatchPrediction
    market_probs: pd.DataFrame
    bet_slip: pd.DataFrame


def selected_model_name() -> str:
    path = project_root() / REPORTS / "MODEL_SELECTION.md"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("**Selected model:**"):
                return line.split("`")[1]
    return ENSEMBLE_WEIGHTED


def kickoff_utc(cfg: AppConfig) -> pd.Timestamp:
    return fixture_kickoff(load_live_fixtures(cfg)[0])


def resolve_target_match(matches: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    return resolve_fixture_match(matches, load_live_fixtures(cfg)[0])


def refresh_live_inputs(cfg: AppConfig) -> None:
    """Refresh FIFA, results, and live odds snapshots before I lock the pre-match call."""
    try:
        from nlvswe.data import acquire, clean
    except ImportError as exc:
        logger.warning("Data refresh skipped (acquire/clean unavailable): %s", exc)
        return

    logger.info("Refreshing results and FIFA ranking raw data")
    acquire.main(["--source", "results"])
    acquire.main(["--source", "fifa_rank"])
    try:
        acquire.main(["--source", "odds_live"])
    except Exception as exc:
        logger.warning("Live odds snapshot not refreshed: %s", exc)

    logger.info("Rebuilding interim matches, ratings, odds")
    clean.main(["--table", "matches"])
    clean.main(["--table", "ratings"])
    clean.main(["--table", "odds"])


def build_live_features_for_fixture(
    fixture: TargetMatch,
    cfg: AppConfig,
    *,
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    conditions: pd.DataFrame,
    venues: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    from nlvswe.features.build import _prepare_features, build_features, features_for_match

    target = resolve_fixture_match(matches, fixture)
    kickoff = fixture_kickoff(fixture)

    intl = matches[matches["corpus"] == CORPUS_INTERNATIONAL].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    feat = features_for_match(
        target,
        intl,
        ratings,
        conditions,
        venues,
        form_windows=list(cfg.model.form_windows),
        fifa_max_age_days=cfg.data.fifa_max_age_days,
    )
    feat["date_utc"] = kickoff

    batch = build_features(
        intl,
        ratings,
        conditions,
        venues,
        form_windows=list(cfg.model.form_windows),
        corpus=CORPUS_INTERNATIONAL,
        fifa_max_age_days=cfg.data.fifa_max_age_days,
    )
    batch_row = batch[batch["match_id"].astype(str) == str(target["match_id"])]
    if not batch_row.empty:
        br = batch_row.iloc[0]
        skip = {"result_1x2", "home_goals", "away_goals", "total_goals"}
        for col in batch.columns:
            if col in skip:
                continue
            bv, lv = br[col], feat.get(col)
            if pd.isna(bv) and pd.isna(lv):
                continue
            if bv != lv:
                logger.warning("Live/batch feature mismatch on %s: batch=%s live=%s", col, bv, lv)

    df = _prepare_features(pd.DataFrame([feat]))
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return validate_table(df, "features"), target


def build_live_features_row(cfg: AppConfig) -> tuple[pd.DataFrame, pd.Series]:
    matches = read_table("matches", INTERIM)
    ratings = read_table("ratings", INTERIM)
    conditions = read_table("conditions", INTERIM)
    venues = read_table("venues", INTERIM)
    fixture = load_live_fixtures(cfg)[0]
    return build_live_features_for_fixture(
        fixture, cfg, matches=matches, ratings=ratings, conditions=conditions, venues=venues
    )


def induce_live_markets(
    pred: MatchPrediction,
    model_name: str,
    *,
    model: object | None = None,
) -> pd.DataFrame:
    calibrators = None
    if model is not None:
        getter = getattr(model, "totals_calibrators", None)
        if callable(getter):
            calibrators = getter() or None
    rows = induce_row_markets(
        match_id=pred.match_id,
        model=model_name,
        p_home=pred.probs_1x2["home"],
        p_draw=pred.probs_1x2["draw"],
        p_away=pred.probs_1x2["away"],
        scoreline=pred.scoreline,
        totals_calibrators=calibrators,
    )
    df = pd.DataFrame(rows)
    df["model_prob"] = df["model_prob"].astype("float64")
    return validate_table(df, "market_probs")


def plot_live_figures(pred: MatchPrediction, model_name: str, cfg: AppConfig) -> None:
    if pred.scoreline is None:
        logger.warning("No scoreline matrix; skipping live scoreline/sim figures")
        return
    apply_theme()
    mat = np.asarray(pred.scoreline, dtype=float)
    side = min(8, mat.shape[0])
    sub = mat[:side, :side]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sub, cmap="Blues", origin="lower")
    fig.colorbar(im, ax=ax, label="P(score)")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(f"Live scoreline — {model_name}")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "live_scoreline")
    plt.close(fig)

    rng = np.random.default_rng(cfg.seed)
    sims = simulate_scorelines(mat, min(cfg.model.mc_samples, 5000), rng)
    home, away = sims[:, 0], sims[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(home + away, bins=20, color="#00B0FF", edgecolor="#1E293B")
    axes[0].set_title("Simulated total goals")
    axes[1].hist(home - away, bins=20, color="#00E676", edgecolor="#1E293B")
    axes[1].set_title("Simulated goal difference")
    fig.suptitle(f"Live Monte Carlo — {model_name}")
    style_figure(fig)
    for ax in axes:
        style_axes(ax)
    save_figure(fig, "live_sim_scores")
    plt.close(fig)


def plot_live_edges(bet_slip: pd.DataFrame) -> None:
    bets = bet_slip[bet_slip["status"] == "bet"]
    if bets.empty:
        return
    apply_theme()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(bets))))
    labels = [f"{r.market}:{r.selection}@{r.bookmaker}" for r in bets.itertuples()]
    y = np.arange(len(bets))
    ax.barh(y, bets["ev"], color="#00E676", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(get_config().betting.min_edge, color="#64748B", linestyle="--", linewidth=1)
    ax.set_xlabel("Expected value (EV)")
    ax.set_title("Live +EV selections")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "live_edges")
    plt.close(fig)


def write_predictions_md(
    *,
    cfg: AppConfig,
    model_name: str,
    results: list[FixtureRunResult],
    commit_hash: str,
    path: Path,
) -> None:
    title = "Pre-match predictions"
    if len(results) == 1:
        fx = results[0].fixture
        title = f"Pre-match prediction — {fx.home} vs {fx.away}"

    lines = [
        f"# {title}",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"**Git commit:** `{commit_hash}`",
        f"**Model:** `{model_name}`",
        f"**Fixtures priced:** {len(results)}",
        "",
    ]

    for run in results:
        fx = run.fixture
        pred = run.pred
        features = run.features
        bet_slip = run.bet_slip
        lines.extend(
            [
                f"## {fx.home} vs {fx.away} (`{run.match_id}`)",
                "",
                f"**Kickoff (UTC):** {run.kickoff.isoformat()}",
                "",
                "### 1X2 probabilities",
                "",
                f"| Home ({fx.home}) | Draw | Away ({fx.away}) |",
                "|---|---|---|",
                f"| {pred.probs_1x2['home']:.1%} | {pred.probs_1x2['draw']:.1%} | {pred.probs_1x2['away']:.1%} |",
                "",
                "### Key features (pre-kickoff)",
                "",
            ]
        )
        row = features.iloc[0]
        for col in ("home_elo_pre", "away_elo_pre", "elo_diff", "fifa_points_diff", "form_ppg_diff_5", "neutral"):
            if col in row.index:
                lines.append(f"- **{col}:** {row[col]}")

        lines.extend(["", "### Bet slip", ""])
        bets = bet_slip[bet_slip["status"] == "bet"]
        if bets.empty:
            reason = bet_slip.iloc[0]["rationale"]
            lines.append(f"**No bet.** {reason}")
        else:
            lines.append("| Book | Market | Selection | Odds | Model p | De-vig p | Edge | EV | Stake |")
            lines.append("|------|--------|-----------|------|---------|----------|------|-----|-------|")
            for _, b in bets.iterrows():
                lines.append(
                    f"| {b['bookmaker']} | {b['market']} | {b['selection']} | {b['odds_taken']:.2f} | "
                    f"{b['model_prob']:.3f} | {b['book_prob_devig']:.3f} | {b['edge']:.3f} | "
                    f"{b['ev']:.3f} | ${b['stake']:.2f} |"
                )
            lines.extend(["", "#### Rationale", ""])
            for _, b in bets.iterrows():
                lines.append(f"- {b['rationale']}")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Predictions use only information available before kickoff.",
            "- Per-fixture slips also live under `data/processed/live/bet_slip_<match_id>.parquet`.",
            "- Closing odds captured after this run feed CLV once results are settled.",
            "- A positive EV bet is a recommendation on a mock bankroll, not financial advice.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prediction_md(
    *,
    cfg: AppConfig,
    model_name: str,
    features: pd.DataFrame,
    pred: MatchPrediction,
    market_probs: pd.DataFrame,
    bet_slip: pd.DataFrame,
    commit_hash: str,
    path: Path,
) -> None:
    fixture = load_live_fixtures(cfg)[0]
    write_predictions_md(
        cfg=cfg,
        model_name=model_name,
        results=[
            FixtureRunResult(
                fixture=fixture,
                match_id=str(pred.match_id),
                kickoff=fixture_kickoff(fixture),
                features=features,
                pred=pred,
                market_probs=market_probs,
                bet_slip=bet_slip,
            )
        ],
        commit_hash=commit_hash,
        path=path,
    )


def create_prematch_freeze(
    cfg: AppConfig,
    *,
    kickoff: pd.Timestamp,
    message: str | None = None,
) -> str:
    """Commit live artifacts and tag prematch-freeze-<YYYYMMDD> (only before kickoff)."""
    root = project_root()
    fx = load_live_fixtures(cfg)[0]
    msg = message or f"Pre-match freeze: {fx.home} vs {fx.away}"
    paths = [
        "data/processed/live",
        "reports/PREDICTION.md",
        "reports/figures/live_scoreline.png",
        "reports/figures/live_sim_scores.png",
        "reports/figures/live_edges.png",
    ]
    for rel in paths:
        p = root / rel
        if p.exists():
            subprocess.run(["git", "add", rel], cwd=root, check=True, timeout=30)
    subprocess.run(["git", "commit", "-m", msg], cwd=root, check=True, timeout=60)
    commit = git_commit()
    tag = matchday_tag(kickoff)
    subprocess.run(["git", "tag", "-f", tag, commit], cwd=root, check=True, timeout=30)
    logger.info("Created tag %s at %s", tag, commit)
    return commit


def run_live_predict(
    cfg: AppConfig | None = None,
    *,
    model_name: str | None = None,
    refresh: bool = False,
    freeze: bool = False,
) -> dict:
    cfg = cfg or get_config()
    if cfg.scope != "live":
        logger.warning("config scope=%s (expected live for production freeze)", cfg.scope)

    fixtures = load_live_fixtures(cfg)
    model_name = model_name or selected_model_name()
    kickoffs = [fixture_kickoff(fx) for fx in fixtures]
    as_of = min(kickoffs)
    now = pd.Timestamp(datetime.now(timezone.utc))
    if now >= as_of:
        logger.warning("Current time is at or after earliest kickoff — dry-run/recovery mode")

    if refresh:
        refresh_live_inputs(cfg)

    matches = read_table("matches", INTERIM)
    ratings = read_table("ratings", INTERIM)
    conditions = read_table("conditions", INTERIM)
    venues = read_table("venues", INTERIM)

    train = load_model_features()
    train = train[train["result_1x2"].notna()]
    train = train[pd.to_datetime(train["date_utc"], utc=True) < as_of].reset_index(drop=True)

    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()

    predictor = LivePredictor(model_name, cfg, odds, as_of=as_of)
    predictor.fit(train)

    run_results: list[FixtureRunResult] = []
    feature_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    market_frames: list[pd.DataFrame] = []
    slip_frames: list[pd.DataFrame] = []

    for fixture in fixtures:
        kickoff = fixture_kickoff(fixture)
        features, target = build_live_features_for_fixture(
            fixture, cfg, matches=matches, ratings=ratings, conditions=conditions, venues=venues
        )
        target_feat = features.iloc[0]
        pred, fitted = predictor.predict(target_feat)
        pred_row = prediction_to_row(pred, model=model_name, match=target_feat)
        preds_df = pd.DataFrame([pred_row]).drop(columns=["outcome"], errors="ignore")
        preds_df["date_utc"] = pd.to_datetime(preds_df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
        if "has_scoreline" in preds_df.columns:
            preds_df["has_scoreline"] = preds_df["has_scoreline"].astype("bool")
        preds_df = validate_table(preds_df, "live_predictions")

        market_probs = induce_live_markets(pred, model_name, model=fitted)
        bet_slip = build_live_bet_slip(
            market_probs,
            odds,
            cfg,
            match_id=str(target["match_id"]),
            model=model_name,
            kickoff=kickoff,
        )
        bet_slip = validate_table(bet_slip, "bet_slip")

        match_id = str(target["match_id"])
        write_table(
            bet_slip,
            f"bet_slip_{match_id}",
            LIVE_SUBDIR,
            sort_by=["status", "ev"],
            plan=PLAN,
            schema_name="bet_slip",
        )

        run_results.append(
            FixtureRunResult(
                fixture=fixture,
                match_id=match_id,
                kickoff=kickoff,
                features=features,
                pred=pred,
                market_probs=market_probs,
                bet_slip=bet_slip,
            )
        )
        feature_frames.append(features)
        pred_frames.append(preds_df)
        market_frames.append(market_probs)
        slip_frames.append(bet_slip)

    all_features = pd.concat(feature_frames, ignore_index=True)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_markets = pd.concat(market_frames, ignore_index=True)
    all_slips = pd.concat(slip_frames, ignore_index=True)

    write_table(all_features, "features", LIVE_SUBDIR, sort_by=["match_id"], plan=PLAN, schema_name="features")
    write_table(all_preds, "predictions", LIVE_SUBDIR, sort_by=["match_id"], plan=PLAN, schema_name="live_predictions")
    write_table(
        all_markets,
        "market_probs",
        LIVE_SUBDIR,
        sort_by=["match_id", "market", "selection"],
        plan=PLAN,
        schema_name="market_probs",
    )
    write_table(all_slips, "bet_slip", LIVE_SUBDIR, sort_by=["match_id", "status", "ev"], plan=PLAN, schema_name="bet_slip")

    if run_results:
        plot_live_figures(run_results[0].pred, model_name, cfg)
        plot_live_edges(all_slips)

    commit_hash = git_commit()
    write_predictions_md(
        cfg=cfg,
        model_name=model_name,
        results=run_results,
        commit_hash=commit_hash,
        path=project_root() / REPORTS / "PREDICTION.md",
    )

    if freeze:
        for kickoff in sorted(set(kickoffs)):
            day_fixtures = [r for r in run_results if r.kickoff == kickoff]
            label = ", ".join(f"{r.fixture.home} v {r.fixture.away}" for r in day_fixtures)
            commit_hash = create_prematch_freeze(cfg, kickoff=kickoff, message=f"Pre-match freeze: {label}")
        write_predictions_md(
            cfg=cfg,
            model_name=model_name,
            results=run_results,
            commit_hash=commit_hash,
            path=project_root() / REPORTS / "PREDICTION.md",
        )

    n_bets = int((all_slips["status"] == "bet").sum()) if not all_slips.empty else 0
    logger.info(
        "Live prediction complete; model=%s fixtures=%d bets=%d",
        model_name,
        len(run_results),
        n_bets,
    )
    return {
        "model": model_name,
        "match_ids": [r.match_id for r in run_results],
        "commit_hash": commit_hash,
        "n_bets": n_bets,
        "kickoff_utc": as_of.isoformat(),
        "n_fixtures": len(run_results),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pre-match prediction for configured fixtures, optional git freeze")
    parser.add_argument("--model", default=None, help="Model (default: read from MODEL_SELECTION.md)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch raw data and rebuild interim tables")
    parser.add_argument("--freeze", action="store_true", help="Commit artifacts and tag prematch-freeze-<date>")
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    result = run_live_predict(cfg, model_name=args.model, refresh=args.refresh, freeze=args.freeze)
    print(
        f"Live predict done: model={result['model']} fixtures={result['n_fixtures']} "
        f"commit={result['commit_hash']} bets={result['n_bets']}"
    )


if __name__ == "__main__":
    main()
