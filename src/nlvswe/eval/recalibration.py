# Plan 14 acceptance: totals reliability, draw calibration, NL-SWE spot anchor

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.config import AppConfig, get_config, load_config, project_root
from nlvswe.eval.backtest import prediction_to_row
from nlvswe.eval.scoring import bootstrap_ci, score_predictions
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.models.base import BaseModel
from nlvswe.models.data import load_model_features
from nlvswe.models.registry import create_model
from nlvswe.models.totals_calibration import prob_over_from_scoreline
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "14"
PROCESSED = "processed"
INTERIM = "interim"
REPORTS = "reports"
FIGURES = "figures"
DEFAULT_TOTALS_LINE = 2.5
NL_SWE_MARKET_OVER_25 = 0.56


def binary_ece(probs: np.ndarray, outcomes: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    err = 0.0
    weight = 0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        if b < bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        pred = float(probs[mask].mean())
        obs = float(outcomes[mask].mean())
        n = int(mask.sum())
        err += abs(pred - obs) * n
        weight += n
    return float(err / weight) if weight else float("nan")


def totals_reliability_bins(probs: np.ndarray, outcomes: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        if b < bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_mid": 0.5 * (lo + hi),
                "mean_pred": float(probs[mask].mean()),
                "obs_freq": float(outcomes[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_combined(
    models: list[BaseModel],
    features: pd.DataFrame,
    *,
    line: float = DEFAULT_TOTALS_LINE,
    min_history_matches: int = 100,
    refit_every: int = 10,
    corpus: str = "international",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    # One chronological pass: fit once per refit window, collect 1X2 preds and totals rows
    subset = features[features["corpus"] == corpus].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    scored = subset[subset["result_1x2"].notna()].reset_index(drop=True)
    pred_rows: dict[str, list[dict]] = {m.name: [] for m in models}
    total_rows: dict[str, list[dict]] = {m.name: [] for m in models}
    fitted = False
    matches_since_fit = refit_every

    for i in range(len(scored)):
        test = scored.iloc[i]
        train = scored.iloc[:i]
        if len(train) < min_history_matches:
            continue
        if not fitted or matches_since_fit >= refit_every:
            for model in models:
                model.fit(train)
            fitted = True
            matches_since_fit = 0

        total = int(test["home_goals"]) + int(test["away_goals"])
        for model in models:
            try:
                pred = model.predict(test)
            except ValueError:
                continue
            pred_rows[model.name].append(prediction_to_row(pred, model=model.name, match=test))
            if pred.scoreline is None:
                continue
            raw_over = prob_over_from_scoreline(pred.scoreline, line)
            cal_over = raw_over
            getter = getattr(model, "totals_calibrators", None)
            if callable(getter):
                cals = getter()
                if line in cals:
                    cal_over = float(np.clip(cals[line](raw_over), 0.0, 1.0))
            total_rows[model.name].append(
                {
                    "match_id": str(test["match_id"]),
                    "date_utc": pd.Timestamp(test["date_utc"]).tz_convert("UTC"),
                    "model": model.name,
                    "line": line,
                    "p_over_raw": raw_over,
                    "p_over": cal_over,
                    "over_hit": float(total > line),
                    "p_draw": pred.probs_1x2["draw"],
                    "draw_hit": float(test["result_1x2"] == "draw"),
                }
            )
        matches_since_fit += 1
        if i > 0 and i % 500 == 0:
            logger.info("Walk-forward progress: %d / %d matches", i, len(scored))
            print(f"Walk-forward progress: {i} / {len(scored)}", flush=True)

    pred_frames: dict[str, pd.DataFrame] = {}
    total_frames: dict[str, pd.DataFrame] = {}
    for model in models:
        if pred_rows[model.name]:
            out = pd.DataFrame(pred_rows[model.name])
            out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
            out["outcome"] = out["outcome"].astype("string")
            out["has_scoreline"] = out["has_scoreline"].astype("bool")
            pred_frames[model.name] = out.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(
                drop=True
            )
        else:
            pred_frames[model.name] = pd.DataFrame()
        if total_rows[model.name]:
            tout = pd.DataFrame(total_rows[model.name])
            tout["date_utc"] = pd.to_datetime(tout["date_utc"], utc=True).astype("datetime64[ns, UTC]")
            total_frames[model.name] = tout.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(
                drop=True
            )
        else:
            total_frames[model.name] = pd.DataFrame()
    return pred_frames, total_frames


def walk_forward_totals_records(
    model: BaseModel,
    features: pd.DataFrame,
    *,
    line: float = DEFAULT_TOTALS_LINE,
    min_history_matches: int = 100,
    refit_every: int = 10,
    corpus: str = "international",
) -> pd.DataFrame:
    # Walk-forward P(over) with v2 isotonic applied when the fitted model exposes it
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
        if not fitted or matches_since_fit >= refit_every:
            model.fit(train)
            fitted = True
            matches_since_fit = 0

        try:
            pred = model.predict(test)
        except ValueError:
            matches_since_fit += 1
            continue
        if pred.scoreline is None:
            matches_since_fit += 1
            continue

        raw_over = prob_over_from_scoreline(pred.scoreline, line)
        cal_over = raw_over
        getter = getattr(model, "totals_calibrators", None)
        if callable(getter):
            cals = getter()
            if line in cals:
                cal_over = float(np.clip(cals[line](raw_over), 0.0, 1.0))

        total = int(test["home_goals"]) + int(test["away_goals"])
        rows.append(
            {
                "match_id": str(test["match_id"]),
                "date_utc": pd.Timestamp(test["date_utc"]).tz_convert("UTC"),
                "model": model.name,
                "line": line,
                "p_over_raw": raw_over,
                "p_over": cal_over,
                "over_hit": float(total > line),
                "p_draw": pred.probs_1x2["draw"],
                "draw_hit": float(test["result_1x2"] == "draw"),
            }
        )
        matches_since_fit += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return out.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(drop=True)


def draw_calibration_summary(preds: pd.DataFrame) -> dict[str, float]:
    sub = preds[preds["outcome"].notna()].copy()
    if sub.empty:
        return {"pred_draw": float("nan"), "actual_draw": float("nan"), "gap_pp": float("nan")}
    pred_draw = float(sub["p_draw"].mean())
    actual_draw = float((sub["outcome"].astype(str) == "draw").mean())
    return {
        "pred_draw": pred_draw,
        "actual_draw": actual_draw,
        "gap_pp": (pred_draw - actual_draw) * 100.0,
    }


def plot_totals_reliability(
    series: dict[str, pd.DataFrame],
    *,
    line: float,
    bins: int = 10,
    out_name: str = "totals_reliability_recalibration",
) -> Path:
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"dixon_coles": "#FF5252", "dixon_coles_v2_raw": "#FFB74D", "dixon_coles_v2": "#00B0FF"}
    ece_lines: list[str] = []
    # v2 solid first, then raw dashed on top so both stay visible when they overlap
    plot_order = ("dixon_coles", "dixon_coles_v2", "dixon_coles_v2_raw")

    for label in plot_order:
        df = series.get(label)
        if df is None or df.empty:
            continue
        col = "p_over_raw" if label.endswith("_raw") else "p_over"
        probs = df[col].to_numpy(dtype=float)
        outcomes = df["over_hit"].to_numpy(dtype=float)
        ece = binary_ece(probs, outcomes, bins=bins)
        ece_lines.append(f"{label} ECE={ece:.3f}")
        curve = totals_reliability_bins(probs, outcomes, bins=bins)
        if curve.empty:
            continue
        is_raw = label.endswith("_raw")
        ax.plot(
            curve["mean_pred"],
            curve["obs_freq"],
            "o-" if not is_raw else "o--",
            label=label,
            color=colors.get(label, None),
            linewidth=2.0 if is_raw else 1.5,
            zorder=3 if is_raw else 2,
        )

    ax.plot([0, 1], [0, 1], "--", color="#64748B", linewidth=1, label="perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Mean predicted P(over {line})")
    ax.set_ylabel("Observed over rate")
    ax.set_title(f"Totals reliability (over {line})")
    ax.legend(fontsize=8)
    style_figure(fig)
    style_axes(ax)
    fig.text(0.02, 0.02, "  |  ".join(ece_lines), fontsize=8, color="#64748B")
    path = save_figure(fig, out_name)
    plt.close(fig)
    return path


def _resolve_nl_swe_target(features: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    kickoff = pd.Timestamp(cfg.target_match.kickoff_utc).tz_convert("UTC")
    home = cfg.target_match.home.lower().replace(" ", "_")
    away = cfg.target_match.away.lower().replace(" ", "_")
    sub = features[
        (features["home_team_id"].astype(str) == home)
        & (features["away_team_id"].astype(str) == away)
        & (pd.to_datetime(features["date_utc"], utc=True) <= kickoff)
    ].sort_values(["date_utc", "match_id"], kind="mergesort")
    if sub.empty:
        raise ValueError(f"Target match not in features: {cfg.target_match.home} vs {cfg.target_match.away}")
    return sub.iloc[-1]


def spot_anchor_nl_swe(
    cfg: AppConfig | None = None,
    *,
    line: float = DEFAULT_TOTALS_LINE,
    market_over: float = NL_SWE_MARKET_OVER_25,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    features = load_model_features()
    kickoff = pd.Timestamp(cfg.target_match.kickoff_utc).tz_convert("UTC")
    train = features[
        features["result_1x2"].notna() & (pd.to_datetime(features["date_utc"], utc=True) < kickoff)
    ].reset_index(drop=True)
    target = _resolve_nl_swe_target(features, cfg)

    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()

    rows: list[dict] = []
    for name in ("dixon_coles", "dixon_coles_v2"):
        model = create_model(name, cfg, odds)
        model.fit(train)
        pred = model.predict(target)
        if pred.scoreline is None:
            continue
        raw = prob_over_from_scoreline(pred.scoreline, line)
        cal = raw
        getter = getattr(model, "totals_calibrators", None)
        if callable(getter) and line in getter():
            cal = float(getter()[line](raw))
        rows.append(
            {
                "model": name,
                "line": line,
                "p_over_raw": raw,
                "p_over": cal,
                "market_consensus": market_over,
                "gap_to_market_raw_pp": (raw - market_over) * 100.0,
                "gap_to_market_pp": (cal - market_over) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _load_or_empty(name: str) -> pd.DataFrame:
    try:
        return read_table(name, PROCESSED)
    except FileNotFoundError:
        return pd.DataFrame()


def run_recalibration_acceptance(
    cfg: AppConfig | None = None,
    *,
    refit_every: int | None = None,
    tail_matches: int | None = None,
    skip_walkforward: bool = False,
) -> dict[str, object]:
    cfg = cfg or get_config()
    set_seeds(cfg.seed)
    features = load_model_features()
    if tail_matches is not None:
        scored = features[features["result_1x2"].notna()].sort_values(
            ["date_utc", "match_id"], kind="mergesort"
        )
        features = pd.concat(
            [
                features[features["result_1x2"].isna()],
                scored.tail(int(tail_matches)),
            ],
            ignore_index=True,
        )
    line = DEFAULT_TOTALS_LINE
    bins = cfg.eval.calibration_bins
    refit = refit_every if refit_every is not None else cfg.model.refit_every

    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()

    logger.info("Running combined walk-forward for dixon_coles and dixon_coles_v2")
    dc_model = create_model("dixon_coles", cfg, odds)
    v2_model = create_model("dixon_coles_v2", cfg, odds)
    min_hist = cfg.eval.min_history_matches

    logger.info("Combined walk-forward (refit_every=%d, min_history=%d)", refit, min_hist)
    if skip_walkforward:
        dc_totals = _load_or_empty("totals_calibration_dixon_coles")
        v2_totals = _load_or_empty("totals_calibration_dixon_coles_v2")
        pred_frames = {
            "dixon_coles_v2": _load_or_empty("predictions_dixon_coles_v2"),
        }
    else:
        pred_frames, total_frames = walk_forward_combined(
            [dc_model, v2_model],
            features,
            line=line,
            min_history_matches=min_hist,
            refit_every=refit,
        )
        dc_totals = total_frames.get("dixon_coles", pd.DataFrame())
        v2_totals = total_frames.get("dixon_coles_v2", pd.DataFrame())

    v2_preds = pred_frames.get("dixon_coles_v2", pd.DataFrame())
    if not skip_walkforward and not v2_preds.empty and tail_matches is None:
        v2_preds = validate_table(v2_preds, "predictions")
        write_table(
            v2_preds,
            "predictions_dixon_coles_v2",
            PROCESSED,
            sort_by=["date_utc", "match_id"],
            plan=PLAN,
            schema_name="predictions",
        )
    elif skip_walkforward or tail_matches is not None:
        v2_preds = _load_or_empty("predictions_dixon_coles_v2")

    v2_raw = v2_totals.copy()
    if not v2_raw.empty:
        v2_raw["p_over"] = v2_raw["p_over_raw"]
    plot_totals_reliability(
        {
            "dixon_coles": dc_totals,
            "dixon_coles_v2_raw": v2_raw,
            "dixon_coles_v2": v2_totals,
        },
        line=line,
        bins=bins,
        out_name="totals_reliability_recalibration",
    )

    if not skip_walkforward:
        if not dc_totals.empty:
            write_table(
                dc_totals,
                "totals_calibration_dixon_coles",
                PROCESSED,
                sort_by=["date_utc", "match_id"],
                plan=PLAN,
            )
        if not v2_totals.empty:
            write_table(
                v2_totals,
                "totals_calibration_dixon_coles_v2",
                PROCESSED,
                sort_by=["date_utc", "match_id"],
                plan=PLAN,
            )

    try:
        v1_preds = read_table("predictions_dixon_coles", PROCESSED)
    except FileNotFoundError:
        v1_preds = pred_frames.get("dixon_coles", pd.DataFrame())

    common_ids = sorted(
        set(v1_preds["match_id"].astype(str)) & set(v2_preds["match_id"].astype(str))
    )
    v1_sub = v1_preds[v1_preds["match_id"].astype(str).isin(common_ids)].copy()
    v2_sub = v2_preds[v2_preds["match_id"].astype(str).isin(common_ids)].copy()
    v1_sub["model"] = "dixon_coles"
    v2_sub["model"] = "dixon_coles_v2"
    stacked = pd.concat([v1_sub, v2_sub], ignore_index=True)
    rps_scores = score_predictions(stacked, bootstrap_samples=cfg.eval.bootstrap_samples, seed=cfg.seed)
    rps_board = rps_scores[rps_scores["metric"] == "rps"]

    draw_v1 = draw_calibration_summary(v1_sub)
    draw_v2 = draw_calibration_summary(v2_sub)
    anchor = spot_anchor_nl_swe(cfg, line=line)

    summary = {
        "refit_every": refit,
        "totals_ece": {
            "dixon_coles": binary_ece(
                dc_totals["p_over"].to_numpy(), dc_totals["over_hit"].to_numpy(), bins=bins
            )
            if not dc_totals.empty
            else float("nan"),
            "dixon_coles_v2_raw": binary_ece(
                v2_totals["p_over_raw"].to_numpy(), v2_totals["over_hit"].to_numpy(), bins=bins
            )
            if not v2_totals.empty
            else float("nan"),
            "dixon_coles_v2": binary_ece(
                v2_totals["p_over"].to_numpy(), v2_totals["over_hit"].to_numpy(), bins=bins
            )
            if not v2_totals.empty
            else float("nan"),
        },
        "rps": rps_board.set_index("model").to_dict(orient="index") if not rps_board.empty else {},
        "draw_calibration": {"dixon_coles": draw_v1, "dixon_coles_v2": draw_v2},
        "spot_anchor": anchor.to_dict(orient="records"),
        "n_totals": {"dixon_coles": len(dc_totals), "dixon_coles_v2": len(v2_totals)},
    }

    summary_path = project_root() / REPORTS / "RECALIBRATION_ACCEPTANCE.md"
    _write_acceptance_md(summary, summary_path)
    logger.info("Acceptance summary -> %s", summary_path)
    return summary


def _write_acceptance_md(summary: dict[str, object], path: Path) -> None:
    ece = summary.get("totals_ece", {})
    rps = summary.get("rps", {})
    draw = summary.get("draw_calibration", {})
    anchor = summary.get("spot_anchor", [])

    lines = [
        "# Plan 14 recalibration acceptance",
        "",
        f"Walk-forward refit_every={summary.get('refit_every', '?')} (v1 RPS from existing predictions_dixon_coles at refit_every=10).",
        "",
        "## Totals reliability (over 2.5)",
        "",
        f"Figure: [`figures/totals_reliability_recalibration.png`](figures/totals_reliability_recalibration.png)",
        "",
        "| Model | ECE |",
        "|---|---|",
    ]
    for name, val in ece.items():
        lines.append(f"| {name} | {val:.4f} |" if isinstance(val, float) and not math.isnan(val) else f"| {name} | — |")

    lines.extend(["", "## 1X2 RPS (common match set)", "", "| Model | RPS | CI |", "|---|---|---|"])
    for model, row in rps.items():
        lines.append(f"| {model} | {row['mean']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")

    lines.extend(["", "## Draw rate (aggregate)", "", "| Model | Pred draw | Actual draw | Gap (pp) |", "|---|---|---|---|"])
    for model, row in draw.items():
        lines.append(
            f"| {model} | {row['pred_draw']:.3f} | {row['actual_draw']:.3f} | {row['gap_pp']:+.1f} |"
        )

    lines.extend(["", "## NL-SWE spot anchor (over 2.5)", "", "| Model | Raw P(over) | Cal P(over) | Market | Gap raw (pp) | Gap cal (pp) |", "|---|---|---|---|---|---|"])
    for row in anchor:
        lines.append(
            f"| {row['model']} | {row['p_over_raw']:.3f} | {row['p_over']:.3f} | {row['market_consensus']:.3f} | {row['gap_to_market_raw_pp']:+.1f} | {row['gap_to_market_pp']:+.1f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan 14 goal-model recalibration acceptance")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=None,
        help="Walk-forward refit interval (default: config; 100 is much faster on full history)",
    )
    parser.add_argument(
        "--tail-matches",
        type=int,
        default=None,
        help="If set, only walk-forward the last N scored internationals (faster totals pass)",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "walkforward", "report", "compare"),
        default="all",
        help="walkforward: backtest+totals only; report: plots+md from disk; compare: leaderboard",
    )
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    refit = args.refit_every or cfg.model.refit_every

    if args.phase in ("all", "walkforward"):
        run_recalibration_acceptance(
            cfg,
            refit_every=refit,
            tail_matches=args.tail_matches,
            skip_walkforward=False,
        )
    if args.phase in ("all", "report"):
        run_recalibration_acceptance(
            cfg,
            refit_every=refit,
            skip_walkforward=True,
        )
    if args.phase in ("all", "compare"):
        from nlvswe.eval.compare import run_compare

        run_compare(cfg)


if __name__ == "__main__":
    main()
