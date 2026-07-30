# Cumulative live bet ledger: append-on-settle, track-record figures and summary

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.betting.clv import aggregate_clv
from nlvswe.config import AppConfig, get_config, project_root
from nlvswe.eval.scoring import bootstrap_ci
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "16"
LIVE_SUBDIR = "processed/live"
REPORTS = "reports"

_BET_KEY = ("match_id", "bookmaker", "market", "selection")


def bet_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row["match_id"]),
        str(row.get("bookmaker") or ""),
        str(row.get("market") or ""),
        str(row.get("selection") or ""),
    )


def _load_ledger() -> pd.DataFrame:
    try:
        return read_table("ledger", LIVE_SUBDIR)
    except FileNotFoundError:
        return pd.DataFrame()


def _prepare_ledger_rows(results: pd.DataFrame, *, settled_at: pd.Timestamp) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    rows = results.copy()
    rows["settled_at"] = settled_at
    rows["settled_at"] = pd.to_datetime(rows["settled_at"], utc=True).astype("datetime64[ns, UTC]")
    rows["kickoff_utc"] = pd.to_datetime(rows["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    rows["captured_at"] = pd.to_datetime(rows["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    if "closing_captured_at" in rows.columns:
        rows["closing_captured_at"] = pd.to_datetime(rows["closing_captured_at"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    rows["measurable"] = rows["measurable"].astype("bool")
    for col in ("home_goals", "away_goals"):
        rows[col] = rows[col].astype("Int64")
    return rows


def _sort_ledger(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = ["kickoff_utc", "match_id", "market", "selection", "bookmaker"]
    present = [c for c in sort_cols if c in df.columns]
    return df.sort_values(present, kind="mergesort").reset_index(drop=True)


def _recompute_running_totals(ledger: pd.DataFrame, *, initial_bankroll: float) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = _sort_ledger(ledger)
    out["bet_seq"] = np.arange(1, len(out) + 1, dtype=int)
    out["cumulative_pnl"] = out["pnl"].astype(float).cumsum()
    out["bankroll"] = float(initial_bankroll) + out["cumulative_pnl"]
    return out


def append_to_ledger(
    results: pd.DataFrame,
    *,
    cfg: AppConfig | None = None,
    settled_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Merge settled bets into the cumulative ledger (replace same bet key if re-settled)."""
    cfg = cfg or get_config()
    if results.empty:
        return _load_ledger()

    ts = settled_at or pd.Timestamp(datetime.now(timezone.utc))
    incoming = _prepare_ledger_rows(results, settled_at=ts)
    existing = _load_ledger()

    if existing.empty:
        merged = incoming
    else:
        existing = existing.drop(columns=["bet_seq", "cumulative_pnl", "bankroll"], errors="ignore")
        incoming_keys = {bet_key(r) for _, r in incoming.iterrows()}
        keep = existing[~existing.apply(bet_key, axis=1).isin(incoming_keys)]
        merged = pd.concat([keep, incoming], ignore_index=True)

    ledger = _recompute_running_totals(merged, initial_bankroll=cfg.betting.bankroll)
    ledger = validate_table(ledger, "live_ledger")
    write_table(
        ledger,
        "ledger",
        LIVE_SUBDIR,
        sort_by=["bet_seq"],
        plan=PLAN,
        schema_name="live_ledger",
    )
    logger.info("Ledger updated: %d bets, cumulative P&L=%.2f", len(ledger), float(ledger["pnl"].sum()))
    return ledger


def ledger_summary(ledger: pd.DataFrame, *, cfg: AppConfig | None = None) -> dict:
    cfg = cfg or get_config()
    if ledger.empty:
        return {
            "n_bets": 0,
            "n_measurable_clv": 0,
            "total_staked": 0.0,
            "total_pnl": 0.0,
            "bankroll": float(cfg.betting.bankroll),
            **aggregate_clv(ledger),
        }
    clv = aggregate_clv(ledger)
    return {
        "n_bets": int(len(ledger)),
        "n_measurable_clv": int(clv["n_with_clv"]),
        "total_staked": float(ledger["stake"].sum()),
        "total_pnl": float(ledger["pnl"].sum()),
        "bankroll": float(ledger["bankroll"].iloc[-1]),
        **clv,
    }


def _cumulative_clv_with_ci(
    ledger: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measurable = ledger[ledger["measurable"]].copy()
    if measurable.empty:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    clv = measurable["clv_odds_ratio"].astype(float).to_numpy()
    steps = np.arange(1, len(clv) + 1)
    means = np.empty(len(clv), dtype=float)
    lo = np.empty(len(clv), dtype=float)
    hi = np.empty(len(clv), dtype=float)
    rng = np.random.default_rng(seed)
    for i in range(len(clv)):
        prefix = clv[: i + 1]
        means[i] = prefix.mean()
        if len(prefix) < 2:
            lo[i] = means[i]
            hi[i] = means[i]
            continue
        boot = np.empty(bootstrap_samples)
        n = len(prefix)
        for j in range(bootstrap_samples):
            boot[j] = prefix[rng.integers(0, n, size=n)].mean()
        lo[i] = float(np.quantile(boot, 0.025))
        hi[i] = float(np.quantile(boot, 0.975))
    # map measurable steps back to overall bet_seq for plotting
    bet_seq = measurable["bet_seq"].to_numpy(dtype=int)
    return bet_seq, means, lo, hi


def plot_track_record(
    ledger: pd.DataFrame,
    *,
    cfg: AppConfig | None = None,
    out_name: str = "live_track_record",
) -> None:
    cfg = cfg or get_config()
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if ledger.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No settled bets yet", ha="center", va="center", transform=ax.transAxes)
            style_axes(ax)
        fig.suptitle("Live track record")
        style_figure(fig)
        save_figure(fig, out_name)
        plt.close(fig)
        return

    steps = ledger["bet_seq"].to_numpy()
    axes[0].plot(steps, ledger["bankroll"], color="#00E676", marker="o", linewidth=2, markersize=4)
    axes[0].axhline(cfg.betting.bankroll, color="#64748B", linestyle="--", linewidth=1, label="Start bankroll")
    axes[0].set_xlabel("Bet number")
    axes[0].set_ylabel("Bankroll ($)")
    axes[0].set_title(f"Running bankroll ({len(ledger)} bets)")
    axes[0].legend(fontsize=8)
    style_axes(axes[0])

    seq, means, lo, hi = _cumulative_clv_with_ci(
        ledger,
        bootstrap_samples=cfg.eval.bootstrap_samples,
        seed=cfg.seed,
    )
    if seq.size:
        axes[1].plot(seq, means, color="#00B0FF", marker="o", linewidth=2, markersize=4, label="Mean CLV")
        axes[1].fill_between(seq, lo, hi, color="#00B0FF", alpha=0.2, label="95% bootstrap CI")
        axes[1].axhline(0, color="#64748B", linestyle="--", linewidth=1)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No measurable CLV yet\n(single-snapshot fixtures excluded)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=9,
        )
    axes[1].set_xlabel("Bet number (measurable CLV only)")
    axes[1].set_ylabel("Cumulative mean CLV (odds ratio)")
    axes[1].set_title("Cumulative CLV")
    if seq.size:
        axes[1].legend(fontsize=8)
    style_axes(axes[1])

    fig.suptitle("Live track record")
    style_figure(fig)
    save_figure(fig, out_name)
    plt.close(fig)


def write_track_record_md(
    ledger: pd.DataFrame,
    *,
    cfg: AppConfig | None = None,
    path: Path | None = None,
) -> Path:
    cfg = cfg or get_config()
    summary = ledger_summary(ledger, cfg=cfg)
    path = path or (project_root() / REPORTS / "TRACK_RECORD.md")

    lines = [
        "# Live track record",
        "",
        f"**Updated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"**Ledger rows:** {summary['n_bets']}",
        "",
        "## Summary",
        "",
        f"| Bets | Measurable CLV | Total staked | Total P&L | Bankroll |",
        "|---:|---:|---:|---:|---:|",
        f"| {summary['n_bets']} | {summary['n_measurable_clv']} | "
        f"${summary['total_staked']:.2f} | ${summary['total_pnl']:+.2f} | ${summary['bankroll']:.2f} |",
        "",
    ]

    if summary["n_measurable_clv"] > 0:
        lines.extend(
            [
                "### CLV (measurable bets only)",
                "",
                f"- Mean CLV (odds ratio): {summary['mean_clv_odds_ratio']:.4f}",
                f"- Median CLV (odds ratio): {summary['median_clv_odds_ratio']:.4f}",
                f"- % positive CLV: {summary['pct_positive_clv']:.1%}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### CLV",
                "",
                "No measurable CLV yet. Single pre-kickoff snapshots are recorded with "
                "`measurable=False` rather than CLV=0.",
                "",
            ]
        )

    if not ledger.empty:
        lines.extend(["## Settled bets", "", "| # | Match | Market | Selection | Stake | P&L | CLV | Measurable |"])
        lines.append("|---:|---|---|---|---:|---:|---:|:---:|")
        for row in ledger.itertuples():
            clv = row.clv_odds_ratio
            clv_str = f"{clv:.3f}" if pd.notna(clv) and row.measurable else "n/a"
            lines.append(
                f"| {row.bet_seq} | `{row.match_id}` | {row.market} | {row.selection} | "
                f"${row.stake:.2f} | ${row.pnl:+.2f} | {clv_str} | {row.measurable} |"
            )
        lines.append("")

    lines.extend(
        [
            "## What this ledger records",
            "",
            "Only the **machine-generated** bet slips produced by `live.predict`, which is",
            "what the automated strategy actually pre-registered. Discretionary bets a human",
            "added by hand are deliberately excluded, so the track record measures the model",
            "rather than the operator.",
            "",
            "This is why the NL vs SWE row set differs from `reports/BETS.md` and the P&L",
            "figure in `reports/REPORT.md` ($36.94). That writeup settles the human bet",
            "ticket: one Netherlands win at 1.74 for $31, plus a speculative over 2.5 for $20",
            "that the model itself had flagged as a likely losing bet. The frozen machine slip",
            "instead held two Netherlands win bets, at Pinnacle 1.74 and William Hill 1.73,",
            "and no totals bet at all (totals markets were disabled in config). Both numbers",
            "are correct for what they describe; neither restates the other.",
            "",
            "## Artifacts",
            "",
            "- Ledger: `data/processed/live/ledger.parquet`",
            "- Figure: `reports/figures/live_track_record.png`",
            "- Per-match results: `data/processed/live/results.parquet`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def refresh_track_record(
    ledger: pd.DataFrame | None = None,
    *,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    ledger = ledger if ledger is not None else _load_ledger()
    plot_track_record(ledger, cfg=cfg)
    write_track_record_md(ledger, cfg=cfg)
    return ledger
